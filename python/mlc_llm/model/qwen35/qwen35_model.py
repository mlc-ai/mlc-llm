"""
Implementation for Qwen3.5 GatedDeltaNet hybrid architecture.
75% GatedDeltaNet (recurrent linear attention), 25% standard GQA softmax attention.
"""

import dataclasses
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tvm
from tvm import te, tir
from tvm import relax as R
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Qwen35Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Qwen3.5 model."""

    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_hidden_layers: int = 0
    num_key_value_heads: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    rope_theta: float = 10000000.0
    head_dim: int = 256
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # GatedDeltaNet-specific
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    # Runtime
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float32"
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Handle VLM wrapper: Qwen3.5 HF config has all text params inside text_config
        if "text_config" in self.kwargs:
            text_config = self.kwargs.pop("text_config")
            if isinstance(text_config, dict):
                field_names = {f.name for f in dataclasses.fields(self.__class__)}
                for k, v in text_config.items():
                    if k in field_names and k != "kwargs":
                        setattr(self, k, v)
                    else:
                        self.kwargs[k] = v
                # Extract rope params from nested rope_parameters
                rope_params = text_config.get("rope_parameters", {})
                if isinstance(rope_params, dict):
                    if "rope_theta" in rope_params:
                        self.rope_theta = rope_params["rope_theta"]
                    if "partial_rotary_factor" in rope_params:
                        self.partial_rotary_factor = rope_params["partial_rotary_factor"]

        # Also handle rope_parameters at top level
        if "rope_parameters" in self.kwargs:
            rope_params = self.kwargs.pop("rope_parameters")
            if isinstance(rope_params, dict):
                if "rope_theta" in rope_params:
                    self.rope_theta = rope_params["rope_theta"]
                if "partial_rotary_factor" in rope_params:
                    self.partial_rotary_factor = rope_params["partial_rotary_factor"]

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 2048)

    @property
    def num_linear_layers(self) -> int:
        """Number of GatedDeltaNet linear attention layers."""
        return self.num_hidden_layers - self.num_attention_layers

    @property
    def num_attention_layers(self) -> int:
        """Number of full attention layers."""
        return self.num_hidden_layers // self.full_attention_interval

    def layer_types(self) -> List[str]:
        """Returns list of layer types: 'linear_attention' or 'full_attention'."""
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types


# pylint: disable=invalid-name,missing-docstring,too-many-locals


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
}


class Qwen35Embedding(nn.Embedding):
    def lm_head_forward(self, x: nn.Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen35Attention(nn.Module):
    """Standard GQA attention with output gate for full_attention layers (every 4th layer).

    attn_output_gate=True: q_proj outputs 2*num_heads*head_dim, split into (Q, gate).
    Gate is sigmoid-applied to attention output before o_proj.
    """

    def __init__(self, config: Qwen35Config):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.rope_theta = config.rope_theta

        # c_attn: Q (2x for gate) + K + V fused projection
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(2 * self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)
        self.k_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        # c_attn outputs flat: [Q_with_gate (h_q * 2 * d), K (h_kv * d), V (h_kv * d)]
        proj = self.c_attn(hidden_states)
        # Reshape to heads: (b, s, 2*h_q + 2*h_kv, d)
        proj = op.reshape(proj, (b, s, 2 * h_q + 2 * h_kv, d))
        # Split: first 2*h_q heads have interleaved [Q, gate] per head, then h_kv K, h_kv V
        q_gate, k, v = op.split(proj, [2 * h_q, 2 * h_q + h_kv], axis=2)
        # q_gate shape: (b, s, 2*h_q, d). Even heads are Q, odd heads are gate
        # But HF layout is per-head [Q_d, gate_d], so reshape to (b, s, h_q, 2*d) then split
        q_gate = op.reshape(q_gate, (b, s, h_q, 2 * d))
        q, gate = op.split(q_gate, [d], axis=3)
        # gate: (b, s, h_q, d) -> flatten to (b, s, h_q*d)
        gate = op.reshape(gate, (b, s, h_q * d))
        q = self.q_norm(q)
        k = self.k_norm(k)
        qkv = op.concat([q, k, v], dim=2)
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        # Apply output gate: sigmoid(gate) * attn_output
        output = output * op.sigmoid(gate)
        return self.o_proj(output)


# ============================================================================
# GatedDeltaNet TIR kernel
# ============================================================================


def create_gated_delta_net_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a TIR function for the GatedDeltaNet recurrent computation.

    Thread-per-column design: each thread owns one column of the state matrix.
    State S is (key_head_dim x value_head_dim) per head, accumulated in fp32.

    Supports arbitrary sequence length via an inner `for t in range(seq_len)` loop,
    matching RWKV6's approach. During prefill (seq_len > 1), the recurrence accumulates
    state across all tokens sequentially. During decode (seq_len = 1), it's a single step.

    For GVA (num_value_heads > num_key_heads), Q/K are expanded via repeat.
    The kernel operates on value_heads (the larger dimension).
    """
    heads_per_group = num_value_heads // num_key_heads  # 1 for 0.8B, 2 for 4B
    K = key_head_dim  # 128
    V = value_head_dim  # 128

    @T.prim_func
    def gdn_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        # q, k: (batch, seq_len, key_heads, K)
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        # v: (batch, seq_len, value_heads, V)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        # gate and beta: (batch, seq_len, value_heads)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        # State: per value_head, K x V matrix in fp32
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        # Outputs: out in fp32 for numerical stability (cast to model dtype by caller)
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )
        state_out_buf = T.match_buffer(
            state_out_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    kh = h_idx // heads_per_group

                    # Init state from state_in
                    for row in range(K):
                        with T.sblock("init_state"):
                            vb, vh, vr, vc = T.axis.remap("SSSS", [b_idx, h_idx, row, col])
                            state_out_buf[vb, vh, vr, vc] = state_in_buf[vb, vh, vr, vc]

                    # Sequential loop over tokens (like RWKV6)
                    for t in range(seq_len):
                        # 1. Decay state: S = gate * S
                        for row in range(K):
                            with T.sblock("decay"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vr = T.axis.opaque(K, row)
                                vc = T.axis.spatial(V, col)
                                state_out_buf[vb, vh, vr, vc] = (
                                    state_out_buf[vb, vh, vr, vc] * gate_buf[vb, vt, vh]
                                )

                        # 2. Compute dot(S[:, col], k[:]) → out_buf (fp32)
                        with T.sblock("dot_sk_init"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = T.float32(0)

                        for row in range(K):
                            with T.sblock("dot_sk"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] + state_out_buf[
                                    vb, vh, vr, vc
                                ] * T.cast(k_buf[vb, vt, kh, vr], "float32")

                        # 3. Delta rule: S += k * beta * (v - dot_sk)
                        for row in range(K):
                            with T.sblock("delta"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                state_out_buf[vb, vh, vr, vc] = state_out_buf[
                                    vb, vh, vr, vc
                                ] + T.cast(k_buf[vb, vt, kh, vr], "float32") * beta_buf[
                                    vb, vt, vh
                                ] * (
                                    T.cast(v_buf[vb, vt, vh, vc], "float32")
                                    - out_buf[vb, vt, vh, vc]
                                )

                        # 4. Output: o[t, col] = dot(S_updated[:, col], q[t, :]) * scale
                        with T.sblock("out_init"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = T.float32(0)

                        for row in range(K):
                            with T.sblock("dot_sq"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] + state_out_buf[
                                    vb, vh, vr, vc
                                ] * T.cast(q_buf[vb, vt, kh, vr], "float32")

                        # 5. Apply scale
                        with T.sblock("scale"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] * T.float32(
                                1.0 / math.sqrt(K)
                            )

    return gdn_func


# ============================================================================
# GatedDeltaNet Linear Attention Layer
# ============================================================================


def _te_scatter_layer(stacked: te.Tensor, layer_data: te.Tensor, layer_idx: int):
    """Scatter a single layer's data into a stacked tensor at the given index.

    stacked: (num_layers, ...) — the full stacked tensor
    layer_data: (...) — data for one layer
    Returns a new tensor where stacked[layer_idx] = layer_data, rest unchanged.
    """
    # We need to handle arbitrary trailing dims. For state: (n_layers, b, h, K, V)
    # For conv: (n_layers, b, kernel-1, d)
    shape = stacked.shape
    ndim = len(shape)

    if ndim == 5:
        return te.compute(
            shape,
            lambda i0, i1, i2, i3, i4: tir.if_then_else(
                i0 == layer_idx,
                layer_data[i1, i2, i3, i4],
                stacked[i0, i1, i2, i3, i4],
            ),
            name="scatter_layer",
        )
    elif ndim == 4:
        return te.compute(
            shape,
            lambda i0, i1, i2, i3: tir.if_then_else(
                i0 == layer_idx,
                layer_data[i1, i2, i3],
                stacked[i0, i1, i2, i3],
            ),
            name="scatter_layer",
        )
    else:
        raise ValueError(f"Unsupported ndim={ndim} for scatter_layer")


def _te_extract_layer_5d(stacked: te.Tensor, layer_idx: int):
    """Extract stacked[layer_idx] from a 5D tensor (n_layers, b, h, K, V) → (b, h, K, V)."""
    shape = stacked.shape
    return te.compute(
        (shape[1], shape[2], shape[3], shape[4]),
        lambda i1, i2, i3, i4: stacked[layer_idx, i1, i2, i3, i4],
        name="extract_layer",
    )


def _te_extract_layer_4d(stacked: te.Tensor, layer_idx: int):
    """Extract stacked[layer_idx] from a 4D tensor (n_layers, b, k, d) → (b, k, d)."""
    shape = stacked.shape
    return te.compute(
        (shape[1], shape[2], shape[3]),
        lambda i1, i2, i3: stacked[layer_idx, i1, i2, i3],
        name="extract_layer",
    )


class Qwen35GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention layer."""

    def __init__(self, config: Qwen35Config, linear_layer_idx: int):
        self.config = config
        self.linear_layer_idx = linear_layer_idx  # index among linear layers only
        self.key_head_dim = config.linear_key_head_dim  # 128
        self.value_head_dim = config.linear_value_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 16 or 32
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype

        qkv_dim = (
            (self.num_key_heads * self.key_head_dim)
            + (self.num_key_heads * self.key_head_dim)
            + (self.num_value_heads * self.value_head_dim)
        )

        # Projections — matching HF weight names
        self.in_proj_qkv = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(
            config.hidden_size, self.num_value_heads * self.value_head_dim, bias=False
        )
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(
            self.num_value_heads * self.value_head_dim, config.hidden_size, bias=False
        )

        # Causal depthwise Conv1D kernel
        self.conv1d_weight = nn.Parameter(
            (qkv_dim, 1, config.linear_conv_kernel_dim),
        )

        # Decay parameters (no .weight suffix in HF)
        self.A_log = nn.Parameter((self.num_value_heads,))
        self.dt_bias = nn.Parameter((self.num_value_heads,))

        # Output gating norm — per-head RMSNorm (shared weight across heads)
        self.norm = nn.RMSNorm(self.value_head_dim, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        gdn_state_in: Tensor,
        gdn_state_out: Tensor,
        gdn_conv_in: Tensor,
        gdn_conv_out: Tensor,
    ) -> Tensor:
        b, s, _ = hidden_states.shape
        K = self.key_head_dim
        V = self.value_head_dim
        n_kh = self.num_key_heads
        n_vh = self.num_value_heads
        layer_idx = self.linear_layer_idx

        # Input projections
        qkv = self.in_proj_qkv(hidden_states)  # (b, s, qkv_dim)
        z = self.in_proj_z(hidden_states)  # (b, s, n_vh * V)
        alpha = self.in_proj_a(hidden_states)  # (b, s, n_vh)
        beta_raw = self.in_proj_b(hidden_states)  # (b, s, n_vh)

        # Causal Conv1D on QKV
        # For decode (s=1): gather last (kernel_size-1) cached tokens + current
        # Conv state shape: (b, kernel_size-1, qkv_dim) per layer
        qkv, gdn_conv_out = self._causal_conv1d(qkv, gdn_conv_in, gdn_conv_out, layer_idx)

        # SiLU activation on QKV after conv
        qkv = op.silu(qkv)

        # Split QKV — use op.split along last dim
        q_dim = n_kh * K
        k_dim = n_kh * K
        v_dim = n_vh * V
        qkv_parts = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        q = op.reshape(qkv_parts[0], (b, s, n_kh, K))
        k = op.reshape(qkv_parts[1], (b, s, n_kh, K))
        v = op.reshape(qkv_parts[2], (b, s, n_vh, V))

        # L2 normalize Q and K
        q = self._l2_normalize(q)
        k = self._l2_normalize(k)

        # Gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
        # beta = sigmoid(beta_raw)
        # We compute exp(g) for the kernel
        gate, beta = self._compute_gate_beta(alpha, beta_raw)

        # Note: beta is already (b, s, n_vh) since in_proj_b outputs num_value_heads.
        # No GVA expansion needed for beta.

        # Extract this layer's state from the stacked state tensor
        # gdn_state_in shape: (num_linear_layers, b, n_vh, K, V)
        # We need: (b, n_vh, K, V)
        # op_pattern=8 (kOpaque) prevents fusion with upstream ops to stay under
        # WebGPU's 10 storage buffers per shader limit.
        state_in_layer = op.tensor_expr_op(
            lambda s: _te_extract_layer_5d(s, layer_idx),
            "extract_state",
            [gdn_state_in],
            attrs={"op_pattern": 8},
        )

        # Recurrent computation via TIR kernel — handles full sequence length.
        # q, k: (b, s, n_kh, K), v: (b, s, n_vh, V), gate/beta: (b, s, n_vh)
        out_recurrent, state_out_layer = op.tensor_ir_op(
            create_gated_delta_net_func(
                num_key_heads=n_kh,
                num_value_heads=n_vh,
                key_head_dim=K,
                value_head_dim=V,
                dtype=self.dtype,
            ),
            "gated_delta_net",
            [q, k, v, gate, beta, state_in_layer],
            [
                Tensor.placeholder([b, s, n_vh, V], "float32"),
                Tensor.placeholder([b, n_vh, K, V], "float32"),
            ],
        )

        # Cast recurrent output back to model dtype
        out_recurrent = op.astype(out_recurrent, self.dtype)

        # Write updated state back
        # state_out_layer: (b, n_vh, K, V) → needs to go into gdn_state_out[layer_idx]
        # We handle this via a te.compute to scatter into the right slot
        gdn_state_out = self._scatter_state(gdn_state_out, state_out_layer, layer_idx)

        # Output gating: per-head RMSNorm(out) * SiLU(z)
        # norm weight is (V,), applied per-head
        # out_recurrent is already (b, s, n_vh, V)
        out_normed = self.norm(out_recurrent)
        out_flat = op.reshape(out_normed, (b, s, n_vh * V))
        out_gated = out_flat * op.silu(z)
        return self.out_proj(out_gated), gdn_state_out, gdn_conv_out

    def forward_rnn(self, hidden_states: Tensor, state: RNNState) -> Tuple[Tensor, RNNState]:
        """Forward using RNNState (for MLCEngine batch methods)."""
        b, s, _ = hidden_states.shape
        K = self.key_head_dim
        V = self.value_head_dim
        n_kh = self.num_key_heads
        n_vh = self.num_value_heads
        layer_idx = self.linear_layer_idx

        # Input projections
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        alpha = self.in_proj_a(hidden_states)
        beta_raw = self.in_proj_b(hidden_states)

        # Get conv state from RNNState (state_id=1)
        qkv_dim = qkv.shape[-1]
        conv_state = state.get(
            layer_idx,
            1,
            (b, self.config.linear_conv_kernel_dim - 1, qkv_dim),
            self.dtype,
        )

        # Causal Conv1D using existing helper logic
        qkv, new_conv_state = self._causal_conv1d_with_state(qkv, conv_state)
        state = state.set(layer_idx, 1, new_conv_state)

        # SiLU activation on QKV after conv
        qkv = op.silu(qkv)

        # Split QKV
        q_dim = n_kh * K
        k_dim = n_kh * K
        qkv_parts = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        q = op.reshape(qkv_parts[0], (b, s, n_kh, K))
        k = op.reshape(qkv_parts[1], (b, s, n_kh, K))
        v = op.reshape(qkv_parts[2], (b, s, n_vh, V))

        # L2 normalize Q and K
        q = self._l2_normalize(q)
        k = self._l2_normalize(k)

        # Gate computation
        gate, beta = self._compute_gate_beta(alpha, beta_raw)
        # beta is already (b, s, n_vh) — no GVA expansion needed.

        # Get recurrent state from RNNState (state_id=0)
        state_in_layer = state.get(layer_idx, 0, (b, n_vh, K, V), "float32")

        # Recurrent computation via TIR kernel
        out_recurrent, state_out_layer = op.tensor_ir_op(
            create_gated_delta_net_func(
                num_key_heads=n_kh,
                num_value_heads=n_vh,
                key_head_dim=K,
                value_head_dim=V,
                dtype=self.dtype,
            ),
            "gated_delta_net",
            [q, k, v, gate, beta, state_in_layer],
            [
                Tensor.placeholder([b, s, n_vh, V], "float32"),
                Tensor.placeholder([b, n_vh, K, V], "float32"),
            ],
        )

        # Cast recurrent output back to model dtype
        out_recurrent = op.astype(out_recurrent, self.dtype)

        # Write updated state back to RNNState (state_id=0)
        state = state.set(layer_idx, 0, state_out_layer)

        # Output gating
        out_normed = self.norm(out_recurrent)
        out_flat = op.reshape(out_normed, (b, s, n_vh * V))
        out_gated = out_flat * op.silu(z)
        return self.out_proj(out_gated), state

    def _causal_conv1d_with_state(self, qkv: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Causal Conv1D using a pre-extracted conv_state tensor (for RNNState path)."""
        b, s, d = qkv.shape
        kernel_size = self.config.linear_conv_kernel_dim

        # Update conv state
        def _te_update_conv_state(old_state: te.Tensor, qkv_in: te.Tensor):
            ks_minus_1 = old_state.shape[1]
            seq = qkv_in.shape[1]
            return te.compute(
                old_state.shape,
                lambda bi, ti, di: tir.if_then_else(
                    seq + ti < ks_minus_1,
                    old_state[bi, seq + ti, di],
                    qkv_in[bi, seq + ti - ks_minus_1, di],
                ),
                name="update_conv_state",
            )

        new_conv_state = op.tensor_expr_op(
            _te_update_conv_state, "update_conv_state", [conv_state, qkv]
        )

        # Depthwise conv
        def _te_depthwise_conv(state: te.Tensor, qkv_in: te.Tensor, weight: te.Tensor):
            ks_m1 = state.shape[1]
            seq = qkv_in.shape[1]
            kk = te.reduce_axis((0, kernel_size), name="kk")
            return te.compute(
                (qkv_in.shape[0], seq, qkv_in.shape[2]),
                lambda bi, si, di: te.sum(
                    tir.if_then_else(
                        si + kk < ks_m1,
                        state[bi, si + kk, di],
                        qkv_in[bi, si + kk - ks_m1, di],
                    )
                    * weight[di, 0, kk],
                    axis=kk,
                ),
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _te_depthwise_conv,
            "depthwise_conv1d",
            [conv_state, qkv, self.conv1d_weight],
            attrs={"op_pattern": 8},
        )
        return result, new_conv_state

    def _causal_conv1d(
        self, qkv: Tensor, conv_in: Tensor, conv_out: Tensor, layer_idx: int
    ) -> Tensor:
        """Causal depthwise Conv1D with state caching.

        For decode (s=1): uses cached state + current token.
        conv_in shape: (num_linear_layers, b, kernel_size-1, qkv_dim)
        """
        b, s, d = qkv.shape
        kernel_size = self.config.linear_conv_kernel_dim  # 4

        # Extract this layer's conv state
        # conv_in[layer_idx]: (b, kernel_size-1, d)
        conv_state = op.tensor_expr_op(
            lambda s: _te_extract_layer_4d(s, layer_idx),
            "extract_conv_state",
            [conv_in],
            attrs={"op_pattern": 8},
        )

        # Update conv state: store last kernel_size-1 tokens from combined [old_state, qkv]
        def _te_update_conv_state(old_state: te.Tensor, qkv_in: te.Tensor):
            ks_minus_1 = old_state.shape[1]  # kernel_size - 1
            seq = qkv_in.shape[1]
            # Combined window = [old_state[0..ks_m1-1], qkv[0..seq-1]]
            # We want the last ks_m1 entries, i.e. positions [seq+ti] in the combined window
            return te.compute(
                old_state.shape,
                lambda bi, ti, di: tir.if_then_else(
                    seq + ti < ks_minus_1,
                    old_state[bi, seq + ti, di],
                    qkv_in[bi, seq + ti - ks_minus_1, di],
                ),
                name="update_conv_state",
            )

        new_conv_state = op.tensor_expr_op(
            _te_update_conv_state, "update_conv_state", [conv_state, qkv]
        )
        conv_out = self._scatter_conv_state(conv_out, new_conv_state, layer_idx)

        # Depthwise conv: compute directly from conv_state and qkv without concat
        # For decode (s=1): window = [state[0], state[1], state[2], qkv[0]]
        # For general s: window[pos] = state[pos] if pos < ks-1 else qkv[pos - (ks-1)]
        def _te_depthwise_conv(state: te.Tensor, qkv_in: te.Tensor, weight: te.Tensor):
            # state: (b, ks-1, d), qkv_in: (b, s, d), weight: (d, 1, ks)
            ks_m1 = state.shape[1]  # kernel_size - 1
            seq = qkv_in.shape[1]
            kk = te.reduce_axis((0, kernel_size), name="kk")
            # window[si+kk] = state[si+kk] if si+kk < ks_m1 else qkv_in[si+kk-ks_m1]
            return te.compute(
                (qkv_in.shape[0], seq, qkv_in.shape[2]),
                lambda bi, si, di: te.sum(
                    tir.if_then_else(
                        si + kk < ks_m1,
                        state[bi, si + kk, di],
                        qkv_in[bi, si + kk - ks_m1, di],
                    )
                    * weight[di, 0, kk],
                    axis=kk,
                ),
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _te_depthwise_conv,
            "depthwise_conv1d",
            [conv_state, qkv, self.conv1d_weight],
            attrs={"op_pattern": 8},
        )
        return result, conv_out

    def _l2_normalize(self, x: Tensor) -> Tensor:
        """L2 normalize along last dimension with eps=1e-6."""
        # x: (b, s, h, d) — compute in float32 for numerical stability
        x_f32 = op.astype(x, "float32")
        x_sq = x_f32 * x_f32
        sum_sq = op.sum(x_sq, axis=-1, keepdims=True)  # (b, s, h, 1)
        inv_norm = op.sqrt(sum_sq + 1e-6)
        return op.astype(x_f32 / inv_norm, self.dtype)

    def _compute_gate_beta(self, alpha: Tensor, beta_raw: Tensor):
        """Compute decay gate and update rate.

        gate = exp(-exp(A_log) * softplus(alpha + dt_bias))  (per value_head)
        beta = sigmoid(beta_raw)  (per value_head)
        """

        # alpha: (b, s, n_vh), dt_bias: (n_vh,), A_log: (n_vh,)
        def _te_gate(alpha: te.Tensor, A_log: te.Tensor, dt_bias: te.Tensor):
            b, s, h = alpha.shape

            def _softplus(x):
                # softplus(x) = x if x > 20 else log(1 + exp(x))
                return tir.if_then_else(x > 20.0, x, tir.log(1.0 + tir.exp(x)))

            return te.compute(
                (b, s, h),
                lambda bi, si, hi: tir.exp(
                    -tir.exp(A_log[hi].astype("float32"))
                    * _softplus((alpha[bi, si, hi] + dt_bias[hi]).astype("float32"))
                ),
                name="gate",
            )

        gate = op.tensor_expr_op(
            _te_gate,
            "gate",
            [alpha, self.A_log, self.dt_bias],
            attrs={"op_pattern": 8},
        )

        beta = op.sigmoid(beta_raw).astype("float32")
        return gate, beta

    def _repeat_interleave(self, x: Tensor, repeats: int, axis: int) -> Tensor:
        """Repeat interleave along given axis (for GVA expansion)."""
        if repeats == 1:
            return x
        # x: (b, s, n_kh) → (b, s, n_vh) where n_vh = n_kh * repeats
        b, s, h = x.shape

        def _te_repeat(x: te.Tensor):
            return te.compute(
                (b, s, h * repeats),
                lambda bi, si, hi: x[bi, si, hi // repeats],
                name="repeat_interleave",
            )

        return op.tensor_expr_op(_te_repeat, "repeat_interleave", [x])

    def _scatter_state(self, state_out: Tensor, layer_state: Tensor, layer_idx: int) -> Tensor:
        """Scatter a single layer's state into the stacked state tensor."""
        return op.tensor_expr_op(
            lambda s, l: _te_scatter_layer(s, l, layer_idx),
            "scatter_state",
            [state_out, layer_state],
            attrs={"op_pattern": 8},
        )

    def _scatter_conv_state(self, conv_out: Tensor, layer_conv: Tensor, layer_idx: int) -> Tensor:
        """Scatter a single layer's conv state into the stacked conv tensor."""
        return op.tensor_expr_op(
            lambda s, l: _te_scatter_layer(s, l, layer_idx),
            "scatter_conv_state",
            [conv_out, layer_conv],
            attrs={"op_pattern": 8},
        )

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        # A_log and dt_bias must stay float32
        self.A_log.to("float32")
        self.dt_bias.to("float32")


# ============================================================================
# Decoder Layer (dispatches between GDN and standard attention)
# ============================================================================


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_id: int, linear_layer_idx: int = -1):
        self.layer_type = config.layer_types()[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35GatedDeltaNet(config, linear_layer_idx)

        self.mlp = Qwen35MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        attn_layer_id: int,
        gdn_state_in: Optional[Tensor] = None,
        gdn_state_out: Optional[Tensor] = None,
        gdn_conv_in: Optional[Tensor] = None,
        gdn_conv_out: Optional[Tensor] = None,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, attn_layer_id)
        else:
            out, gdn_state_out, gdn_conv_out = self.linear_attn(
                out, gdn_state_in, gdn_state_out, gdn_conv_in, gdn_conv_out
            )
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, gdn_state_out, gdn_conv_out

    def forward_rnn(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        attn_layer_id: int,
        state: RNNState,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, attn_layer_id)
        else:
            out, state = self.linear_attn.forward_rnn(out, state)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, state

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        layer_types = config.layer_types()
        linear_idx = 0
        layers = []
        for i in range(config.num_hidden_layers):
            if layer_types[i] == "linear_attention":
                layers.append(Qwen35DecoderLayer(config, i, linear_layer_idx=linear_idx))
                linear_idx += 1
            else:
                layers.append(Qwen35DecoderLayer(config, i))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.layer_types = layer_types

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        gdn_state_in: Optional[Tensor] = None,
        gdn_state_out: Optional[Tensor] = None,
        gdn_conv_in: Optional[Tensor] = None,
        gdn_conv_out: Optional[Tensor] = None,
    ):
        hidden_states = inputs
        attn_layer_id = 0
        for layer_id, layer in enumerate(self.layers):
            if self.layer_types[layer_id] == "full_attention":
                hidden_states, _, _ = layer(hidden_states, paged_kv_cache, attn_layer_id)
                attn_layer_id += 1
            else:
                hidden_states, gdn_state_out, gdn_conv_out = layer(
                    hidden_states,
                    paged_kv_cache,
                    -1,  # not used for linear layers
                    gdn_state_in,
                    gdn_state_out,
                    gdn_conv_in,
                    gdn_conv_out,
                )
        hidden_states = self.norm(hidden_states)
        return hidden_states, gdn_state_out, gdn_conv_out

    def forward_rnn(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        hidden_states = inputs
        attn_layer_id = 0
        for layer_id, layer in enumerate(self.layers):
            if self.layer_types[layer_id] == "full_attention":
                hidden_states, state = layer.forward_rnn(
                    hidden_states, paged_kv_cache, attn_layer_id, state
                )
                attn_layer_id += 1
            else:
                hidden_states, state = layer.forward_rnn(hidden_states, paged_kv_cache, -1, state)
        hidden_states = self.norm(hidden_states)
        return hidden_states, state


class Qwen35LMHeadModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Qwen35Config):
        self.config = config
        self.model = Qwen35Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.partial_rotary_factor = config.partial_rotary_factor
        # GDN config
        self.num_linear_layers = config.num_linear_layers
        self.num_attention_layers = config.num_attention_layers
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def prefill(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        gdn_state_in: Tensor,
        gdn_state_out: Tensor,
        gdn_conv_in: Tensor,
        gdn_conv_out: Tensor,
    ):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states, gdn_state_out, gdn_conv_out = self.model(
            input_embed, paged_kv_cache, gdn_state_in, gdn_state_out, gdn_conv_in, gdn_conv_out
        )
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, gdn_state_out, gdn_conv_out

    def decode(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        gdn_state_in: Tensor,
        gdn_state_out: Tensor,
        gdn_conv_in: Tensor,
        gdn_conv_out: Tensor,
    ):
        op_ext.configure()

        hidden_states, gdn_state_out, gdn_conv_out = self.model(
            input_embed, paged_kv_cache, gdn_state_in, gdn_state_out, gdn_conv_in, gdn_conv_out
        )
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, gdn_state_out, gdn_conv_out

    def _forward_rnn(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        """Shared forward for batch methods using RNNState."""
        op_ext.configure()
        hidden_states, state = self.model.forward_rnn(input_embed, paged_kv_cache, state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_rnn(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_rnn(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_rnn(input_embeds, paged_kv_cache, rnn_state)

    def create_rnn_state(
        self,
        max_batch_size: tir.Var,
        max_history: tir.Var,
    ) -> RNNState:
        K = self.linear_key_head_dim
        V = self.linear_value_head_dim
        n_vh = self.linear_num_value_heads
        n_kh = self.config.linear_num_key_heads
        qkv_dim = n_kh * K * 2 + n_vh * V
        conv_ks_m1 = self.config.linear_conv_kernel_dim - 1

        init_values = [
            R.const(np.zeros((n_vh, K, V), "float32")),
            R.const(np.zeros((conv_ks_m1, qkv_dim), self.dtype)),
        ]
        
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_linear_layers,
            max_history=max_history,
            init_values=init_values,
        )

    def create_paged_kv_cache(
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            # Only attention layers use the KV cache
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rotary_dim=rotary_dim,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        K = self.linear_key_head_dim
        V = self.linear_value_head_dim
        n_vh = self.linear_num_value_heads
        n_lin = self.num_linear_layers
        conv_dim = self.config.linear_conv_kernel_dim
        # QKV dim for conv state
        qkv_dim = self.config.linear_num_key_heads * K * 2 + n_vh * V
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "gdn_state_in": nn.spec.Tensor([n_lin, 1, n_vh, K, V], "float32"),
                "gdn_state_out": nn.spec.Tensor([n_lin, 1, n_vh, K, V], "float32"),
                "gdn_conv_in": nn.spec.Tensor([n_lin, 1, conv_dim - 1, qkv_dim], self.dtype),
                "gdn_conv_out": nn.spec.Tensor([n_lin, 1, conv_dim - 1, qkv_dim], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "gdn_state_in": nn.spec.Tensor([n_lin, 1, n_vh, K, V], "float32"),
                "gdn_state_out": nn.spec.Tensor([n_lin, 1, n_vh, K, V], "float32"),
                "gdn_conv_in": nn.spec.Tensor([n_lin, 1, conv_dim - 1, qkv_dim], self.dtype),
                "gdn_conv_out": nn.spec.Tensor([n_lin, 1, conv_dim - 1, qkv_dim], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
