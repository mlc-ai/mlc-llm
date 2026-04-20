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
from tvm import relax as R
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

from mlc_llm import op as op_ext
from mlc_llm.model.model_utils import index_last_token
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
    rope_theta: int = 10000000
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
# GatedDeltaNet kernels
# ============================================================================


def create_gated_delta_net_recurrent_func(
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
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
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


def create_gated_delta_net_chunked_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
    chunk_size: int,
):
    """Creates a Relax/TIR hybrid function for prefill chunked GatedDeltaNet."""

    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "num_value_heads must be divisible by num_key_heads "
            f"(got {num_value_heads} and {num_key_heads})"
        )

    heads_per_group = num_value_heads // num_key_heads

    def _expand_key_heads_to_value_heads(x: Tensor, name: str) -> Tensor:
        if heads_per_group == 1:
            return x

        def _te(inp: te.Tensor):
            batch_size, seq_len, _, head_dim = inp.shape
            return te.compute(
                (batch_size, seq_len, num_value_heads, head_dim),
                lambda bi, si, hi, di: inp[bi, si, hi // heads_per_group, di],
                name=name,
            )

        return op.tensor_expr_op(_te, name, [x], attrs={"op_pattern": 8})

    def _transpose_pad_bshd_to_bhsd(
        x: Tensor,
        padded_len,
        name: str,
        scale: Optional[float] = None,
    ) -> Tensor:
        def _te(inp: te.Tensor):
            batch_size, seq_len, num_heads, head_dim = inp.shape
            zero = tirx.const(0.0, inp.dtype)
            if scale is None:
                return te.compute(
                    (batch_size, num_heads, padded_len, head_dim),
                    lambda bi, hi, si, di: tirx.if_then_else(
                        si < seq_len,
                        inp[bi, si, hi, di],
                        zero,
                    ),
                    name=name,
                )

            scale_const = tirx.const(scale, inp.dtype)
            return te.compute(
                (batch_size, num_heads, padded_len, head_dim),
                lambda bi, hi, si, di: tirx.if_then_else(
                    si < seq_len,
                    inp[bi, si, hi, di] * scale_const,
                    zero,
                ),
                name=name,
            )

        return op.tensor_expr_op(_te, name, [x], attrs={"op_pattern": 8})

    def _transpose_pad_bsh_to_bhs(x: Tensor, padded_len, name: str) -> Tensor:
        def _te(inp: te.Tensor):
            batch_size, seq_len, num_heads = inp.shape
            zero = tirx.const(0.0, inp.dtype)
            return te.compute(
                (batch_size, num_heads, padded_len),
                lambda bi, hi, si: tirx.if_then_else(
                    si < seq_len,
                    inp[bi, si, hi],
                    zero,
                ),
                name=name,
            )

        return op.tensor_expr_op(_te, name, [x], attrs={"op_pattern": 8})

    def _crop_padded_output(x: Tensor, seq_len, name: str = "crop_padded_output") -> Tensor:
        def _te(inp: te.Tensor):
            batch_size, _, num_heads, value_dim = inp.shape
            return te.compute(
                (batch_size, seq_len, num_heads, value_dim),
                lambda bi, si, hi, vi: inp[bi, si, hi, vi],
                name=name,
            )

        return op.tensor_expr_op(_te, name, [x], attrs={"op_pattern": 8})

    def _create_associative_scan_primfunc():
        _chunk_size = chunk_size
        @T.prim_func
        def associative_scan(attn_mm: T.handle, g_cumsum: T.handle, attn_out: T.handle):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            batch_size = T.int64()
            num_heads = T.int64()
            num_chunks = T.int64()
            attn_mm_buf = T.match_buffer(
                attn_mm,
                (batch_size, num_heads, num_chunks, T.int64(_chunk_size), T.int64(_chunk_size)),
                "float32",
            )
            g_cumsum_buf = T.match_buffer(
                g_cumsum,
                (batch_size, num_heads, num_chunks, T.int64(_chunk_size)),
                "float32",
            )
            attn_out_buf = T.match_buffer(
                attn_out,
                (batch_size, num_heads, num_chunks, T.int64(_chunk_size), T.int64(_chunk_size)),
                "float32",
            )
            scan_buf = T.alloc_buffer((T.int64(_chunk_size), T.int64(_chunk_size)), "float32", scope="shared")

            with T.sblock("root"):
                for b in T.thread_binding(batch_size, thread="blockIdx.x"):
                    for h in T.thread_binding(num_heads, thread="blockIdx.y"):
                        for c in T.thread_binding(num_chunks, thread="blockIdx.z"):
                            for tx in T.thread_binding(_chunk_size, thread="threadIdx.x"):
                                # init phase: start from all zeros (diag must stay 0 during recurrence)
                                for i in T.serial(_chunk_size):
                                    scan_buf[i, tx] = T.float32(0.0)

                                T.tvm_storage_sync("shared")

                                # recurrence
                                for i in T.serial(1, _chunk_size):
                                    g_i = g_cumsum_buf[b, h, c, i]
                                    scan_buf[i, tx] = T.if_then_else(
                                        tx < i,
                                        -attn_mm_buf[b, h, c, i, tx]
                                        * T.exp(g_i - g_cumsum_buf[b, h, c, tx]),
                                        scan_buf[i, tx],
                                    )
                                    for k in T.serial(i):
                                        if tx < i:
                                            decayed = (
                                                -attn_mm_buf[b, h, c, i, k]
                                                * T.exp(g_i - g_cumsum_buf[b, h, c, k])
                                            )
                                            scan_buf[i, tx]  = scan_buf[i, tx] + decayed * scan_buf[k, tx]
                                    T.tvm_storage_sync("shared")

                                # writeback
                                for i in T.serial(_chunk_size):
                                    attn_out_buf[b, h, c, i, tx] = T.if_then_else(
                                        tx < i,
                                        scan_buf[i, tx],
                                        T.if_then_else(tx == i, T.float32(1.0), T.float32(0.0)),
                                    )
        return associative_scan

    def _create_inter_chunk_recurrent_primfunc():
        @T.prim_func
        def inter_chunk_recurrent(
            query: T.handle,
            key: T.handle,
            value_intra: T.handle,
            k_cumdecay: T.handle,
            g_cumsum: T.handle,
            initial_state: T.handle,
            core_attn_out: T.handle,
            recurrent_state_out: T.handle,
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            batch_size = T.int64()
            num_chunks = T.int64()

            query_buf = T.match_buffer(
                query,
                (batch_size, T.int64(num_value_heads), num_chunks, T.int64(chunk_size), T.int64(key_head_dim)),
                dtype="float32",
            )
            key_buf = T.match_buffer(
                key,
                (batch_size, T.int64(num_value_heads), num_chunks, T.int64(chunk_size), T.int64(key_head_dim)),
                dtype="float32",
            )
            value_intra_buf = T.match_buffer(
                value_intra,
                (
                    batch_size,
                    T.int64(num_value_heads),
                    num_chunks,
                    T.int64(chunk_size),
                    T.int64(value_head_dim),
                ),
                dtype="float32",
            )
            k_cumdecay_buf = T.match_buffer(
                k_cumdecay,
                (batch_size, T.int64(num_value_heads), num_chunks, T.int64(chunk_size), T.int64(key_head_dim)),
                dtype="float32",
            )
            g_cumsum_buf = T.match_buffer(
                g_cumsum,
                (batch_size, T.int64(num_value_heads), num_chunks, T.int64(chunk_size)),
                dtype="float32",
            )
            initial_state_buf = T.match_buffer(
                initial_state,
                (batch_size, T.int64(num_value_heads), T.int64(key_head_dim), T.int64(value_head_dim)),
                dtype="float32",
            )
            core_attn_out_buf = T.match_buffer(
                core_attn_out,
                (
                    batch_size,
                    T.int64(num_value_heads),
                    num_chunks,
                    T.int64(chunk_size),
                    T.int64(value_head_dim),
                ),
                dtype="float32",
            )
            recurrent_state_out_buf = T.match_buffer(
                recurrent_state_out,
                (batch_size, T.int64(num_value_heads), T.int64(key_head_dim), T.int64(value_head_dim)),
                dtype="float32",
            )

            attn_buf = T.alloc_buffer(
                (
                    batch_size,
                    T.int64(num_value_heads),
                    num_chunks,
                    T.int64(chunk_size),
                    T.int64(chunk_size),
                ),
                dtype="float32",
            )
            v_new_buf = T.alloc_buffer(
                (
                    batch_size,
                    T.int64(num_value_heads),
                    num_chunks,
                    T.int64(chunk_size),
                    T.int64(value_head_dim),
                ),
                dtype="float32",
            )
            attn_inter_buf = T.alloc_buffer(
                (
                    batch_size,
                    T.int64(num_value_heads),
                    num_chunks,
                    T.int64(chunk_size),
                    T.int64(value_head_dim),
                ),
                dtype="float32",
            )

            with T.sblock("root"):
                for b in T.thread_binding(batch_size, thread="blockIdx.x"):
                    for h in T.thread_binding(T.int64(num_value_heads), thread="blockIdx.y"):
                        for kd in T.serial(T.int64(key_head_dim)):
                            for vd in T.serial(T.int64(value_head_dim)):
                                recurrent_state_out_buf[b, h, kd, vd] = initial_state_buf[b, h, kd, vd]

                        for chunk_idx in T.serial(num_chunks):
                            for c1 in T.serial(T.int64(chunk_size)):
                                g_i = g_cumsum_buf[b, h, chunk_idx, c1]
                                for c2 in T.serial(T.int64(chunk_size)):
                                    attn_buf[b, h, chunk_idx, c1, c2] = T.float32(0.0)
                                    for kd in T.serial(T.int64(key_head_dim)):
                                        attn_buf[b, h, chunk_idx, c1, c2] += (
                                            query_buf[b, h, chunk_idx, c1, kd]
                                            * key_buf[b, h, chunk_idx, c2, kd]
                                        )
                                    attn_buf[b, h, chunk_idx, c1, c2] = T.if_then_else(
                                        c2 > c1,
                                        T.float32(0.0),
                                        attn_buf[b, h, chunk_idx, c1, c2]
                                        * T.exp(g_i - g_cumsum_buf[b, h, chunk_idx, c2]),
                                    )

                            for c in T.serial(T.int64(chunk_size)):
                                g_curr = g_cumsum_buf[b, h, chunk_idx, c]
                                exp_curr = T.exp(g_curr)
                                for vd in T.serial(T.int64(value_head_dim)):
                                    v_new_buf[b, h, chunk_idx, c, vd] = value_intra_buf[
                                        b, h, chunk_idx, c, vd
                                    ]
                                    for kd in T.serial(T.int64(key_head_dim)):
                                        v_new_buf[b, h, chunk_idx, c, vd] -= (
                                            k_cumdecay_buf[b, h, chunk_idx, c, kd]
                                            * recurrent_state_out_buf[b, h, kd, vd]
                                        )

                                for vd in T.serial(T.int64(value_head_dim)):
                                    attn_inter_buf[b, h, chunk_idx, c, vd] = T.float32(0.0)
                                    for kd in T.serial(T.int64(key_head_dim)):
                                        attn_inter_buf[b, h, chunk_idx, c, vd] += (
                                            query_buf[b, h, chunk_idx, c, kd]
                                            * exp_curr
                                            * recurrent_state_out_buf[b, h, kd, vd]
                                        )

                                for vd in T.serial(T.int64(value_head_dim)):
                                    core_attn_out_buf[b, h, chunk_idx, c, vd] = attn_inter_buf[
                                        b, h, chunk_idx, c, vd
                                    ]
                                    for r in T.serial(T.int64(chunk_size)):
                                        core_attn_out_buf[b, h, chunk_idx, c, vd] += (
                                            attn_buf[b, h, chunk_idx, c, r]
                                            * v_new_buf[b, h, chunk_idx, r, vd]
                                        )

                            for kd in T.serial(T.int64(key_head_dim)):
                                for vd in T.serial(T.int64(value_head_dim)):
                                    g_last = g_cumsum_buf[b, h, chunk_idx, T.int64(chunk_size - 1)]
                                    exp_last = T.exp(g_last)
                                    recurrent_state_out_buf[b, h, kd, vd] = (
                                        recurrent_state_out_buf[b, h, kd, vd] * exp_last
                                    )
                                    for c in T.serial(T.int64(chunk_size)):
                                        recurrent_state_out_buf[b, h, kd, vd] += (
                                            key_buf[b, h, chunk_idx, c, kd]
                                            * v_new_buf[b, h, chunk_idx, c, vd]
                                            * T.exp(g_last - g_cumsum_buf[b, h, chunk_idx, c])
                                        )

        return inter_chunk_recurrent

    def gdn_chunked_func(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        gate: Tensor,
        beta: Tensor,
        state_in: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        q = _expand_key_heads_to_value_heads(q, "expand_query_to_value_heads")
        k = _expand_key_heads_to_value_heads(k, "expand_key_to_value_heads")

        q = op.astype(q, "float32")
        k = op.astype(k, "float32")
        v = op.astype(v, "float32")
        gate = op.astype(gate, "float32")
        beta = op.astype(beta, "float32")
        state_in = op.astype(state_in, "float32")

        batch_size, seq_len, _, _ = q.shape
        num_chunks = tirx.ceildiv(seq_len, chunk_size)
        padded_len = num_chunks * chunk_size
        scale = 1.0 / math.sqrt(key_head_dim)

        query_t = _transpose_pad_bshd_to_bhsd(q, padded_len, "transpose_pad_query", scale=scale)
        key_t = _transpose_pad_bshd_to_bhsd(k, padded_len, "transpose_pad_key")
        value_t = _transpose_pad_bshd_to_bhsd(v, padded_len, "transpose_pad_value")
        g_t = _transpose_pad_bsh_to_bhs(gate, padded_len, "transpose_pad_g")
        beta_t = _transpose_pad_bsh_to_bhs(beta, padded_len, "transpose_pad_beta")

        v_beta = value_t * op.unsqueeze(beta_t, -1)
        k_beta = key_t * op.unsqueeze(beta_t, -1)

        query_chunked = op.reshape(
            query_t,
            (batch_size, num_value_heads, num_chunks, chunk_size, key_head_dim),
        )
        key_chunked = op.reshape(
            key_t,
            (batch_size, num_value_heads, num_chunks, chunk_size, key_head_dim),
        )
        k_beta_chunked = op.reshape(
            k_beta,
            (batch_size, num_value_heads, num_chunks, chunk_size, key_head_dim),
        )
        v_beta_chunked = op.reshape(
            v_beta,
            (batch_size, num_value_heads, num_chunks, chunk_size, value_head_dim),
        )
        g_chunked = op.reshape(g_t, (batch_size, num_value_heads, num_chunks, chunk_size))

        g_cumsum = op.cumsum(g_chunked, axis=-1)
        exp_g_cumsum = op.exp(g_cumsum)

        key_chunked_t = op.permute_dims(key_chunked, [0, 1, 2, 4, 3])
        attn_mm = op.matmul(k_beta_chunked, key_chunked_t, out_dtype="float32")

        attn_identity = op.tensor_ir_op(
            _create_associative_scan_primfunc(),
            "gdn_associative_scan",
            [attn_mm, g_cumsum],
            Tensor.placeholder(
                (batch_size, num_value_heads, num_chunks, chunk_size, chunk_size),
                "float32",
            ),
        )

        value_intra = op.matmul(attn_identity, v_beta_chunked, out_dtype="float32")
        k_cumdecay_input = k_beta_chunked * op.unsqueeze(exp_g_cumsum, -1)
        k_cumdecay = op.matmul(attn_identity, k_cumdecay_input, out_dtype="float32")

        core_attn_out_chunked, recurrent_state_out = op.tensor_ir_op(
            _create_inter_chunk_recurrent_primfunc(),
            "gdn_inter_chunk_recurrent",
            [
                query_chunked,
                key_chunked,
                value_intra,
                k_cumdecay,
                g_cumsum,
                state_in,
            ],
            [
                Tensor.placeholder(
                    (batch_size, num_value_heads, num_chunks, chunk_size, value_head_dim),
                    "float32",
                ),
                Tensor.placeholder(
                    (batch_size, num_value_heads, key_head_dim, value_head_dim),
                    "float32",
                ),
            ],
        )

        core_attn_out_padded = op.reshape(
            core_attn_out_chunked,
            (batch_size, num_value_heads, padded_len, value_head_dim),
        )
        core_attn_out_padded = op.permute_dims(core_attn_out_padded, [0, 2, 1, 3])
        core_attn_out = _crop_padded_output(core_attn_out_padded, seq_len)
        return core_attn_out, recurrent_state_out

    return gdn_chunked_func


# ============================================================================
# GatedDeltaNet Linear Attention Layer
# ============================================================================


class Qwen35GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention layer."""

    def __init__(self, config: Qwen35Config, linear_layer_idx: int):
        self.config = config
        self.linear_layer_idx = linear_layer_idx  # index among linear layers only
        self.key_head_dim = config.linear_key_head_dim  # 128
        self.value_head_dim = config.linear_value_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 16 or 32
        # Kernel chunk size is algorithmic and should stay small (e.g. 64).
        # Do not use engine prefill_chunk_size (e.g. 2048), which controls request splitting.
        self.gdn_chunk_size = 64
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
        self, hidden_states: Tensor, state: RNNState, is_prefill: bool
    ) -> Tuple[Tensor, RNNState]:
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
        gate_log, gate_exp, beta = self._compute_gate_beta(alpha, beta_raw)
        # beta is already (b, s, n_vh) — no GVA expansion needed.

        # Get recurrent state from RNNState (state_id=0)
        state_in_layer = state.get(layer_idx, 0, (b, n_vh, K, V), "float32")

        # Dispatch recurrent path for decode and chunked path for prefill.
        if is_prefill:
            out_recurrent, state_out_layer = create_gated_delta_net_chunked_func(
                num_key_heads=n_kh,
                num_value_heads=n_vh,
                key_head_dim=K,
                value_head_dim=V,
                dtype=self.dtype,
                chunk_size=self.gdn_chunk_size,
            )(q, k, v, gate_log, beta, state_in_layer)
            #out_recurrent, state_out_layer = op.tensor_ir_op(
            #    create_gated_delta_net_recurrent_func(
            #        num_key_heads=n_kh,
            #        num_value_heads=n_vh,
            #        key_head_dim=K,
            #        value_head_dim=V,
            #        dtype=self.dtype,
            #    ),
            #    "gated_delta_net_recurrent",
            #    [q, k, v, gate, beta, state_in_layer],
            #    [
            #        Tensor.placeholder([b, s, n_vh, V], "float32"),
            #        Tensor.placeholder([b, n_vh, K, V], "float32"),
            #    ],
            #)
        else:
            out_recurrent, state_out_layer = op.tensor_ir_op(
                create_gated_delta_net_recurrent_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net_recurrent",
                [q, k, v, gate_exp, beta, state_in_layer],
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
                lambda bi, ti, di: tirx.if_then_else(
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
                    tirx.if_then_else(
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

        gate_log = -exp(A_log) * softplus(alpha + dt_bias)  (per value_head)
        gate_exp = exp(gate_log)  (per value_head)
        beta = sigmoid(beta_raw)  (per value_head)

        chunked prefill path expects gate_log, while recurrent path expects gate_exp.
        """

        # alpha: (b, s, n_vh), dt_bias: (n_vh,), A_log: (n_vh,)
        def _te_gate_log(alpha: te.Tensor, A_log: te.Tensor, dt_bias: te.Tensor):
            b, s, h = alpha.shape

            def _softplus(x):
                # softplus(x) = x if x > 20 else log(1 + exp(x))
                return tirx.if_then_else(x > 20.0, x, tirx.log(1.0 + tirx.exp(x)))

            return te.compute(
                (b, s, h),
                lambda bi, si, hi: -tirx.exp(A_log[hi].astype("float32"))
                * _softplus((alpha[bi, si, hi] + dt_bias[hi]).astype("float32")),
                name="gate_log",
            )

        gate_log = op.tensor_expr_op(
            _te_gate_log,
            "gate_log",
            [alpha, self.A_log, self.dt_bias],
            attrs={"op_pattern": 8},
        )
        gate_exp = op.exp(gate_log)

        beta = op.sigmoid(beta_raw).astype("float32")
        return gate_log, gate_exp, beta

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
    def __init__(self, config: Qwen35Config, layer_id: int, category_id: int):
        """
        layer_id is the id of the layer within all of the layers
        category_id is the index of the layer within the category of layers that it belongs to
        ie, linear attention or regular attention
        """
        self.layer_type = config.layer_types()[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35GatedDeltaNet(config, category_id)
        self.category_id = category_id
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
        state: RNNState,
        is_prefill: bool,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id)
        else:
            out, state = self.linear_attn(out, state, is_prefill)
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
        attn_idx = 0
        layers = []
        for i, ltype in enumerate(layer_types):
            if ltype == "linear_attention":
                layers.append(Qwen35DecoderLayer(config, i, category_id=linear_idx))
                linear_idx += 1
            else:
                layers.append(Qwen35DecoderLayer(config, i, category_id=attn_idx))
                attn_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        is_prefill: bool,
    ):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states, state = layer.forward(hidden_states, paged_kv_cache, state, is_prefill)
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

    def _forward(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        is_prefill: bool,
        logit_positions: Optional[Tensor] = None,
    ):
        """Shared forward for the RNNState path."""
        op_ext.configure()
        hidden_states, state = self.model.forward(input_embed, paged_kv_cache, state, is_prefill)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, state

    def prefill(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        op_ext.configure()

        hidden_states, rnn_state = self.model.forward(
            input_embed, paged_kv_cache, rnn_state, True
        )
        hidden_states = index_last_token(hidden_states)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        
        return logits, paged_kv_cache, rnn_state

    def decode(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embed, paged_kv_cache, rnn_state, False)

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, True, logit_positions)

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, False)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, False)

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
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
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
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
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
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
