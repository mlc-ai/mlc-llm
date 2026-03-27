"""
Implementation for NemotronH hybrid Mamba2-Attention-MoE architecture.
Reference: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
"""

import dataclasses
from typing import Any, Dict, List, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NemotronHConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes  # pylint: disable=too-many-instance-attributes
    """Configuration for the NemotronH hybrid Mamba2-Attention-MoE model."""

    vocab_size: int
    hidden_size: int
    # Attention
    num_attention_heads: int
    num_key_value_heads: int
    layers_block_type: List[str] = dataclasses.field(default_factory=list)
    head_dim: int = 0
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    # MLP / MoE
    intermediate_size: int = 21504
    mlp_hidden_act: str = "relu2"
    mlp_bias: bool = False
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    moe_latent_size: Optional[int] = None

    # Mamba2
    ssm_state_size: int = 128
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_n_groups: int = 8

    # Norm
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False

    # MLC-LLM runtime
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    pipeline_parallel_stages: int = 1
    max_batch_size: int = 1
    disaggregation: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        _pattern_map = {"M": "mamba", "-": "mlp", "E": "moe", "*": "attention"}
        # Map HF field names to our internal names via kwargs
        if "n_groups" in self.kwargs:
            self.mamba_n_groups = self.kwargs.pop("n_groups")
        if "conv_kernel" in self.kwargs:
            self.mamba_d_conv = self.kwargs.pop("conv_kernel")
        if "chunk_size" in self.kwargs:
            self.mamba_chunk_size = self.kwargs.pop("chunk_size")
        if "use_conv_bias" in self.kwargs:
            self.mamba_conv_bias = self.kwargs.pop("use_conv_bias")
        if "rms_norm_eps" in self.kwargs:
            self.layer_norm_epsilon = self.kwargs.pop("rms_norm_eps")
        # Handle layers_block_type from various sources
        if isinstance(self.layers_block_type, str) and self.layers_block_type:
            self.layers_block_type = [_pattern_map[c] for c in self.layers_block_type]
        elif not self.layers_block_type:
            pattern = self.kwargs.pop("hybrid_override_pattern", None)
            if pattern:
                self.layers_block_type = [_pattern_map[c] for c in pattern]

        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.context_window_size == 0:
            self.context_window_size = self.max_position_embeddings

        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 8192)

        # Build attention layer index maps
        self.attention_layer_ids = [
            i for i, t in enumerate(self.layers_block_type) if t == "attention"
        ]
        self.layer_id_to_attn_idx: Dict[int, int] = {
            layer_id: attn_idx for attn_idx, layer_id in enumerate(self.attention_layer_ids)
        }

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers_block_type)

    @property
    def num_attention_layers(self) -> int:
        return len(self.attention_layer_ids)

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim


# pylint: disable=invalid-name,missing-docstring


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def _relu2(x: Tensor) -> Tensor:
    return op.square(op.relu(x))


def _get_act_fn(name: str):
    if name == "relu2":
        return _relu2
    if name == "silu":
        return op.silu
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class NemotronHRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((hidden_size,), dtype="float32")
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        return op.rms_norm(x, self.weight, axes=[-1], epsilon=self.variance_epsilon)


class NemotronHRMSNormGated(nn.Module):
    """Gated RMSNorm: rms_norm(x) * silu(z)"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((hidden_size,), dtype="float32")
        self.variance_epsilon = eps

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x_normed = op.rms_norm(x, self.weight, axes=[-1], epsilon=self.variance_epsilon)
        return x_normed * op.silu(z)


# ---------------------------------------------------------------------------
# Mamba2 Mixer
# ---------------------------------------------------------------------------


class NemotronHMamba2Mixer(nn.Module):  # pylint: disable=too-many-instance-attributes  # pylint: disable=too-many-instance-attributes
    """
    Mamba-2 selective state space mixer (pure Relax, no custom CUDA kernels).
    Implements the chunked SSD algorithm from NemotronHMamba2Mixer.torch_forward.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_intermediate_size
        self.n_groups = config.mamba_n_groups
        self.head_dim = config.mamba_head_dim
        self.num_heads = config.mamba_num_heads
        self.chunk_size = config.mamba_chunk_size

        # conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # in_proj: hidden -> gate(intermediate) + x_bc(conv_dim) + dt(num_heads)
        self.in_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            bias=config.mamba_proj_bias,
        )

        # Depthwise conv weights: [conv_dim, conv_kernel_size]
        # (HF stores [conv_dim, 1, kernel]; squeezed in the loader)
        self.conv1d_weight = nn.Parameter((self.conv_dim, self.conv_kernel_size), dtype="float32")
        if config.mamba_conv_bias:
            self.conv1d_bias = nn.Parameter((self.conv_dim,), dtype="float32")
        else:
            self.conv1d_bias = None

        self.dt_bias = nn.Parameter((self.num_heads,), dtype="float32")
        self.A_log = nn.Parameter((self.num_heads,), dtype="float32")
        self.D = nn.Parameter((self.num_heads,), dtype="float32")

        self.norm = NemotronHRMSNormGated(self.intermediate_size, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=config.mamba_proj_bias
        )

    def _causal_conv1d(self, x: Tensor) -> Tensor:
        """Causal depthwise conv1d + silu.  x: [b, s, conv_dim] -> [b, s, conv_dim]"""
        b, s, c = x.shape
        k = self.conv_kernel_size
        x_t = op.permute_dims(x, [0, 2, 1])  # [b, c, s]

        def _conv(x_in: te.Tensor, w: te.Tensor):
            j = te.reduce_axis((0, k), name="j")
            return te.compute(
                (b, c, s),
                lambda bi, ci, si: te.sum(
                    te.if_then_else(
                        si - k + 1 + j >= 0,
                        x_in[bi, ci, si - k + 1 + j] * w[ci, j],
                        tir.const(0, "float32"),
                    ),
                    axis=j,
                ),
                name="causal_conv1d",
            )

        x_conv = op.tensor_expr_op(_conv, name_hint="causal_conv1d", args=[x_t, self.conv1d_weight])
        x_conv = op.permute_dims(x_conv, [0, 2, 1])  # [b, s, c]
        if self.conv1d_bias is not None:
            x_conv = x_conv + self.conv1d_bias.astype(x_conv.dtype)
        return op.silu(x_conv)

    def forward(self, hidden_states: Tensor, _paged_kv_cache=None, _layer_id=None) -> Tensor:
        """Chunked SSD prefill forward (no recurrent state carried across calls)."""
        b, s, _ = hidden_states.shape
        chunk_size = self.chunk_size
        dtype = hidden_states.dtype

        # 1. Input projection -> gate, x_bc, dt_raw
        proj = self.in_proj(hidden_states)
        i0 = self.intermediate_size
        i1 = i0 + self.conv_dim
        i2 = i1 + self.num_heads
        gate, x_bc, dt_r = op.split(proj, indices_or_sections=[i0, i1], axis=2)

        # 2. Conv + silu
        x_bc = self._causal_conv1d(x_bc)

        # 3. Split x, B, C
        gs = self.n_groups * self.ssm_state_size
        x, B, C = op.split(
            x_bc, indices_or_sections=[self.intermediate_size, self.intermediate_size + gs], axis=2
        )

        # 4. dt = softplus(dt_raw + dt_bias),  A = -exp(A_log)
        dt = op.softplus(dt_r + self.dt_bias.astype(dtype))  # [b, s, nh]
        A = op.negative(op.exp(self.A_log.astype(dtype)))  # [nh]

        # 5. Reshape and repeat B/C to num_heads
        x = op.reshape(x.astype(dtype), (b, s, self.num_heads, self.head_dim))
        B = op.reshape(B, (b, s, self.n_groups, self.ssm_state_size))
        C = op.reshape(C, (b, s, self.n_groups, self.ssm_state_size))
        rep = self.num_heads // self.n_groups
        B = op.repeat(B, rep, axis=2)  # [b, s, nh, d_state]
        C = op.repeat(C, rep, axis=2)

        # 6. Discretise: dx = x * dt[...,None],  dA = A * dt
        dt4 = op.unsqueeze(dt, -1)  # [b, s, nh, 1]
        dx = x * dt4  # [b, s, nh, hd]
        dA = op.reshape(A, (1, 1, self.num_heads)) * dt  # [b, s, nh]

        # 7. Pad to multiple of chunk_size
        pad = (chunk_size - s % chunk_size) % chunk_size

        def _pad(t: Tensor, p: int) -> Tensor:
            shape = list(t.shape)
            shape[1] = p
            return op.concat([t, op.zeros(shape, dtype=t.dtype)], 1)

        dx_p = _pad(dx, pad)
        dA_p = _pad(dA, pad)
        B_p = _pad(B, pad)
        C_p = _pad(C, pad)
        S = s + pad
        nc = S // chunk_size

        dx_c = op.reshape(dx_p, (b, nc, chunk_size, self.num_heads, self.head_dim))
        dA_c = op.reshape(dA_p, (b, nc, chunk_size, self.num_heads))
        B_c = op.reshape(B_p, (b, nc, chunk_size, self.num_heads, self.ssm_state_size))
        C_c = op.reshape(C_p, (b, nc, chunk_size, self.num_heads, self.ssm_state_size))

        # dA cumsum along chunk dim: [b, nh, nc, cs]
        dA_t = op.permute_dims(dA_c, [0, 3, 1, 2])
        dA_cumsum = op.cumsum(dA_t, axis=-1)

        # Causal mask via L[i,j] = exp(cumsum[i] - cumsum[j])
        dA_i = op.unsqueeze(dA_cumsum, -1)  # [b, nh, nc, cs, 1]
        dA_j = op.unsqueeze(dA_cumsum, -2)  # [b, nh, nc, 1, cs]
        L = op.exp(dA_i - dA_j).astype(dtype)  # [b, nh, nc, cs, cs]

        # Permute for matmuls
        dx_t = op.permute_dims(dx_c, [0, 3, 1, 2, 4])  # [b, nh, nc, cs, hd]
        C_t = op.permute_dims(C_c, [0, 3, 1, 2, 4])  # [b, nh, nc, cs, d_state]
        B_t = op.permute_dims(B_c, [0, 3, 1, 2, 4])  # [b, nh, nc, cs, d_state]

        # Intra-chunk: G = C @ B^T, M = G*L, Y_diag = M @ dx
        G = op.matmul(C_t, op.permute_dims(B_t, [0, 1, 2, 4, 3]))  # [..., cs, cs]
        M = G.astype(dtype) * L
        Y_diag = op.matmul(M, dx_t)  # [..., cs, hd]

        # Inter-chunk states
        # Extract last element: split at [chunk_size-1] gives [..., cs-1] and [..., 1]
        _, last_cs = op.split(
            dA_cumsum, indices_or_sections=[self.chunk_size - 1], axis=-1
        )  # [b, nh, nc, 1]
        decay = op.exp(last_cs - dA_cumsum)  # [b, nh, nc, cs]
        decay_b = op.unsqueeze(op.permute_dims(decay, [0, 1, 2, 3]), -1) * B_t.astype(
            dtype
        )  # [..., cs, d_state]
        states = op.matmul(
            op.permute_dims(dx_t.astype(dtype), [0, 1, 2, 4, 3]),  # [..., hd, cs]
            decay_b,  # [..., cs, d_state]
        )  # [b, nh, nc, hd, d_state]

        # Propagate states across chunks
        # Extract last element of cumsum: [b, nh, nc, cs] -> [b, nh, nc]
        def _last_elem(t: te.Tensor):
            return te.compute(
                (b, self.num_heads, nc),
                lambda bi, hi, ci: t[bi, hi, ci, self.chunk_size - 1],
                name="last_elem",
            )

        lc = op.tensor_expr_op(_last_elem, name_hint="last_elem", args=[dA_cumsum])
        zero_lc = op.zeros((b, self.num_heads, 1), dtype=dtype)
        lc_pad = op.concat([zero_lc, lc], -1)  # [b, nh, nc+1]
        lc_i = op.unsqueeze(lc_pad, -1)
        lc_j = op.unsqueeze(lc_pad, -2)
        dc_full = op.exp(lc_i - lc_j)  # [b, nh, nc+1, nc+1]
        _, dc = op.split(dc_full, indices_or_sections=[1], axis=2)

        zero_st = op.zeros((b, self.num_heads, 1, self.head_dim, self.ssm_state_size), dtype=dtype)
        st_all = op.concat([zero_st, states], 2)  # [b, nh, nc+1, hd, d_state]
        st_flat = op.reshape(
            st_all, (b, self.num_heads, nc + 1, self.head_dim * self.ssm_state_size)
        )
        ns_flat = op.matmul(dc, st_flat)  # [b, nh, nc, hd*d_state]
        new_states = op.reshape(
            ns_flat, (b, self.num_heads, nc, self.head_dim, self.ssm_state_size)
        )

        # Y_off = (C @ new_states^T) * exp(dA_cumsum)
        sdo = op.unsqueeze(
            op.permute_dims(op.exp(dA_cumsum).astype(dtype), [0, 1, 2, 3]), -1
        )  # [..., cs, 1]
        Y_off = (
            op.matmul(
                C_t.astype(dtype),
                op.permute_dims(new_states.astype(dtype), [0, 1, 2, 4, 3]),  # [..., d_state, hd]
            )
            * sdo
        )  # [b, nh, nc, cs, hd]

        y = Y_diag + Y_off

        # D skip connection
        D4 = op.reshape(self.D.astype(dtype), (1, self.num_heads, 1, 1, 1))
        y = y + D4 * dx_t

        # Reshape back and trim padding
        y = op.permute_dims(y, [0, 2, 3, 1, 4])  # [b, nc, cs, nh, hd]
        y = op.reshape(y, (b, S, self.num_heads, self.head_dim))
        y = op.reshape(y, (b, S, self.intermediate_size))

        # Trim padding using tensor_expr_op
        def _trim(t: te.Tensor):
            return te.compute(
                (b, s, self.intermediate_size),
                lambda bi, si, ci: t[bi, si, ci],
                name="trim_pad",
            )

        y = op.tensor_expr_op(_trim, name_hint="trim_pad", args=[y])

        # Gated norm + output projection
        y = self.norm(y, gate)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NemotronHAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """GQA with separate q/k/v projections and RoPE via PagedKVCache."""

    def __init__(self, config: NemotronHConfig, attn_layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.attn_layer_idx = attn_layer_idx

        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, _layer_id: int
    ) -> Tensor:
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape

        q = op.reshape(self.q_proj(hidden_states), (b, s, h_q, d))
        k = op.reshape(self.k_proj(hidden_states), (b, s, h_kv, d))
        v = op.reshape(self.v_proj(hidden_states), (b, s, h_kv, d))

        qkv = op.concat(
            [
                op.reshape(q, (b, s, h_q * d)),
                op.reshape(k, (b, s, h_kv * d)),
                op.reshape(v, (b, s, h_kv * d)),
            ],
            -1,
        )
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))

        out = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                self.attn_layer_idx,
                qkv,
                self.num_q_heads,
                sm_scale=self.head_dim**-0.5,
            ),
            (b, s, h_q * d),
        )
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class NemotronHMLP(nn.Module):
    """Non-gated MLP: down(act(up(x)))."""

    def __init__(self, config: NemotronHConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        mid = intermediate_size or config.intermediate_size
        self.act_fn = _get_act_fn(config.mlp_hidden_act)
        self.up_proj = nn.Linear(config.hidden_size, mid, bias=config.mlp_bias)
        self.down_proj = nn.Linear(mid, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class NemotronHMoE(nn.Module):  # pylint: disable=too-many-instance-attributes  # pylint: disable=too-many-instance-attributes
    """
    MoE with non-gated experts (up+down only, no gate_proj).
    Routing: sigmoid + group-limited greedy top-k (noaux_tc, matches DeepSeek V2).
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        moe_mid = config.moe_intermediate_size

        # Router
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False, out_dtype="float32")
        # Bias lives directly on MoE (same as DeepSeek V2 noaux_tc pattern)
        self.e_score_correction_bias = nn.Parameter((self.num_experts,), dtype="float32")

        # Non-gated routed experts: weight shape = (n_experts, out_features, in_features)
        self.experts_up = MixtralExperts(
            self.num_experts, in_features=config.hidden_size, out_features=moe_mid
        )
        self.experts_down = MixtralExperts(
            self.num_experts, in_features=moe_mid, out_features=config.hidden_size
        )
        self.act_fn = _get_act_fn(config.mlp_hidden_act)

        # Always-active shared expert
        self.shared_expert = NemotronHMLP(
            config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
        )
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        self.e_score_correction_bias.to("float32")  # always float32

    def forward(self, x: Tensor) -> Tensor:
        b, s, h = x.shape
        num_tokens = b * s
        x_flat = op.reshape(x, (num_tokens, h))

        # Router
        logits = self.gate(x_flat)  # float32 [T, E]
        scores = op.sigmoid(logits)

        expert_weights, expert_indices = op_ext.moe_misc.group_limited_greedy_topk(
            scores,
            self.num_experts_per_tok,
            self.num_experts,
            self.n_group,
            self.topk_group,
            "noaux_tc",
            num_tokens,
            self.e_score_correction_bias,
        )

        if self.norm_topk_prob:
            denom = op.sum(expert_weights, axis=-1, keepdims=True) + 1e-20
            expert_weights = expert_weights / denom
        expert_weights = expert_weights * self.routed_scaling_factor

        def _expert_forward(inp: Tensor, indptr: Tensor) -> Tensor:
            up = self.experts_up(inp, indptr)  # non-gated: just up_proj
            up = self.act_fn(up)
            return self.experts_down(up, indptr)

        if num_tokens == 1:
            # Decode: indptr is shape [1, k] (expert indices directly)
            moe_out = _expert_forward(x_flat, expert_indices)
        else:
            # Prefill: sort -> batch-GEMM -> scatter
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, self.num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            indptr = op_ext.moe_misc.get_indptr(
                cumsum, self.num_experts, num_tokens, inclusive=False, out_dtype="int32"
            )

            moe_out = op.take(x_flat, token_indices, axis=0)
            moe_out = _expert_forward(moe_out, indptr)
            moe_out = op_ext.moe_misc.scatter_output(moe_out, reverse_indices)

        # Weighted sum: [T, k, h] * weights -> [T, h]
        expert_weights = op.reshape(
            expert_weights.astype(x_flat.dtype),
            (num_tokens, self.num_experts_per_tok, 1),
        )
        moe_out = op.reshape(moe_out, (num_tokens, self.num_experts_per_tok, h)) * expert_weights
        moe_out = op_ext.moe_misc.moe_sum(moe_out, dim=1)  # [T, h]

        # Add shared expert
        moe_out = moe_out + self.shared_expert(x_flat)
        return op.reshape(moe_out, (b, s, h))


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------


class NemotronHDecoderLayer(nn.Module):
    def __init__(self, config: NemotronHConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.block_type = config.layers_block_type[layer_id]
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.tensor_parallel_shards = config.tensor_parallel_shards

        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config)
        elif self.block_type == "attention":
            attn_idx = config.layer_id_to_attn_idx[layer_id]
            self.mixer = NemotronHAttention(config, attn_layer_idx=attn_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config)
        elif self.block_type == "moe":
            self.mixer = NemotronHMoE(config)
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int) -> Tensor:
        residual = hidden_states
        normed = self.norm(hidden_states)
        if self.block_type == "attention":
            out = self.mixer(normed, paged_kv_cache, layer_id)
        else:  # mamba, mlp, moe
            out = self.mixer(normed)
        return self._residual(out, residual)

    def _residual(self, out: Tensor, residual: Tensor) -> Tensor:
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class NemotronHModel(nn.Module):
    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = NemotronHRMSNorm(config.hidden_size, config.layer_norm_epsilon)

        stages = config.pipeline_parallel_stages
        lps = (config.num_hidden_layers + stages - 1) // stages
        self.layer_partition = [i * lps for i in range(stages)] + [config.num_hidden_layers]

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            if layer_id != 0 and layer_id in self.layer_partition:
                hidden_states = op_ext.pipeline_stage_boundary(hidden_states)
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        return self.norm(hidden_states)


class NemotronHForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.model = NemotronHModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_layers = config.num_attention_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.disaggregation = config.disaggregation
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        op_ext.configure()
        if self.tie_word_embeddings:
            weight = op.permute_dims(self.model.embed_tokens.weight)
            logits = op.matmul(hidden_states, weight, out_dtype="float32")
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        return self.get_logits(hidden_states), paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        return self.get_logits(hidden_states), paged_kv_cache

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ) -> Tensor:
        op_ext.configure()
        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            if self.tensor_parallel_shards > 1:
                logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        return self.batch_forward(input_embeds, paged_kv_cache, logit_positions), paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            # Only attention layers get KV cache slots (4 for the 4B model)
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            layer_partition=self.model.layer_partition,
            enable_disaggregation=self.disaggregation,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "get_logits": {
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
