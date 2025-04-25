"""
Implementation for Deepseek V2 architecture
"""

import dataclasses
import math
from typing import Any, Dict, Literal, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import position_embedding

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.op import batch_matmul
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DeepseekV2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Deepseek V2 model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    first_k_dense_replace: int
    moe_layer_freq: int
    routed_scaling_factor: float
    scoring_func: str
    topk_method: Literal["greedy", "group_limited_greedy", "noaux_tc"]
    n_group: int
    topk_group: int
    attention_bias: bool
    kv_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    rms_norm_eps: float
    rope_theta: int
    q_lora_rank: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float32"
    max_batch_size: int = 1
    weight_block_size: Optional[Tuple[int, int]] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if "quantization_config" in self.kwargs:
            quantization_config = self.kwargs.get("quantization_config")
            if (
                isinstance(quantization_config, dict)
                and quantization_config.get("activation_scheme", "") == "dynamic"
                and quantization_config.get("fmt", "") == "e4m3"
                and quantization_config.get("quant_method", "") == "fp8"
                and "weight_block_size" in quantization_config
            ):
                self.weight_block_size = quantization_config.get("weight_block_size")
                if (
                    not isinstance(self.weight_block_size, (tuple, list))
                    or len(self.weight_block_size) != 2
                ):
                    raise ValueError(
                        "Invalid DeepSeek model quantization config: "
                        "weight_block_size must be a tuple of two integers, "
                        f"got {self.weight_block_size} of type {type(self.weight_block_size)}"
                    )
            else:
                raise ValueError(
                    "Invalid DeepSeek model quantization config: unrecognized quantization config: "
                    f"{quantization_config}"
                )
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
        assert self.num_attention_heads % self.num_key_value_heads == 0
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class DeepseekV2MLP(nn.Module):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        if intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = intermediate_size // config.tensor_parallel_shards

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2YarnRotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        self.rope_fn = position_embedding.switch_rope_freq_func(config.rope_scaling)
        self.rotary_dim = config.qk_rope_head_dim
        self.theta = config.rope_theta

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions: Tensor,
    ):
        def _rope_fused(x: te.Tensor, positions: te.Tensor):
            _, _, _, d_dim = x.shape
            d_dim_half = d_dim // 2
            dtype = x.dtype

            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                d1 = d // d_dim_half
                d2 = d % d_dim_half

                cos_freq, sin_freq, var_map = self.rope_fn(
                    positions[s], d, self.rotary_dim, self.theta, dtype
                )
                cos = x[b, s, h, d2 * 2 + d1] * cos_freq

                partner_d = tir.if_then_else(
                    d < self.rotary_dim // 2,
                    d + self.rotary_dim // 2,
                    d - self.rotary_dim // 2,
                )

                partner_d1 = partner_d // d_dim_half
                partner_d2 = partner_d % d_dim_half
                sin = (
                    x[b, s, h, partner_d2 * 2 + partner_d1]
                    * sin_freq
                    * tir.if_then_else(
                        d < self.rotary_dim // 2, tir.const(-1, dtype), tir.const(1, dtype)
                    )
                )
                expr = cos + sin
                for var, val in var_map.items():
                    expr = tir.Let(var, val, expr)
                return expr

            return te.compute(x.shape, compute, name="yarn_rope")

        q_embed = op.tensor_expr_op(_rope_fused, "rope", [q, positions])
        k_embed = op.tensor_expr_op(_rope_fused, "rope", [k, positions])
        return q_embed, k_embed


class DeepseekV2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_heads = config.num_attention_heads // config.tensor_parallel_shards

        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.block_size = config.weight_block_size

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, -1, config.rms_norm_eps, bias=False)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, -1, config.rms_norm_eps, bias=False)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.w_uk = nn.Parameter((self.num_heads, config.kv_lora_rank, self.qk_nope_head_dim))
        self.w_uv = nn.Parameter((self.num_heads, self.v_head_dim, config.kv_lora_rank))

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.rotary_emb = DeepseekV2YarnRotaryEmbedding(config)

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        query_positions: Tensor,
        forward_mode: Literal["prefill", "decode", "extend"],
    ) -> Tuple[Tensor, PagedKVCache]:
        b, s, _ = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )  # (b, s, num_heads * q_head_dim)
        q = op.reshape(q, (b, s, self.num_heads, self.q_head_dim))  # (b, s, num_heads, q_head_dim)
        q_nope, q_pe = op.split(
            q, [self.qk_nope_head_dim], axis=-1
        )  # (b, s, num_heads, qk_nope_head_dim), (b, s, num_heads, qk_rope_head_dim)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states).reshape(
            b, s, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )  # (b, s, 1, kv_lora_rank + qk_rope_head_dim)
        compressed_kv, k_pe = op.split(
            compressed_kv, [self.config.kv_lora_rank], axis=-1
        )  # (b, s, 1, kv_lora_rank), (b, s, 1, qk_rope_head_dim)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe, query_positions)
        kv_states = op.concat(
            [compressed_kv, k_pe], dim=-1
        )  # (b, s, 1, kv_lora_rank + qk_rope_head_dim)
        paged_kv_cache = paged_kv_cache.append_mla_kv(layer_id, kv_states)

        if forward_mode == "prefill":
            output, _ = self.self_attn(q_nope, compressed_kv, q_pe, k_pe, paged_kv_cache, layer_id)
        elif forward_mode == "decode":
            output, _ = self.cross_attn(q_nope, q_pe, paged_kv_cache, layer_id)
        elif forward_mode == "extend":
            o1, lse1 = self.self_attn(q_nope, compressed_kv, q_pe, k_pe, paged_kv_cache, layer_id)
            o2, lse2 = self.cross_attn(q_nope, q_pe, paged_kv_cache, layer_id)
            output, _ = paged_kv_cache.merge_attn_output_inplace(o1, lse1, o2, lse2)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode}")

        return self.o_proj(output.reshape(b, s, self.num_heads * self.v_head_dim)), paged_kv_cache

    def self_attn(  # pylint: disable=too-many-arguments
        self,
        q_nope: Tensor,
        compressed_kv: Tensor,
        q_pe: Tensor,
        k_pe: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
    ) -> Tuple[Tensor, Tensor]:
        b, s, _, _ = q_nope.shape
        q = op.concat(
            [q_nope, q_pe], dim=-1
        )  # (b, s, num_heads, qk_nope_head_dim + qk_rope_head_dim)
        kv = op.reshape(
            self.kv_b_proj(compressed_kv),
            (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
        )
        k, v = op.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_pe = op.broadcast_to(k_pe, (b, s, self.num_heads, self.qk_rope_head_dim))
        k = op.concat([k, k_pe], dim=-1)
        output, lse = paged_kv_cache.self_attention(layer_id, q, k, v, self.softmax_scale)
        return output, lse

    def cross_attn(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
    ) -> Tuple[Tensor, Tensor]:
        b, s, _, _ = q_nope.shape
        if not hasattr(self, "w_uk_scale_inv"):
            q_nope = op.matmul(
                q_nope.reshape(b * s, self.num_heads, self.qk_nope_head_dim).permute_dims(1, 0, 2),
                self.w_uk.permute_dims(0, 2, 1),
            )
        else:
            q_nope = batch_matmul.quantized_bmm(
                q_nope.reshape(b * s, self.num_heads, self.qk_nope_head_dim).permute_dims(1, 0, 2),
                self.w_uk,
                self.w_uk_scale_inv,  # pylint: disable=no-member
                self.block_size,
            )
        q_nope = q_nope.permute_dims(1, 0, 2).reshape(
            b, s, self.num_heads, self.kv_lora_rank
        )  # (b, s, num_heads, kv_lora_rank)
        query_states = op.concat(
            [q_nope, q_pe], dim=-1
        )  # (b, s, num_heads, kv_lora_rank + qk_rope_head_dim)

        output, lse = paged_kv_cache.cross_attention(
            layer_id,
            query_states,
            v_head_dim=self.kv_lora_rank,
            sm_scale=self.softmax_scale,
        )  # (b, s, num_heads, kv_lora_rank)
        if getattr(self, "w_uv_scale_inv", None) is None:
            output = op.matmul(
                output.reshape(b * s, self.num_heads, self.kv_lora_rank).permute_dims(1, 0, 2),
                self.w_uv.permute_dims(0, 2, 1),
            )
        else:
            output = batch_matmul.quantized_bmm(
                output.reshape(b * s, self.num_heads, self.kv_lora_rank).permute_dims(1, 0, 2),
                self.w_uv,
                self.w_uv_scale_inv,  # pylint: disable=no-member
                self.block_size,
            )
        output = output.permute_dims(1, 0, 2).reshape(b, s, self.num_heads * self.v_head_dim)
        return output, lse


class DeepseekV2MoE(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gate = nn.Linear(
            config.hidden_size, self.num_routed_experts, bias=False, out_dtype="float32"
        )
        self.e_score_correction_bias = (
            nn.Parameter((config.n_routed_experts,), dtype="float32")
            if config.topk_method == "noaux_tc"
            else None
        )
        self.norm_topk_prob = config.norm_topk_prob
        if config.moe_intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.moe_intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards

        self.moe_gate_up_proj = MixtralExperts(
            self.num_routed_experts,
            in_features=config.hidden_size,
            out_features=2 * self.moe_intermediate_size,
        )
        self.moe_down_proj = MixtralExperts(
            self.num_routed_experts,
            in_features=self.moe_intermediate_size,
            out_features=config.hidden_size,
        )

        self.shared_experts = DeepseekV2MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.dtype = "float32"

    def forward(self, x: Tensor):
        def _expert_forward(x: Tensor, indptr: Tensor):
            x1_x2 = self.moe_gate_up_proj(x, indptr)
            x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
            x = self.moe_down_proj(op.silu(x1) * x2, indptr)
            return x

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_routed_experts
        b, s, h = x.shape
        num_tokens = b * s
        x = op.reshape(x, (num_tokens, h))
        logits = self.gate(x)  # (num_tokens, num_routed_experts)
        assert logits.dtype == "float32"
        if self.scoring_func == "softmax":
            scores = op.softmax(logits, axis=-1)
        elif self.scoring_func == "sigmoid":
            scores = op.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported deepseek scoring function: {self.scoring_func}")

        # select top-k experts
        if self.topk_method == "greedy":
            expert_weights, expert_indices = op_ext.moe_misc.gating_topk(scores, experts_per_tok)
        elif self.topk_method in ["group_limited_greedy", "noaux_tc"]:
            expert_weights, expert_indices = op_ext.moe_misc.group_limited_greedy_topk(
                scores,
                self.num_experts_per_tok,
                self.num_routed_experts,
                self.n_group,
                self.topk_group,
                self.topk_method,
                num_tokens,
                self.e_score_correction_bias,
            )
        else:
            raise ValueError(f"Unsupported deepseek topk method: {self.topk_method}")

        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = op.sum(expert_weights, axis=-1, keepdims=True) + 1e-20
            expert_weights = expert_weights / denominator
        expert_weights = expert_weights * self.routed_scaling_factor

        use_ft = (
            (op_ext.get_store().cutlass_group_gemm or op_ext.get_store().faster_transformer)
            and self.dtype == "float16"
            and type(self.moe_gate_up_proj)  # pylint: disable=unidiomatic-typecheck
            is MixtralExperts
        )

        if num_tokens == 1:
            # x: [num_tokens * experts_per_tok, hidden_size]
            moe_hidden_states = _expert_forward(x, expert_indices)
        else:
            # cumsum: [num_tokens * local_experts]
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, num_experts)
            # indices: [num_tokens * experts_per_tok]
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            if use_ft:
                # indptr: [num_routed_experts]
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=True, out_dtype="int64"
                )
            else:
                # indptr: [num_routed_experts + 1]
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=False, out_dtype="int32"
                )
            # x: [num_tokens * experts_per_tok, hidden_size]
            moe_hidden_states = op.take(x, token_indices, axis=0)
            moe_hidden_states = _expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)

        # moe_hidden_states: [num_tokens, experts_per_tok, hidden_size]
        expert_weights = expert_weights.reshape(num_tokens, experts_per_tok, 1).astype(x.dtype)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, experts_per_tok, h) * expert_weights
        )
        # moe_hidden_states: [num_tokens, hidden_size]
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        shared_expert_hidden_states = self.shared_experts(x)

        final_hidden_states = moe_hidden_states + shared_expert_hidden_states
        final_hidden_states = op.reshape(final_hidden_states, (b, s, h))
        return final_hidden_states

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        # Force e_score_correction_bias to be float32
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias.to("float32")


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            if self.self_attn.q_lora_rank is None:
                _set(
                    self.self_attn.q_proj.weight,
                    tp.ShardSingleDim("_shard_q_weight", dim=0),
                )
            else:
                _set(
                    self.self_attn.q_b_proj.weight,
                    tp.ShardSingleDim("_shard_q_b_weight", dim=0),
                )

            _set(
                self.self_attn.kv_b_proj.weight,
                tp.ShardSingleDim("_shard_kv_b_weight", dim=0),
            )
            _set(
                self.self_attn.w_uk,
                tp.ShardSingleDim("_shard_kv_b_weight_w_uk", dim=0),
            )
            _set(
                self.self_attn.w_uv,
                tp.ShardSingleDim("_shard_kv_b_weight_w_uv", dim=0),
            )
            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))

            if isinstance(self.mlp, DeepseekV2MoE):
                si = self.mlp.shared_experts.intermediate_size
                mi = self.mlp.moe_intermediate_size
                _set(
                    self.mlp.shared_experts.gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_shared_experts_gate_up", segs=[si, si], dim=0),
                )
                _set(
                    self.mlp.shared_experts.down_proj.weight,
                    tp.ShardSingleDim("_shard_shared_experts_down", dim=1),
                )
                _set(
                    self.mlp.moe_gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_moe_gate_up", segs=[mi, mi], dim=1),
                )
                _set(self.mlp.moe_down_proj.weight, tp.ShardSingleDim("_shard_moe_mlp_down", dim=2))
            else:
                assert isinstance(self.mlp, DeepseekV2MLP)
                si = self.mlp.intermediate_size
                _set(
                    self.mlp.gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_gate_up", segs=[si, si], dim=0),
                )
                _set(
                    self.mlp.down_proj.weight,
                    tp.ShardSingleDim("_shard_down", dim=1),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        query_positions: Tensor,
        forward_mode: Literal["prefill", "decode", "extend"],
    ) -> Tuple[Tensor, PagedKVCache]:
        out = self.input_layernorm(hidden_states)
        out, paged_kv_cache = self.self_attn(
            out, paged_kv_cache, layer_id, query_positions, forward_mode
        )
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)  # type: ignore[operator]
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, paged_kv_cache

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class DeepseekV2Model(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        forward_mode: Literal["prefill", "decode", "extend"],
    ):
        hidden_states = inputs
        query_positions = paged_kv_cache.get_query_positions(inputs.shape[0] * inputs.shape[1])
        for layer_id, layer in enumerate(self.layers):
            hidden_states, paged_kv_cache = layer(
                hidden_states, paged_kv_cache, layer_id, query_positions, forward_mode
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, paged_kv_cache


class DeepseekV2ForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekV2Config):
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.weight_block_size = config.weight_block_size

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        forward_mode: Literal["prefill", "decode", "extend"],
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states, paged_kv_cache = self.model(input_embeds, paged_kv_cache, forward_mode)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "prefill")
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def extend(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "extend")
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "decode")
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits, paged_kv_cache = self.batch_forward(
            input_embeds, paged_kv_cache, "prefill", logit_positions
        )
        return logits, paged_kv_cache

    def batch_extend(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits, paged_kv_cache = self.batch_forward(
            input_embeds, paged_kv_cache, "extend", logit_positions
        )
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits, paged_kv_cache = self.batch_forward(input_embeds, paged_kv_cache, "decode", None)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits, paged_kv_cache = self.batch_forward(input_embeds, paged_kv_cache, "extend", None)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mla",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=1,
            qk_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            mla_original_qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            mla_original_v_head_dim=self.v_head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
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
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "extend": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_extend": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
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
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
