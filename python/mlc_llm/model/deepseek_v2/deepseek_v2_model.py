"""
Implementation for Deepseek V2 architecture
"""

import dataclasses
import math
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import position_embedding
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
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
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
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
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)

        if self.tensor_parallel_shards != 1:
            raise ValueError("Only support single device at this moment.")


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class DeepseekV2MLP(nn.Module):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

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
        def _rope(x: te.Tensor, positions: te.Tensor):
            dtype = x.dtype

            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                cos_freq, sin_freq, var_map = self.rope_fn(
                    positions[s], d, self.rotary_dim, self.theta, dtype
                )
                cos = cos_freq * x[b, s, h, d]
                sin = sin_freq * tir.if_then_else(
                    d < self.rotary_dim // 2,
                    -x[b, s, h, d + self.rotary_dim // 2],
                    x[b, s, h, d - self.rotary_dim // 2],
                )
                expr = cos + sin
                for var, value in var_map.items():
                    expr = tir.Let(var, value, expr)
                return expr

            return te.compute(x.shape, compute, name="yarn_rope")

        b, s, h, d = q.shape
        q = op.reshape(
            op.permute_dims(op.reshape(q, (b, s, h, d // 2, 2)), [0, 1, 2, 4, 3]), (b, s, h, d)
        )

        b, s, h, d = k.shape
        k = op.reshape(
            op.permute_dims(op.reshape(k, (b, s, h, d // 2, 2)), [0, 1, 2, 4, 3]), (b, s, h, d)
        )

        q_embed = op.tensor_expr_op(_rope, "rope", [q, positions])
        k_embed = op.tensor_expr_op(_rope, "rope", [k, positions])
        return q_embed, k_embed


class DeepseekV2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

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

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        b, s, _ = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )  # (b, s, num_heads * q_head_dim)
        q = op.reshape(q, (b, s, self.num_heads, self.q_head_dim))  # (b, s, num_heads, q_head_dim)
        _, q_pe = op.split(
            q, [self.qk_nope_head_dim], axis=-1
        )  # (b, s, num_heads, qk_nope_head_dim), (b, s, num_heads, qk_rope_head_dim)

        compressed_kv = self.kv_a_proj_with_mqa(
            hidden_states
        )  # (b, s, kv_lora_rank + qk_rope_head_dim)
        compressed_kv, k_pe = op.split(
            compressed_kv, [self.config.kv_lora_rank], axis=-1
        )  # (b, s, kv_lora_rank), (b, s, qk_rope_head_dim)

        k_pe = op.reshape(k_pe, (b, s, 1, self.qk_rope_head_dim))
        kv = self.kv_b_proj(
            self.kv_a_layernorm(compressed_kv)
        )  # (b, s, num_heads * (qk_nope_head_dim + v_head_dim))
        kv = op.reshape(kv, (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))

        k_nope, value_states = op.split(
            kv, [self.qk_nope_head_dim], axis=-1
        )  # (b, s, num_heads, qk_nope_head_dim), (b, s, num_heads, v_head_dim)

        q_pe, k_pe = self.rotary_emb(q_pe, k_pe, paged_kv_cache.get_query_positions(s))

        @T.prim_func
        def inplace_q(var_q: T.handle, var_pe: T.handle):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            b = T.int64(is_size_var=True)
            s = T.int64(is_size_var=True)
            q_data = T.match_buffer(var_q, (b, s, self.num_heads, self.q_head_dim), q.dtype)
            pe_data = T.match_buffer(var_pe, (b, s, self.num_heads, self.qk_rope_head_dim), q.dtype)

            for iters in T.grid(b, s, self.num_heads, self.q_head_dim):
                with T.block("T_inplace_q"):
                    vb, vs, vh, vq = T.axis.remap("SSSS", iters)
                    T.reads(pe_data[vb, vs, vh, vq - self.qk_nope_head_dim])
                    T.writes(q_data[vb, vs, vh, vq])
                    if vq >= self.qk_nope_head_dim:
                        q_data[vb, vs, vh, vq] = pe_data[vb, vs, vh, vq - self.qk_nope_head_dim]

        query_states = op.tensor_ir_inplace_op(
            inplace_q,
            "concat_q",
            args=[q, q_pe],
            inplace_indices=[0],
            out=Tensor.placeholder(q.shape, q.dtype),
        )  # (b, s, num_heads, q_head_dim)

        def concat_k(var_k_nope: T.handle, var_pe: T.handle):
            return te.compute(
                (b, s, self.num_heads, self.q_head_dim),
                lambda _b, _s, _h, _d: te.if_then_else(
                    _d < self.qk_nope_head_dim,
                    var_k_nope[_b, _s, _h, _d],
                    var_pe[_b, _s, 0, _d - self.qk_nope_head_dim],
                ),
            )

        key_states = op.tensor_expr_op(
            concat_k, "concat_k", [k_nope, k_pe]
        )  # (b, s, num_heads, q_head_dim)

        q_pad = op.pad(query_states, [0, 0, 0, 0, 0, 0, 0, 256 - self.q_head_dim])
        k_pad = op.pad(key_states, [0, 0, 0, 0, 0, 0, 0, 256 - self.q_head_dim])
        v_pad = op.pad(value_states, [0, 0, 0, 0, 0, 0, 0, 256 - self.v_head_dim])

        qkv = op.concat([q_pad, k_pad, v_pad], dim=2)  # (b, s, 3 * num_heads, 256)
        output = op.split(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id,
                qkv,
                self.num_heads,
                self.softmax_scale
                * math.sqrt(256),  # This is to cancel out the 1/sqrt(d) in normal attention
            ),
            [self.v_head_dim],
            axis=-1,
        )[0].reshape(b, s, self.num_heads * self.v_head_dim)

        return self.o_proj(output)


class DeepseekV2MoE(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        self.norm_topk_prob = config.norm_topk_prob
        self.moe_intermediate_size = config.moe_intermediate_size

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

        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV2MLP(config, intermediate_size=intermediate_size)

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
        gate = self.gate(x)  # (b * s, num_routed_experts)
        expert_weights, expert_indices = op_ext.moe_misc.gating_softmax_topk(
            gate, experts_per_tok, norm_topk_prob=self.norm_topk_prob
        )
        expert_weights = expert_weights * self.routed_scaling_factor

        if num_tokens == 1:
            # x: [num_tokens * experts_per_tok, hidden_size]
            moe_hidden_states = _expert_forward(x, expert_indices)
        else:
            # cumsum: [num_tokens * local_experts]
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, num_experts)
            # indices: [num_tokens * experts_per_tok]
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            # indptr: [num_local_experts + 1]
            indptr = op_ext.moe_misc.get_indptr(
                cumsum, num_experts, num_tokens, inclusive=False, out_dtype="int32"
            )
            # x: [num_tokens * experts_per_tok, hidden_size]
            moe_hidden_states = op.take(x, token_indices, axis=0)
            moe_hidden_states = _expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)

        # moe_hidden_states: [num_tokens, experts_per_tok, hidden_size]
        expert_weights = expert_weights.reshape(num_tokens, experts_per_tok, 1)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, experts_per_tok, h) * expert_weights
        )
        # moe_hidden_states: [num_tokens, hidden_size]
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        shared_expert_hidden_states = self.shared_experts(x)

        final_hidden_states = moe_hidden_states + shared_expert_hidden_states
        final_hidden_states = op.reshape(final_hidden_states, (b, s, h))
        return final_hidden_states


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

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id)
        hidden_states = hidden_states + out
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)  # type: ignore[operator]
        hidden_states = hidden_states + out
        return hidden_states


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

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


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
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
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
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=256,
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
