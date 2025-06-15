"""
Implementation for QWEN2MOE architecture.
"""

import dataclasses
from typing import Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.qwen3.qwen3_model import ACT2FN, Qwen3Attention, Qwen3Config
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Qwen3MoeConfig(Qwen3Config):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Qwen3Moe model."""

    moe_intermediate_size: int = 0
    num_experts_per_tok: int = 0
    num_experts: int = 0
    decoder_sparse_step: int = 0
    norm_topk_prob: bool = False


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: Optional[int] = None):
        intermediate_size = intermediate_size or config.intermediate_size
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen3MoeSparseMoeBlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """MoE layer for Qwen3MoE model."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        if config.moe_intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.moe_intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        )
        self.moe_gate_up_proj = MixtralExperts(
            self.num_experts,
            in_features=config.hidden_size,
            out_features=2 * self.moe_intermediate_size,
        )
        self.moe_down_proj = MixtralExperts(
            self.num_experts,
            in_features=self.moe_intermediate_size,
            out_features=config.hidden_size,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        def _expert_forward(x: Tensor, indptr: Tensor):
            x1_x2 = self.moe_gate_up_proj(x, indptr)
            x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
            x = self.moe_down_proj(self.act_fn(x1) * x2, indptr)
            return x

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_experts
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.reshape(num_tokens, hidden_size)
        gate = self.gate(x)
        # expert_weights: [num_tokens, experts_per_tok]
        # expert_indices: [num_tokens, experts_per_tok]
        expert_weights, expert_indices = op_ext.moe_misc.gating_softmax_topk(
            gate, experts_per_tok, norm_topk_prob=self.norm_topk_prob
        )
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
            moe_hidden_states.reshape(num_tokens, experts_per_tok, hidden_size) * expert_weights
        )
        # moe_hidden_states: [num_tokens, hidden_size]
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        final_hidden_states = moe_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_size)
        return final_hidden_states


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        assert (
            config.num_experts > 0 and config.decoder_sparse_step == 1
        ), "Currently only support use moe for every layer."
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_attention_heads * hd
            k = self.self_attn.num_key_value_heads * hd
            v = self.self_attn.num_key_value_heads * hd
            mi = self.mlp.moe_intermediate_size
            _set(
                self.self_attn.c_attn.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            if config.attention_bias:
                _set(
                    self.self_attn.c_attn.bias,
                    tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
                )
            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))
            _set(
                self.mlp.moe_gate_up_proj.weight,
                tp.ShardSingleDim("_shard_moe_mlp_up", segs=[mi, mi], dim=1),
            )
            _set(self.mlp.moe_down_proj.weight, tp.ShardSingleDim("_shard_moe_mlp_down", dim=2))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Qwen3MoeConfig):
        self.model = Qwen3MoeModel(config)
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
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.weight_block_size = config.weight_block_size

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
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
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
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
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
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
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
