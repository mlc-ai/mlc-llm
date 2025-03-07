"""
Implementation for Deepseek architecture.
"""

import dataclasses
from functools import partial
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DeepseekConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Deepseek model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    moe_layer_freq: int
    first_k_dense_replace: int
    hidden_act: str
    norm_topk_prob: bool
    attention_bias: bool
    rms_norm_eps: float
    use_cache: bool
    bos_token_id: int
    eos_token_id: int
    tie_word_embeddings: bool = False
    rope_theta: int = 10000
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    max_batch_size: int = 1
    num_experts_per_tok: int = 0
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
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size
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


# pylint: disable=invalid-name,missing-docstring


class DeepseekAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekConfig):
        super().__init__()  # Make sure to call the parent class constructor
        self.hidden_size = config.hidden_size
        self.rope_theta = config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )

        self.attention_bias = config.attention_bias
        self.num_heads = config.num_attention_heads // self.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // self.tensor_parallel_shards
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.context_window_size

        self.wqkv_pack = nn.Linear(
            in_features=self.hidden_size,
            out_features=(self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=self.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = self.wqkv_pack(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        attn_output = self.o_proj(output)
        return attn_output


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class DeepseekMLP(nn.Module):
    def __init__(self, config: DeepseekConfig, intermediate_size=None):
        self.hidden_size = config.hidden_size
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        ) // config.tensor_parallel_shards

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


class DeepseekMoE(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekConfig):
        self.num_local_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.norm_topk_prob = config.norm_topk_prob
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards
        self.moe_gate_up_proj = MixtralExperts(
            self.num_local_experts,
            in_features=config.hidden_size,
            out_features=2 * self.moe_intermediate_size,
            tensor_parallel_shards=config.tensor_parallel_shards,
        )
        self.moe_down_proj = MixtralExperts(
            self.num_local_experts,
            in_features=self.moe_intermediate_size,
            out_features=config.hidden_size,
            tensor_parallel_shards=config.tensor_parallel_shards,
        )
        self.dtype = "float32"

        if config.n_shared_experts is not None:
            intermediate_size = self.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config, intermediate_size=intermediate_size)

    def forward(self, x: Tensor):  # pylint: disable=too-many-locals
        def _expert_forward(x: Tensor, indptr: Tensor):
            x1_x3 = self.moe_gate_up_proj(x, indptr)
            x1, x3 = op.split(x1_x3, indices_or_sections=2, axis=-1)
            x = self.moe_down_proj(op.silu(x1) * x3, indptr)
            return x

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_local_experts
        b, s, h = x.shape
        num_tokens = b * s
        x = op.reshape(x, (num_tokens, h))
        gate = self.gate(x)  # (b * s, num_routed_experts)
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
            moe_hidden_states.reshape(num_tokens, experts_per_tok, h) * expert_weights
        )
        # moe_hidden_states: [num_tokens, hidden_size]
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        shared_expert_hidden_states = self.shared_experts(x)

        final_hidden_states = moe_hidden_states + shared_expert_hidden_states
        final_hidden_states = op.reshape(final_hidden_states, (b, s, h))
        return final_hidden_states


class DeepseekDecoderLayer(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekConfig, layer_idx: int):
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.self_attn = DeepseekAttention(config)
        self.num_experts = config.n_routed_experts
        self.mlp = (
            DeepseekMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekMLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_heads * hd
            k = self.self_attn.num_key_value_heads * hd
            v = self.self_attn.num_key_value_heads * hd

            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            ):
                i = self.mlp.moe_intermediate_size
            else:
                i = self.mlp.intermediate_size
            _set(
                self.self_attn.wqkv_pack.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))

            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            ):
                _set(
                    self.mlp.moe_gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=1),
                )
                _set(self.mlp.moe_down_proj.weight, tp.ShardSingleDim("_shard_mlp_down", dim=2))

            else:
                _set(
                    self.mlp.gate_up_proj.weight,
                    tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0),
                )
                _set(self.mlp.down_proj.weight, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)  # type: ignore[operator]
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class DeepseekModel(nn.Module):
    def __init__(self, config: DeepseekConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepseekDecoderLayer(config, layer_idx)
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


class DeepseekForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: DeepseekConfig):
        self.model = DeepseekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.dtype = "float32"

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
