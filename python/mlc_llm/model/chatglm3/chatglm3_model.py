"""
Implementation for CHATGLM3 architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GLMConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the ChatGLM model."""

    hidden_size: int
    num_layers: int
    kv_channels: int
    num_attention_heads: int
    ffn_hidden_size: int
    layernorm_epsilon: float
    post_layer_norm: bool
    rmsnorm: bool
    add_bias_linear: bool
    add_qkv_bias: bool
    apply_query_key_layer_scaling: bool
    multi_query_attention: bool
    multi_query_group_num: int
    vocab_size: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.vocab_size == 0:
            for name in ["padded_vocab_size"]:
                if name in self.kwargs:
                    self.vocab_size = self.kwargs.pop(name)
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "seq_length"]:
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


class GLMAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GLMConfig):
        self.hidden_size = config.hidden_size
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads"
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.multi_query_attention = config.multi_query_attention
        self.num_key_value_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        ) // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(
            config.hidden_size,
            (2 * self.num_key_value_heads + self.num_heads) * self.head_dim,
            bias=config.add_bias_linear or config.add_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.add_bias_linear
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, h_q, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        attn_output = self.dense(output)
        return attn_output


class GLMMLP(nn.Module):
    def __init__(self, config: GLMConfig):
        if config.ffn_hidden_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split ffn hidden size {config.ffn_hidden_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.ffn_hidden_size = config.ffn_hidden_size // config.tensor_parallel_shards

        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            self.ffn_hidden_size * 2,
            bias=config.add_bias_linear,
        )
        self.dense_4h_to_h = nn.Linear(
            self.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )

        def swiglu(x):
            x = nn.chunk(x, 2, dim=-1)
            return nn.silu(x[0]) * x[1]

        self.activation_func = swiglu

    def forward(self, x):
        intermediate_parallel = self.dense_h_to_4h(x)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    def __init__(self, config: GLMConfig):
        self.self_attention = GLMAttention(config=config)
        self.mlp = GLMMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.layernorm_epsilon, bias=False
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.layernorm_epsilon, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attention.num_heads * hd
            k = self.self_attention.num_key_value_heads * hd
            v = self.self_attention.num_key_value_heads * hd
            _set(
                self.self_attention.query_key_value.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            if config.add_bias_linear or config.add_qkv_bias:
                _set(
                    self.self_attention.query_key_value.bias,
                    tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
                )
            _set(self.self_attention.dense.weight, tp.ShardSingleDim("_shard_dense_weight", dim=1))
            if config.add_bias_linear:
                _set(self.self_attention.dense.bias, tp.ShardSingleDim("_shard_dense_bias", dim=0))
            _set(
                self.mlp.dense_h_to_4h.weight,
                tp.ShardSingleDim("_shard_dense_h_to_4h_weight", dim=0),
            )
            if config.add_bias_linear:
                _set(
                    self.mlp.dense_h_to_4h.bias,
                    tp.ShardSingleDim("_shard_dense_h_to_4h_bias", dim=0),
                )
            _set(self.mlp.dense_4h_to_h.weight, tp.ShardSingleDim("_shard_dense_4h_to_h", dim=1))
            if config.add_bias_linear:
                _set(
                    self.mlp.dense_4h_to_h.bias,
                    tp.ShardSingleDim("_shard_dense_4h_to_h_bias", dim=1),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attention(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config: GLMConfig):
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList([GLMBlock(config) for _ in range(config.num_layers)])

        if self.post_layer_norm:
            if config.rmsnorm:
                self.final_layernorm = nn.RMSNorm(
                    config.hidden_size, -1, config.layernorm_epsilon, bias=False
                )
            else:
                self.final_layernorm = nn.LayerNorm(config.hidden_size, config.layernorm_epsilon)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class ChatGLMModel(nn.Module):
    def __init__(self, config: GLMConfig):
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = GLMTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        hidden_states = self.encoder(hidden_states, paged_kv_cache)
        return hidden_states


class ChatGLMForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GLMConfig):
        self.transformer = ChatGLMModel(config)
        self.num_hidden_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        )
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.rope_theta = 10000
        self.tensor_parallel_shards = config.tensor_parallel_shards
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

        hidden_states = self.transformer(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.transformer.output_layer(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.transformer.embedding(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.transformer(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.transformer.output_layer(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.transformer(input_embed, paged_kv_cache)
        logits = self.transformer.output_layer(hidden_states)
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
