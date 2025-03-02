"""
Implementation for GPTNeoX architecture.
"""

import dataclasses
import logging
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTNeoXConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPTNeoX model."""

    use_parallel_residual: bool
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    layer_norm_eps: float
    vocab_size: int
    rotary_pct: float
    position_embedding_base: int = 0
    context_window_size: int = 0
    head_dim: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    ffn_out_dtype: str = "float32"
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
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
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


class GPTNeoXAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GPTNeoXConfig):
        self.rope_theta = config.position_embedding_base
        self.hidden_size = config.hidden_size
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(
            in_features=self.hidden_size,
            out_features=3 * self.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=True
        )

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape

        # q/k/v states: [batch_size, seq_len, hidden_size]
        qkv = self.query_key_value(hidden_states)
        qkv = op.reshape(qkv, (batch_size, seq_len, 3 * self.num_attention_heads, self.head_dim))

        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim**-0.5
            ),
            (batch_size, seq_len, self.head_dim * self.num_attention_heads),
        )
        attn_output = self.dense(output)
        return attn_output


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__()
        out_dtype = config.ffn_out_dtype
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            out_dtype=out_dtype,
        )
        self.dense_4h_to_h = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            out_dtype=out_dtype,
        )

    def forward(self, hidden_states: Tensor):
        dtype = hidden_states.dtype
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = op.gelu(hidden_states)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        hidden_states = self.dense_4h_to_h(hidden_states)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.astype(dtype)
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.use_parallel_residual = config.use_parallel_residual

        def _set_tp():
            def _set(param, hint):
                param.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = k = v = self.attention.num_attention_heads * hd
            _set(
                self.attention.query_key_value.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            _set(
                self.attention.query_key_value.bias,
                tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
            )
            _set(self.attention.dense.weight, tp.ShardSingleDim("_shard_dense", dim=1))
            _set(
                self.mlp.dense_h_to_4h.weight,
                tp.ShardSingleDim("_shard_dense_h_to_4h_weight", dim=0),
            )
            _set(self.mlp.dense_h_to_4h.bias, tp.ShardSingleDim("_shard_dense_h_to_4h_bias", dim=0))
            _set(self.mlp.dense_4h_to_h.weight, tp.ShardSingleDim("_shard_dense_4h_to_h", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        dtype = hidden_states.dtype
        attn_input = self.input_layernorm(hidden_states)
        with tp.shard_bias(self.attention.dense, self.tensor_parallel_shards):
            attn_output = self.attention(
                attn_input,
                paged_kv_cache,
                layer_id,
            )
        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = self._apply_residual(attn_output, hidden_states)
            mlp_input = self.post_attention_layernorm(attn_output)
            with tp.shard_bias(self.mlp.dense_4h_to_h, self.tensor_parallel_shards):
                mlp_output = self.mlp(mlp_input)
            hidden_states = self._apply_residual(mlp_output.astype(dtype), attn_output)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out + residual / self.tensor_parallel_shards, "sum")
        return out + residual


class GPTNeoXModel(nn.Module):
    def __init__(self, config: GPTNeoXConfig):
        self.embed_in = nn.Embedding(num="vocab_size", dim=config.hidden_size)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs

        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTNeoXForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: GPTNeoXConfig):
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(
            in_features=config.hidden_size,
            out_features="vocab_size",
            bias=False,
            dtype="float32",
        )
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"
        self.rotary_pct = config.rotary_pct

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

        hidden_states = self.gpt_neox(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.embed_out(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.gpt_neox.embed_in(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.gpt_neox(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.embed_out(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.gpt_neox(input_embed, paged_kv_cache)
        logits = self.embed_out(hidden_states)
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
            num_key_value_heads=self.num_attention_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
            rotary_dim=int(self.head_dim * self.rotary_pct),
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
