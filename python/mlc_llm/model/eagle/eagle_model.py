"""
Implementation for EAGLE architecture.
"""

import dataclasses
from typing import Optional

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.llama.llama_model import LlamaAttention, LlamaConfig, LlamaFFN
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EagleConfig(LlamaConfig):
    """Configuration of the Eagle model."""

    bias: bool = True  # Whether to use bias in the fc layers


# pylint: disable=invalid-name,missing-docstring


class EagleDecoderLayer(nn.Module):
    def __init__(self, config: EagleConfig, index: int):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaFFN(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)
        out = self.self_attn(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class EagleForCasualLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: EagleConfig):
        # Put the model definition here to align with EAGLE's original structure
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = nn.Embedding("vocab_size", config.hidden_size)
        self.layers = nn.ModuleList(
            [EagleDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.fc = nn.Linear(
            in_features=2 * config.hidden_size, out_features=config.hidden_size, bias=config.bias
        )

        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def fuse_embed_hidden_states(self, input_embed: Tensor, hidden_states: Tensor):
        hidden_states = op.concat([input_embed, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states

    def forward_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        return hidden_states

    def forward(self, input_embed: Tensor, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = self.fuse_embed_hidden_states(input_embed, hidden_states)
        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return hidden_states

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.embed_tokens(input_ids)

    def prefill_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def decode_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_prefill_to_last_hidden_states(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        hidden_states = self.batch_forward(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_decode_to_last_hidden_states(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache
    ):
        hidden_states = self.batch_forward(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

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
            "fuse_embed_hidden_states": {
                "input_embed": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
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
