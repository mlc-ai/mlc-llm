"""Shared helpers for causal LM models."""

# pylint: disable=missing-function-docstring,no-member

# pylint: disable=missing-function-docstring,no-member

from __future__ import annotations

from typing import Any, Optional, cast

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn.kv_cache import PagedKVCache, RopeMode


def index_last_token(x: te.Tensor) -> te.Tensor:
    """Return the last token slice along sequence dimension."""
    b, s, d = x.shape
    return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")


class BaseForCausalLM(
    nn.Module
):  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """Shared default implementations for causal LM models."""

    dtype: str
    model: nn.Module
    lm_head: nn.Module
    tensor_parallel_shards: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: int
    rope_scale: int
    rope_mode: RopeMode
    rope_scaling: Optional[dict]
    rotary_dim: Optional[int]
    layer_partition: Optional[list]
    disaggregation: bool

    def __init__(self):
        super().__init__()
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def _get_backbone(self):
        return self.model

    def _get_embed_module(self):
        return self._get_backbone().embed_tokens

    def _get_lm_head(self):
        return self.lm_head

    def _backbone_forward(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self._get_backbone()(input_embeds, paged_kv_cache)

    def get_logits(self, hidden_states: Tensor):
        logits = self._get_lm_head()(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self._get_embed_module()(input_ids)

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self._backbone_forward(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self._backbone_forward(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(
            index_last_token,
            name_hint="index",
            args=[hidden_states],
        )
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self._backbone_forward(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
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
        num_key_value_heads = getattr(self, "num_key_value_heads", self.num_attention_heads)
        rope_mode = getattr(self, "rope_mode", RopeMode.NORMAL)
        rope_scale = getattr(self, "rope_scale", 1)
        rope_theta = getattr(self, "rope_theta", 10000)
        rope_scaling = getattr(self, "rope_scaling", None)
        rotary_dim = getattr(self, "rotary_dim", None)
        layer_partition = getattr(self, "layer_partition", None)
        if layer_partition is None and hasattr(self, "model"):
            layer_partition = getattr(self.model, "layer_partition", None)
        enable_disaggregation = getattr(self, "disaggregation", False)
        attn_kind = cast(Any, getattr(self, "attn_kind", "mha"))

        return PagedKVCache.create_generic(
            attn_kind=attn_kind,
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=rope_mode,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rotary_dim=rotary_dim,
            layer_partition=layer_partition,
            enable_disaggregation=enable_disaggregation,
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
