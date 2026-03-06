# pylint: disable=missing-function-docstring,no-member

from __future__ import annotations

from typing import Optional

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn.kv_cache import PagedKVCache


def index_last_token(x: te.Tensor) -> te.Tensor:
    """Return the last token slice along sequence dimension."""
    b, s, d = x.shape
    return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")


class CausalLMABC(nn.Module):
    """
    Minimal base class for causal-LM models.

    Subclasses are expected to define:
    * ``dtype``  (str attribute)
    * ``hidden_size``  (int attribute)
    * ``tensor_parallel_shards``  (int attribute)
    * ``get_logits(hidden_states) -> Tensor``
    * ``create_paged_kv_cache(...) -> PagedKVCache``
    * ``get_default_spec() -> ModuleSpec``

    Subclasses may also override the hooks ``_get_backbone()``,
    ``_get_embed_module()``, and ``_backbone_forward()`` to customise which
    sub-modules are used by the concrete helper methods.
    """

    # These are expected on every subclass but are set by the subclass __init__,
    # not here. Declared for documentation / IDE support only.
    dtype: str
    hidden_size: int
    tensor_parallel_shards: int

    # Overridable hooks

    def _get_backbone(self) -> nn.Module:
        """Return the transformer backbone (default: ``self.model``)."""
        return self.model  # type: ignore[attr-defined]

    def _get_embed_module(self) -> nn.Module:
        """Return the embedding table (default: ``backbone.embed_tokens``)."""
        return self._get_backbone().embed_tokens

    def _backbone_forward(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        return self._get_backbone()(input_embeds, paged_kv_cache)

    # Methods common to all LMs

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self._get_embed_module()(input_ids)

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
