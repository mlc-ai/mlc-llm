"""Attention KV cache modeling."""

import json
from typing import Any, Dict, List, Literal, Optional, Union  # noqa: UP035

import numpy as np
from tvm import relax as rx
from tvm import tirx
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode


class PagedKVCache(TVMPagedKVCache):
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(
        attn_kind: Union[Literal["mha", "mla"], List[Literal["mha", "mla", "mha_sliding"]]],  # noqa: UP006
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        mla_original_qk_head_dim: int = 0,
        mla_original_v_head_dim: int = 0,
        rotary_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,  # noqa: UP006
        rope_ext_factors: Optional[List[int]] = None,  # noqa: UP006
        layer_partition: Optional[List[int]] = None,  # noqa: UP006
        enable_disaggregation: bool = False,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a multi-head attention PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = qk_head_dim
        if rope_scaling is None:
            rope_scaling = {}
        if layer_partition is None:
            layer_partition = [0, num_hidden_layers]
        if isinstance(attn_kind, List):  # noqa: UP006
            rx_attn_kind = [rx.StringImm(layer_kind) for layer_kind in attn_kind]
        else:
            rx_attn_kind = rx.StringImm(attn_kind)
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
                rx_attn_kind,
                rx.ShapeExpr(
                    [
                        max_batch_size,
                        max_total_seq_len,
                        prefill_chunk_size,
                        page_size,
                        support_sliding_window,
                    ]
                ),
                rx.ShapeExpr(layer_partition),
                rx.PrimValue(num_hidden_layers),
                rx.PrimValue(num_attention_heads),
                rx.PrimValue(num_key_value_heads),
                rx.PrimValue(qk_head_dim),
                rx.PrimValue(v_head_dim),
                rx.PrimValue(mla_original_qk_head_dim),
                rx.PrimValue(mla_original_v_head_dim),
                rx.PrimValue(rope_mode),
                rx.PrimValue(rope_scale),
                rx.PrimValue(rope_theta),
                rx.StringImm(json.dumps(rope_scaling)),
                (
                    rx.const(np.array(rope_ext_factors, "float32"))
                    if rope_ext_factors is not None
                    else rx.PrimValue(0)
                    # NOTE: since relax does not have "Optional" type, we use PrimValue(0)
                    # to represent "undefined".
                ),
                rx.PrimValue(rotary_dim),
                rx.PrimValue(int(enable_disaggregation)),
                rx.DataTypeImm(dtype),
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )
