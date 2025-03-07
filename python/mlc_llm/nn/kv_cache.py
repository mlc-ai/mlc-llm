"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments
import json
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode


class PagedKVCache(TVMPagedKVCache):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(  # pylint: disable=too-many-locals
        attn_kind: Literal["mha", "mla"],
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
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
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_ext_factors: Optional[List[int]] = None,
        layer_partition: Optional[List[int]] = None,
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
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
                rx.StringImm(attn_kind),
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
