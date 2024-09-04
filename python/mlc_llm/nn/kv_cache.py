"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments
import json
from typing import Any, Dict, List, Optional

import numpy as np
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode

from tvm.relax.frontend.nn.llm.kv_cache import (
    _attention_decode,
    _attention_prefill,
    _attention_prefill_ragged,
    _compact_kv_copy,
    _copy_single_page,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace,
    llama_rope_with_position_map,
    tree_attn,
    tree_attn_with_paged_kv_cache,
)

class PagedKVCache(TVMPagedKVCache):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(  # pylint: disable=too-many-locals
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        rotary_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_ext_factors: Optional[List[int]] = None,
        layer_partition: Optional[List[int]] = None,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = head_dim
        if rope_scaling is None:
            rope_scaling = {}
        if layer_partition is None:
            layer_partition = [0, num_hidden_layers]
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
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
                rx.PrimValue(head_dim),
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
                rx.DataTypeImm(dtype),
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )

    def attention_with_fused_qkv(  # pylint: disable=invalid-name
        self,
        layer_id: int,
        qkv: Tensor,
        num_qo_heads: int,
        attn_score_scaling_factor: float = 1.0,
    ) -> Tensor:
        """Compute attention with the given fused q/k/v data and in-cache k/v data
        on the specified layer. Rotary position embeddings are applied to k/v
        within this function.

        - For prefill, the input qkv and output tensor have shape
        (1, total_seq_len) for the first two dimensions.
        - For decode, the input qkv and output tensor have shape
        (batch_size, 1) for the first two dimensions.
        - The input qkv have `2 * num_qo_heads + num_kv_heads` at the third dim.
        - The output tensor have `num_qo_heads` at the third dim.
        - The input qkv and output tensor have `head_dim` at the last dim.
        """
        # pylint: disable=protected-access
        b, s, _, d = qkv._expr.struct_info.shape
        qkv = qkv.reshape(b * s, qkv.shape[2], d)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    [
                        self._expr,
                        rx.PrimValue(layer_id),  # type: ignore[arg-type]
                        rx.PrimValue(attn_score_scaling_factor),
                        qkv._expr,
                    ],
                    out_sinfo=rx.TensorStructInfo((b * s, num_qo_heads, d), qkv.dtype),
                )
            )
        ).reshape(b, s, num_qo_heads, d)

    def get_query_positions(self, total_length: tir.PrimExpr) -> Tensor:
        """Get the in-sequence positions of each slot in the query,
        which are needed for applying positional embeddings in some models.

        Parameters
        ----------
        total_length : tir.PrimExpr
            The summed-up total sequence length of queries in
            the batch being forwarded.

        Returns
        -------
        q_positions : Tensor
            The in-sequence query positions, in shape `(total_length,)`
        """
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_pure_packed(
                    "vm.builtin.attention_kv_cache_get_query_positions",
                    self._expr,
                    sinfo_args=rx.TensorStructInfo((total_length,), "int32"),
                )
            )
        )

    # pylint: enable=protected-access


class FlashInferPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: Dict[str, Any],
        rotary_dim: int,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
    ) -> None:
        """Create a paged KV cache object with FlashInfer kernels.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tir.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tir.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tir.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        """
        if rope_mode == RopeMode.INLINE:
            assert rotary_dim == head_dim, "FlashInfer RoPE does not support partial rotary dim."

        bb = rx.BlockBuilder.current()  # pylint: disable=invalid-name
        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                ]
            ),
            layer_partition,
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(head_dim),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, head_dim, dtype), "kv_cache_transpose_append"),
            rx.extern("flashinfer.attention_kernel_prefill_with_paged_kv_cache"),
            rx.extern("flashinfer.attention_kernel_decode_with_paged_kv_cache"),
            bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype, True, rope_scaling, target), "tir_attention_prefill_sliding_window"),
            bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"),
            rx.extern("flashinfer.attention_kernel_prefill_with_paged_kv_cache_begin_forward"),
            rx.extern("flashinfer.attention_kernel_prefill_with_paged_kv_cache_end_forward"),
            rx.extern("flashinfer.attention_kernel_decode_with_paged_kv_cache_begin_forward"),
            rx.extern("flashinfer.attention_kernel_decode_with_paged_kv_cache_end_forward"),
            rx.extern("flashinfer.merge_state_in_place"),
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
            bb.add_func(_copy_single_page(num_key_value_heads, page_size, head_dim, dtype, target), "kv_cache_copy_single_page"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype), "kv_cache_debug_get_kv"),
            bb.add_func(_compact_kv_copy(num_key_value_heads, head_dim, dtype, target), "kv_cache_compact_kv_copy"),
            bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask"),
            bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_paged_kv_cache"),
            # fmt: on
            # pylint: enable=line-too-long
        ]
        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create",
                *args,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )


class TIRPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using TIR kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_mode: RopeMode,
        head_dim: int,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: Dict[str, Any],
        rotary_dim: int,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
    ) -> None:
        """Create a paged KV cache object with TIR kernels.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tir.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tir.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tir.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        target : Target
            The target to build the model to.
        """

        bb = rx.BlockBuilder.current()
        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                ]
            ),
            layer_partition,
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(head_dim),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype, False, rope_scaling, target), "tir_attention_prefill"),
            bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype, False, rope_scaling, target), "tir_attention_decode"),
            bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype, True, rope_scaling, target), "tir_attention_prefill_sliding_window"),
            bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window"),
            bb.add_func(_attention_prefill_ragged(num_key_value_heads, num_attention_heads, head_dim, dtype, rope_scaling, target), "tir_attention_prefill_ragged"),
            bb.add_func(_merge_state_inplace(num_attention_heads, head_dim, dtype, target), "tir_attention_merge_state"),
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
            bb.add_func(_copy_single_page(num_key_value_heads, page_size, head_dim, dtype, target), "kv_cache_copy_single_page"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype), "kv_cache_debug_get_kv"),
            bb.add_func(_compact_kv_copy(num_key_value_heads, head_dim, dtype, target), "kv_cache_compact_kv_copy"),
            bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask"),
            bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_paged_kv_cache"),
            # fmt: on
            # pylint: enable=line-too-long
        ]
        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create_reduced",
                *args,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )


# mypy: disable-error-code="attr-defined,valid-type,no-redef"
# pylint: disable=too-many-locals
