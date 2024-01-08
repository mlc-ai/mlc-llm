"""Attention KV cache modeling."""
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.script import tir as T


class PagedKVCache(Object):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    def attention(  # pylint: disable=invalid-name
        self,
        layer_id: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Compute attention with the given q/k/v data and in-cache k/v data
        on the specified layer. Rotary position embeddings are applied to k/v
        within this function.

        - For prefill, the input q and output tensor have shape
        (1, total_seq_len, num_attention_heads, head_dim), and the
        k/v tensors have shape (1, total_seq_len, num_key_value_heads, head_dim).
        - For decode, the input q and output tensor have shape
        (batch_size, 1, num_attention_heads, head_dim), and the
        k/v tensors have shape (batch_size, 1, num_key_value_heads, head_dim).
        """
        # pylint: disable=protected-access
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.paged_attention_kv_cache_attention",
                    [
                        self._expr,
                        rx.PrimValue(layer_id),  # type: ignore[arg-type]
                        q._expr,
                        k._expr,
                        v._expr,
                    ],
                    out_sinfo=q._expr.struct_info,
                )
            )
        )
        # pylint: enable=protected-access


class FlashInferPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        page_size: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
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
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        """

        bb = rx.BlockBuilder.current()  # pylint: disable=invalid-name
        args = [
            rx.ShapeExpr([max_batch_size, max_total_seq_len, page_size]),
            rx.PrimValue(num_hidden_layers),
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(head_dim),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rx.op.zeros((), dtype),  # type: ignore[arg-type]
            bb.add_func(
                _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype),
                "kv_cache_transpose_append",
            ),
            rx.extern("paged_kv_cache.attention_kernel_prefill"),
            rx.extern("paged_kv_cache.attention_kernel_decode"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"),
            rx.extern("paged_kv_cache.attention_kernel_prefill_begin_forward"),
            rx.extern("paged_kv_cache.attention_kernel_prefill_end_forward"),
            rx.extern("paged_kv_cache.attention_kernel_decode_begin_forward"),
            rx.extern("paged_kv_cache.attention_kernel_decode_end_forward"),
            rx.extern("flashinfer.batch_qk_apply_rotary_in_place"),
            rx.extern("flashinfer.merge_state_in_place"),
            bb.add_func(
                _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype),
                "kv_cache_debug_get_kv",
            ),
        ]
        super().__init__(
            _expr=rx.Call(
                rx.extern("vm.builtin.paged_attention_kv_cache_create"),
                args=args,
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )


# mypy: disable-error-code="attr-defined"
# pylint: disable=too-many-locals


def _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype):
    """Return the TIR function that appends new k/v data to PagedKVCache."""

    # pylint: disable=line-too-long,invalid-name
    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        ntoken = T.SizeVar("ntoken", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype)
        k_data = T.match_buffer(var_k_data, (ntoken, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_key_value_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32")
        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf])
                position: T.int32 = position_map[vgpos]  # type: ignore
                pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf])
                position: T.int32 = position_map[vgpos] # type: ignore[name-defined,no-redef]
                pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf] = v_data[vgpos, vh, vf]
    # fmt: on
    # pylint: enable=line-too-long,invalid-name

    return tir_kv_cache_transpose_append


def _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    # pylint: disable=line-too-long,invalid-name
    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        seqlen = T.SizeVar("seqlen", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32")
        k_data = T.match_buffer(var_k_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        for p, h, d in T.grid(seqlen, num_key_value_heads, head_dim):
            with T.block("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                k_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd]
                v_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd]
    # fmt: on
    # pylint: enable=line-too-long,invalid-name

    return tir_kv_cache_debug_get_kv
