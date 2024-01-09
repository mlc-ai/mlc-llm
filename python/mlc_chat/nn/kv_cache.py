"""Attention KV cache modeling."""
# pylint: disable=too-many-statements
import math
from typing import Tuple, Union

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


class TIRPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using TIR kernels."""

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
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        """

        bb = rx.BlockBuilder.current()
        args = [
            rx.ShapeExpr([max_batch_size, max_total_seq_len, page_size]),
            rx.PrimValue(num_hidden_layers),
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(head_dim),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rx.op.zeros((), dtype),
            bb.add_func(
                _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype),
                "kv_cache_transpose_append",
            ),
            bb.add_func(
                _attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype),
                "tir_attention_prefill",
            ),
            bb.add_func(
                _attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype),
                "tir_attention_decode",
            ),
            bb.add_func(_qk_rotary_inplace(dtype), "tir_qk_rotary_inplace"),
            bb.add_func(
                _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype),
                "kv_cache_debug_get_kv",
            ),
        ]
        super().__init__(
            _expr=rx.Call(
                rx.extern("vm.builtin.paged_attention_kv_cache_create_reduced"),
                args=args,
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )


# mypy: disable-error-code="attr-defined,valid-type,no-redef"
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


def _qk_rotary_inplace(dtype):
    # pylint: disable=line-too-long,invalid-name,too-many-arguments,too-many-nested-blocks
    # fmt: off
    @T.prim_func
    def tir_rotary(
        q_data: T.handle,
        k_data: T.handle,
        cur_append_length_indptr_handle: T.handle,
        cur_rope_offset_handle: T.handle,
        cur_batch_size: T.int32,
        num_qo_heads: T.int32,
        num_kv_heads: T.int32,
        head_dim: T.int32,
        qkv_layout: T.int32,  # pylint: disable=unused-argument
        rotary_scale: T.float32,
        rotary_theta: T.float32
    ):
        total_seq_length = T.int32()
        T.func_attr({"tir.is_scheduled": 1})
        Q = T.match_buffer(q_data, (total_seq_length, num_qo_heads, head_dim), dtype)
        K = T.match_buffer(k_data, (total_seq_length, num_kv_heads, head_dim), dtype)
        cur_append_length_indptr = T.match_buffer(cur_append_length_indptr_handle, (cur_batch_size + 1,), "int32")
        cur_rope_offset = T.match_buffer(cur_rope_offset_handle, (cur_batch_size,), "int32")
        for b_h_fused in T.thread_binding(cur_batch_size * (num_qo_heads + num_kv_heads), thread="blockIdx.x"):
            cur_batch_seqlen_start: T.int32 = cur_append_length_indptr[b_h_fused // (num_qo_heads + num_kv_heads)]
            offset_start: T.int32 = cur_rope_offset[b_h_fused // (num_qo_heads + num_kv_heads)]
            cur_append_length: T.int32 = cur_append_length_indptr[b_h_fused // (num_qo_heads + num_kv_heads) + 1] - cur_append_length_indptr[b_h_fused // (num_qo_heads + num_kv_heads)]
            for i_0 in range((cur_append_length + 31) // 32):
                for i_1 in T.thread_binding(32, thread="threadIdx.y"):
                    for k_0 in T.thread_binding((head_dim + 3) // 4, thread="threadIdx.x"):
                        for k_1 in T.vectorized(4):
                            if b_h_fused % (num_qo_heads + num_kv_heads) < num_qo_heads:
                                with T.block("q_rope"):
                                    vi = T.axis.spatial(cur_append_length, i_0 * 32 + i_1)
                                    vj = T.axis.spatial(num_qo_heads, b_h_fused % (num_qo_heads + num_kv_heads))
                                    vk = T.axis.spatial(head_dim, k_0 * 4 + k_1)
                                    T.where(k_0 * 4 + k_1 < head_dim and i_0 * 32 + i_1 < cur_append_length)
                                    T.reads(Q[cur_batch_seqlen_start + vi, vj, T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2):T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2) + (T.max(T.max(vk, head_dim // 2 + vk), vk - head_dim // 2) + 1 - T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2))])
                                    T.writes(Q[cur_batch_seqlen_start + vi, vj, vk])
                                    Q[cur_batch_seqlen_start + vi, vj, vk] = T.Cast(dtype, T.cos(T.Cast("float32", offset_start + vi) * rotary_scale / T.pow(rotary_theta, T.Cast("float32", vk * 2 % head_dim) / T.Cast("float32", head_dim)))) * Q[cur_batch_seqlen_start + vi, vj, vk] + T.Cast(dtype, T.sin(T.Cast("float32", offset_start + vi) * rotary_scale / T.pow(rotary_theta, T.Cast("float32", vk * 2 % head_dim) / T.Cast("float32", head_dim)))) * T.if_then_else(vk < head_dim // 2, Q[cur_batch_seqlen_start + vi, vj, vk + head_dim // 2] * T.float16(-1), Q[cur_batch_seqlen_start + vi, vj, vk - head_dim // 2])
                            else:
                                with T.block("k_rope"):
                                    vi = T.axis.spatial(cur_append_length, i_0 * 32 + i_1)
                                    vj = T.axis.spatial(num_kv_heads, b_h_fused % (num_qo_heads + num_kv_heads) + (0 - num_qo_heads))
                                    vk = T.axis.spatial(head_dim, k_0 * 4 + k_1)
                                    T.where(k_0 * 4 + k_1 < head_dim and i_0 * 32 + i_1 < cur_append_length)
                                    T.reads(K[cur_batch_seqlen_start + vi, vj, T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2):T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2) + (T.max(T.max(vk, head_dim // 2 + vk), vk - head_dim // 2) + 1 - T.min(T.min(vk, head_dim // 2 + vk), vk - head_dim // 2))])
                                    T.writes(K[cur_batch_seqlen_start + vi, vj, vk])
                                    K[cur_batch_seqlen_start + vi, vj, vk] = T.Cast(dtype, T.cos(T.Cast("float32", offset_start + vi) * rotary_scale / T.pow(rotary_theta, T.Cast("float32", vk * 2 % head_dim) / T.Cast("float32", head_dim)))) * K[cur_batch_seqlen_start + vi, vj, vk] + T.Cast(dtype, T.sin(T.Cast("float32", offset_start + vi) * rotary_scale / T.pow(rotary_theta, T.Cast("float32", vk * 2 % head_dim) / T.Cast("float32", head_dim)))) * T.if_then_else(vk < head_dim // 2, K[cur_batch_seqlen_start + vi, vj, vk + head_dim // 2] * T.float16(-1), K[cur_batch_seqlen_start + vi, vj, vk - head_dim // 2])
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-nested-blocks
    return tir_rotary


def _rope(  # pylint: disable=too-many-arguments
    buffer: T.Buffer,
    offset: tir.Var,
    rotary_dim: int,
    theta: tir.Var,
    scale: tir.Var,
    indices: Tuple[tir.Var, ...],
    qkv_dtype="float16",
):
    def rope_freq(
        s: tir.Var,
        d: tir.Var,
        d_range: Union[int, float, tir.PrimExpr],
        scale: float,
        theta: float,
        dtype: str,
    ):
        fp32_d_range = (
            tir.const(d_range, "float32")
            if isinstance(d_range, (int, float))
            else tir.Cast("float32", d_range)
        )
        freq = s * scale / tir.power(theta, d * 2 % d_range / fp32_d_range)
        cos_freq = tir.cos(freq).astype(dtype)
        sin_freq = tir.sin(freq).astype(dtype)
        return cos_freq, sin_freq

    d = indices[-1]
    cos_freq, sin_freq = rope_freq(offset, d, rotary_dim, scale, theta, qkv_dtype)
    cos = cos_freq * buffer[indices]
    sin = sin_freq * tir.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    )
    return cos + sin


def _attention_prefill(num_kv_heads, num_qo_heads, head_dim, qkv_dtype):
    assert (
        qkv_dtype == "float16"
    ), f"TIR attention kernel does not support dtype {qkv_dtype} right now"
    # pylint: disable=invalid-name
    qkv_dtype_bytes = 2
    vector_load_bytes = 8
    LOAD_VEC = vector_load_bytes // qkv_dtype_bytes
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    GROUP_SIZE = H_qo // H_kv
    log2e = math.log2(math.exp(1))

    num_warps = 4
    tile_x = 32
    tile_y = D
    tile_z = 16
    L_per_cta = tile_x // GROUP_SIZE

    NUM_BLKS = 16

    def mask(
        causal,
        row,
        col,
        kv_len,
        qo_len,
    ):
        return T.if_then_else(
            causal > 0,
            col < kv_len - qo_len + row + 1,
            col < kv_len,
        )

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_prefill_paged_kv(
        handler_id: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        QO_indptr_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        last_page_len_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        causal: T.int32,
        rotary_mode: T.int32,  # pylint: disable=unused-argument
        rope_scale: T.float32,  # pylint: disable=unused-argument
        rope_theta: T.float32,  # pylint: disable=unused-argument
    ):
        B = T.int32(is_size_var=True)
        L = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        page_size = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (L, H_qo, D), qkv_dtype)
        QO_indptr = T.match_buffer(QO_indptr_handle, (B + 1,), "int32")
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, page_size, D), qkv_dtype
        )
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32")
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32")
        last_page_len = T.match_buffer(last_page_len_handle, (B,), "int32")
        output = T.match_buffer(output_handle, (L, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (L, H_qo), "float32")  # pylint: disable=unused-variable

        sm_scale = 1.0 / math.sqrt(float(D)) * log2e

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(H_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()

                            tile_no = T.alloc_buffer((1,), "int32", scope="local")
                            batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                            batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                            batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                            iterator = T.alloc_buffer((1,), "int32", scope="local")
                            kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                            m_new = T.alloc_buffer((1,), "float32", scope="local")
                            m_prev = T.alloc_buffer((1,), "float32", scope="local")
                            d_new = T.alloc_buffer((1,), "float32", scope="local")

                            Q_smem = T.alloc_buffer((tile_x, D), qkv_dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, D), qkv_dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, D), qkv_dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, D), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x), "float32", scope="shared")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_no[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (QO_indptr[1] - QO_indptr[0]) * GROUP_SIZE
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < B):
                                # advance to next tile
                                while tile_no[0] >= batch_tiles[0] and batch_idx[0] < B:
                                    tile_no[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < B:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (QO_indptr[b_idx + 1] - QO_indptr[b_idx]) * GROUP_SIZE
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < B):
                                    b_idx: T.int32 = batch_idx[0]
                                    L_start: T.int32 = QO_indptr[b_idx] + tile_no[0] * L_per_cta
                                    H_qo_start: T.int32 = by * GROUP_SIZE

                                    cur_page_indptr_begin: T.int32 = page_table_indptr[b_idx]
                                    cur_page_indptr_end: T.int32 = page_table_indptr[b_idx + 1]
                                    cur_last_page_len: T.int32 = last_page_len[b_idx]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        (cur_page_indptr_end - cur_page_indptr_begin - 1) * page_size + cur_last_page_len,
                                        0
                                    )
                                    T.tvm_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                        row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_init"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = L_start + i // GROUP_SIZE
                                            cur_H_qo = H_qo_start + i % GROUP_SIZE
                                            if cur_L < QO_indptr[b_idx + 1]:
                                                Q_smem[i, j] = Q[cur_L, cur_H_qo, j]
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(cur_L, page_size)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, page_size)
                                                    K_smem[i, j] = pages[page_no, 0, by, page_offset, j]
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("V_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(cur_L, page_size)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, page_size)
                                                    V_smem[i, j] = pages[page_no, 1, by, page_offset, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        # Compute S
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with T.block("S_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += Q_smem[i, k] * K_smem[j, k] * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update1"):
                                                    m_prev[0] = m_smem[row]
                                                    m_new[0] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    for j in T.serial(tile_z):
                                                        if mask(causal,
                                                                row=tile_no[0] * L_per_cta + row // GROUP_SIZE,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=QO_indptr[b_idx + 1] - QO_indptr[b_idx]):
                                                            m_new[0] = T.max(m_new[0], S_smem[row, j])
                                                    d_new[0] = d_smem[row] * T.exp2(m_prev[0] - m_new[0])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        if mask(causal,
                                                                row=tile_no[0] * L_per_cta + row // GROUP_SIZE,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=QO_indptr[b_idx + 1] - QO_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[0])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[0])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[0] += S_smem[row, j]
                                                    m_smem[row] = m_new[0]
                                                    d_smem[row] = d_new[0]
                                                    m_prev_smem[row] = m_prev[0]
                                        T.tvm_storage_sync("shared")

                                        # Update O
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with T.block("O_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k] * V_smem[k, j]

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_store"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            if L_start + i // GROUP_SIZE < QO_indptr[b_idx + 1]:
                                                output[L_start + i // GROUP_SIZE, H_qo_start + i % GROUP_SIZE, j] = O_local[i, j] / d_smem[i]

                                    # move to next tile
                                    tile_no[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    sch = tir.Schedule(batch_prefill_paged_kv)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = (int)(math.ceil(math.sqrt(cnt)))
        while cnt % tile_y != 0 and y % tile_y != 0 and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, 32, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile, vec_len=4):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        if tile[1] % vec_len == 0:
            yi, vec = sch.split(yi, factors=[None, vec_len])
            sch.vectorize(vec)
        elif tile[1] in [2, 4]:
            sch.vectorize(yi)

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        sch.decompose_reduction(block, ty)

    tile_s = get_tile_size(tile_x, tile_z, 32 * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, 32 * num_warps)

    apply_to_gemm(sch, sch.get_block("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_block("O_gemm"), tile_o, 2, 3, k_major=False)

    apply_to_so_ewise(sch, sch.get_block("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_block("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_block("O_store"), tile_o)

    apply_to_qkv_load(sch, sch.get_block("Q_load"))
    apply_to_qkv_load(sch, sch.get_block("K_load"))
    apply_to_qkv_load(sch, sch.get_block("V_load"))

    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _attention_decode(num_kv_heads, num_qo_heads, head_dim, qkv_dtype):
    assert (
        qkv_dtype == "float16"
    ), f"TIR attention kernel does not support dtype {qkv_dtype} right now"
    # pylint: disable=invalid-name
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = max(8 // qkv_dtype_bytes, D // 32)
    bdx = D // VEC_SIZE
    bdy = GROUP_SIZE
    threads_per_CTA = max(128, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = 4 if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_decode_paged_kv(
        handler_id: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        last_page_len_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        page_size = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype)
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, page_size, D), qkv_dtype
        )
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32")
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32")
        last_page_len = T.match_buffer(last_page_len_handle, (B,), "int32")
        output = T.match_buffer(output_handle, (B, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (B, H_qo), "float32")  # pylint: disable=unused-variable

        sm_scale = 1.0 / math.sqrt(float(D)) * log2e

        for bx in T.thread_binding(B, thread="blockIdx.x"):
            for by in T.thread_binding(H_kv, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        for tz in T.thread_binding(bdz, thread="threadIdx.z"):
                            with T.block("attn"):
                                Q_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                K_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                V_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                S_allreduce = T.alloc_buffer((bdz, bdy, bdx), "float32", scope="shared")
                                O_allreduce = T.alloc_buffer((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = T.alloc_buffer((bdz, bdy, 2), "float32", scope="shared")

                                S_local = T.alloc_buffer((bdy * tile_size_per_bdx), "float32", scope="local")
                                K_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                V_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                offset = T.alloc_buffer((1,), "int32", scope="local")
                                m_prev = T.alloc_buffer((1,), "float32", scope="local")
                                d_prev = T.alloc_buffer((1,), "float32", scope="local")
                                other_m = T.alloc_buffer((1,), "float32", scope="local")
                                other_d = T.alloc_buffer((1,), "float32", scope="local")
                                other_o = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")
                                st_m = T.alloc_buffer((1,), "float32", scope="local")
                                st_d = T.alloc_buffer((1,), "float32", scope="local")
                                O_local = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")

                                batch_idx: T.int32 = bx
                                cur_page_indptr_begin: T.int32 = page_table_indptr[batch_idx]
                                cur_page_indptr_end: T.int32 = page_table_indptr[batch_idx + 1]
                                cur_last_page_len: T.int32 = last_page_len[batch_idx]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    (cur_page_indptr_end - cur_page_indptr_begin - 1) * page_size + cur_last_page_len,
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(rotary_mode == 1,
                                                                  _rope(Q, kv_chunk_len[0]-1, head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec)),
                                                                  Q[bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec])

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx
                                    # load K from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, page_size)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, page_size)
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(rotary_mode == 1,
                                                                                                               _rope(pages, row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec)),
                                                                                                               pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec])
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # load V from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, page_size)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, page_size)
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j >= kv_chunk_len[0]:
                                            S_local[j] = -5e4
                                        else:
                                            # load K from shared memory to local memory
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_local[vec] = K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                            # compute S = Q * K * sm_scale
                                            S_local[j] = 0
                                            for vec in T.serial(VEC_SIZE):
                                                S_local[j] += Q_local[vec] * K_local[vec] * sm_scale
                                            # allreduce over bdx
                                            S_allreduce[tz, ty, tx] = S_local[j]
                                            T.tvm_storage_sync("shared")
                                            offset[0] = bdx // 2
                                            while offset[0] > 0:
                                                if tx < offset[0]:
                                                    S_allreduce[tz, ty, tx] += S_allreduce[tz, ty, tx + offset[0]]
                                                T.tvm_storage_sync("shared")
                                                offset[0] = offset[0] >> 1
                                            S_local[j] = S_allreduce[tz, ty, 0]
                                        # update st_m
                                        st_m[0] = T.max(st_m[0], S_local[j])

                                    # update st_d, st_O
                                    o_scale: T.float32 = T.exp2(m_prev[0] - st_m[0])
                                    st_d[0] *= o_scale
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        S_local[j] = T.exp2(S_local[j] - st_m[0])
                                        st_d[0] += S_local[j]
                                    for j in T.vectorized(VEC_SIZE):
                                        O_local[j] *= o_scale

                                    # load V from shared memory to local memory
                                    # compute O
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        for vec in T.vectorized(VEC_SIZE):
                                            V_local[vec] = V_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] += V_local[vec] * S_local[j]

                                if bdz > 1:
                                    # allreduce over bdz
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_allreduce[tz, ty, tx * VEC_SIZE + vec] = O_local[vec]
                                    md_allreduce[tz, ty, 0] = st_m[0]
                                    md_allreduce[tz, ty, 1] = st_d[0]
                                    T.tvm_storage_sync("shared")

                                    st_m[0] = -5e4
                                    st_d[0] = 1.0
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_local[vec] = 0.0

                                    for j in T.serial(bdz):
                                        m_prev[0] = st_m[0]
                                        d_prev[0] = st_d[0]
                                        other_m[0] = md_allreduce[j, ty, 0]
                                        other_d[0] = md_allreduce[j, ty, 1]
                                        for vec in T.vectorized(VEC_SIZE):
                                            other_o[vec] = O_allreduce[j, ty, tx * VEC_SIZE + vec]
                                        st_m[0] = T.max(st_m[0], other_m[0])
                                        st_d[0] = d_prev[0] * T.exp2(m_prev[0] - st_m[0]) + other_d[0] * T.exp2(other_m[0] - st_m[0])
                                        for vec in T.serial(VEC_SIZE):
                                            O_local[vec] = O_local[vec] * T.exp2(m_prev[0] - st_m[0]) + other_o[vec] * T.exp2(other_m[0] - st_m[0])

                                # normalize O
                                for vec in T.serial(VEC_SIZE):
                                    O_local[vec] /= st_d[0]

                                # store O to global memory
                                for vec in T.vectorized(VEC_SIZE):
                                    output[batch_idx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec] = O_local[vec]
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    return batch_decode_paged_kv
