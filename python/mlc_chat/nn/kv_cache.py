"""Attention KV cache modeling."""
# pylint: disable=too-many-statements,too-many-lines
import math
from typing import Tuple

from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T

from mlc_chat.op.position_embedding import rope_freq


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
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, head_dim, dtype), "kv_cache_transpose_append"),
            rx.extern("paged_kv_cache.attention_kernel_prefill"),
            rx.extern("paged_kv_cache.attention_kernel_decode"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"),
            rx.extern("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"),
            rx.extern("paged_kv_cache.attention_kernel_prefill_begin_forward"),
            rx.extern("paged_kv_cache.attention_kernel_prefill_end_forward"),
            rx.extern("paged_kv_cache.attention_kernel_decode_begin_forward"),
            rx.extern("paged_kv_cache.attention_kernel_decode_end_forward"),
            rx.extern("flashinfer.merge_state_in_place"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype), "kv_cache_debug_get_kv"),
            # fmt: on
            # pylint: enable=line-too-long
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
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype), "tir_attention_prefill"),
            bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype), "tir_attention_decode"),
            bb.add_func(_attention_prefill_ragged(num_key_value_heads, num_attention_heads, head_dim, dtype), "tir_attention_prefill_ragged"),
            bb.add_func(_merge_state_inplace(num_key_value_heads, head_dim, dtype), "tir_attention_merge_state"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype), "kv_cache_debug_get_kv"),
            # fmt: on
            # pylint: enable=line-too-long
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
        num_pages = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, 16, head_dim), dtype)
        k_data = T.match_buffer(var_k_data, (ntoken, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_key_value_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32")
        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                position: T.int32 = position_map[vgpos]  # type: ignore
                pages[T.floordiv(position, 16), 0, vh, T.floormod(position, 16), vf] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                position: T.int32 = position_map[vgpos] # type: ignore[name-defined,no-redef]
                pages[T.floordiv(position, 16), 1, vh, T.floormod(position, 16), vf] = v_data[vgpos, vh, vf]
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


# pylint: disable=line-too-long,too-many-arguments,too-many-nested-blocks,invalid-name


def _inplace_rope(
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
):
    assert head_dim <= 128, "Rotary embedding currently only supports head_dim <= 128"
    rotary_dim = head_dim

    def _rope(
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        rope_offset: tir.Var,
        instance_offset: tir.Var,
    ):
        cos_freq, sin_freq = rope_freq((s + rope_offset) * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s + instance_offset, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s + instance_offset, h, d + rotary_dim // 2],
            x[s + instance_offset, h, d - rotary_dim // 2],
        )
        return cos + sin

    # fmt: off
    @T.prim_func
    def tir_rotary(
        var_q: T.handle,
        var_k: T.handle,
        var_append_len_indptr: T.handle,
        var_rope_offsets: T.handle,
        _0: T.int32,
        _1: T.int32,
        _2: T.int32,
        _3: T.int32,
        _4: T.int32,
        _5: T.float32,
        _6: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        total_len = T.int32()
        batch_size = T.int32()
        q = T.match_buffer(var_q, (total_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (total_len, num_kv_heads, head_dim), dtype)
        rope_offsets = T.match_buffer(var_rope_offsets, (batch_size,), "int32")
        append_len_indptr = T.match_buffer(var_append_len_indptr, (batch_size + 1,), "int32")
        with T.block():
            for b_h in T.thread_binding(batch_size * (num_q_heads + num_kv_heads), thread="blockIdx.x"):
                b: T.int32 = b_h // (num_q_heads + num_kv_heads)
                h: T.int32 = b_h % (num_q_heads + num_kv_heads)
                instance_offset: T.int32 = append_len_indptr[b]
                rope_offset: T.int32 = rope_offsets[b]
                append_len: T.int32 = append_len_indptr[b + 1] - append_len_indptr[b]
                for s0 in range(T.ceildiv(append_len, 32)):
                    for s1 in T.thread_binding(32, thread="threadIdx.y"):
                        for d0 in T.thread_binding(T.ceildiv(head_dim, 4), thread="threadIdx.x"):
                            for d1 in T.vectorized(4):
                                s: T.int32 = s0 * 32 + s1
                                d: T.int32 = d0 * 4 + d1
                                if s < append_len and d < head_dim:
                                    if h < num_q_heads:
                                        q[s + instance_offset, h, d] = _rope(q, s, h, d, rope_offset, instance_offset)
                                    else:
                                        k[s + instance_offset, h - num_q_heads, d] = _rope(k, s, h - num_q_heads, d, rope_offset, instance_offset)
    return tir_rotary


# pylint: enable=line-too-long,too-many-arguments,too-many-nested-blocks,invalid-name


def _rope(  # pylint: disable=too-many-arguments
    buffer: T.Buffer,
    offset: tir.Var,
    rotary_dim: int,
    theta: tir.Var,
    scale: tir.Var,
    indices: Tuple[tir.Var, ...],
    qkv_dtype="float16",
):
    d = indices[-1]
    cos_freq, sin_freq = rope_freq(offset * scale, d, rotary_dim, theta, qkv_dtype)
    cos = cos_freq * buffer[indices]
    sin = sin_freq * tir.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    )
    return cos + sin


def _var(dtype):
    return T.alloc_buffer((1,), dtype, scope="local")


def _attention_prefill(h_kv, h_q, d, dtype):
    assert dtype == "float16", f"TIR attention kernel does not support dtype {dtype} right now"
    # pylint: disable=invalid-name
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    num_warps = 4
    tile_x, tile_y, tile_z = 32, d, 16
    L_per_cta = tile_x // group_size

    def mask(causal, row, col, kv_len, qo_len):
        return T.if_then_else(
            causal > 0,
            col < kv_len - qo_len + row + 1,
            col < kv_len,
        )

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_prefill_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, 2, h_kv, page_size, d]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_last_page_len: T.handle, # [b]
        var_k_rope_pos_offset: T.handle, # [b]
        var_q_rope_position: T.handle, # [total_len]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
    ):
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (total_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32")
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, 16, d), dtype)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32")
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32")
        last_page_len = T.match_buffer(var_last_page_len, (batch_size,), "int32")
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32")
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32")
        output = T.match_buffer(var_output, (total_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id = _var("int32")
                            batch_idx = _var("int32")
                            batch_tiles = _var("int32")
                            batch_rows = _var("int32")
                            iterator = _var("int32")
                            kv_chunk_len = _var("int32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                            m_new = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    L_start: T.int32 = q_indptr[b_idx] + tile_id[0] * L_per_cta
                                    H_qo_start: T.int32 = by * group_size

                                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                    cur_last_page_len: T.int32 = last_page_len[b_idx]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + cur_last_page_len,
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
                                            cur_L = L_start + i // group_size
                                            cur_H_qo = H_qo_start + i % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j)),
                                                    q[cur_L, cur_H_qo, j]
                                                )
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, 16)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, 16)
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (page_no, 0, by, page_offset, j)),
                                                        pages[page_no, 0, by, page_offset, j]
                                                    )
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, 16)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, 16)
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
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    for j in T.serial(tile_z):
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
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
                                            if L_start + i // group_size < q_indptr[b_idx + 1]:
                                                output[L_start + i // group_size, H_qo_start + i % group_size, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with T.block("lse_store"):
                                            i = T.axis.remap("S", [li])
                                            if L_start + i // group_size < q_indptr[b_idx + 1]:
                                                lse[L_start + i // group_size, H_qo_start + i % group_size] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
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

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

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
    apply_to_md(sch, sch.get_block("lse_store"))
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
    assert bdx == 32
    bdy = GROUP_SIZE
    threads_per_CTA = max(128, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = 4 if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_decode_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        last_page_len_handle: T.handle,
        k_rope_pos_offset_handle: T.handle,
        q_rope_position_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype)
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, 16, D), qkv_dtype
        )
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32")
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32")
        k_rope_pos_offset = T.match_buffer(k_rope_pos_offset_handle, (B,), "int32")
        q_rope_position = T.match_buffer(q_rope_position_handle, (B,), "int32")
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
                                O_allreduce = T.alloc_buffer((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = T.alloc_buffer((bdz, bdy, 2), "float32", scope="shared")
                                S_reduce_local = T.alloc_buffer((1,), "float32", scope="local")
                                t0 = T.alloc_buffer((1,), "float32", scope="local")

                                S_local = T.alloc_buffer((bdy * tile_size_per_bdx), "float32", scope="local")
                                K_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                V_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
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
                                    (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + cur_last_page_len,
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(Q, q_rope_position[batch_idx], head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec)),
                                        Q[bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec]
                                    )

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx
                                    # load K from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, 16)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, 16)
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(pages, k_rope_pos_offset[batch_idx] + row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec)),
                                                    pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec]
                                                )
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # load V from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, 16)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, 16)
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
                                            S_reduce_local[0] = 0
                                            for vec in T.serial(VEC_SIZE):
                                                S_reduce_local[0] += Q_local[vec] * K_local[vec] * sm_scale

                                            with T.block("block_cross_thread"):
                                                T.reads(S_reduce_local[0])
                                                T.writes(t0[0])
                                                T.attr(
                                                    T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                                    "reduce_scope",
                                                    T.reinterpret("handle", T.uint64(0)),
                                                )
                                                T.tvm_thread_allreduce(T.uint32(1), S_reduce_local[0], True, t0[0], tx, dtype="handle")

                                            S_local[j] = t0[0]
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

                                # store lse to global memory
                                lse[batch_idx, by * GROUP_SIZE + ty] = st_m[0] + T.log2(st_d[0])
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    return batch_decode_paged_kv


def _merge_state_inplace(num_heads, head_dim, v_dtype):
    # pylint: disable=invalid-name
    v_dtype_bytes = 2
    VEC_SIZE = max(8 // v_dtype_bytes, head_dim // 32)
    bdx = head_dim // VEC_SIZE
    bdy = num_heads

    @T.prim_func
    def merge_state_inplace(
        v: T.handle,
        s: T.handle,
        v_other: T.handle,
        s_other: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        N = T.int32(is_size_var=True)
        H = T.int32(is_size_var=True)
        D = T.int32(is_size_var=True)

        V = T.match_buffer(v, (N, H, D), v_dtype)
        S = T.match_buffer(s, (N, H), "float32")
        V_other = T.match_buffer(v_other, (N, H, D), v_dtype)
        S_other = T.match_buffer(s_other, (N, H), "float32")

        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                    with T.block("merge"):
                        s_val = _var("float32")
                        s_other_val = _var("float32")
                        s_max = _var("float32")
                        scale = _var("float32")
                        other_scale = _var("float32")

                        v_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")
                        v_other_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")

                        s_val[0] = S[bx, ty]
                        s_other_val[0] = S_other[bx, ty]
                        s_max[0] = T.max(s_val[0], s_other_val[0])
                        s_val[0] = T.exp2(s_val[0] - s_max[0])
                        s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                        scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                        other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])

                        # load v
                        for vec in T.vectorized(VEC_SIZE):
                            v_vec[vec] = V[bx, ty, tx * VEC_SIZE + vec]
                        # load v_other
                        for vec in T.vectorized(VEC_SIZE):
                            v_other_vec[vec] = V_other[bx, ty, tx * VEC_SIZE + vec]

                        # merge
                        for vec in T.serial(VEC_SIZE):
                            v_vec[vec] = v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]

                        # store v
                        for vec in T.vectorized(VEC_SIZE):
                            V[bx, ty, tx * VEC_SIZE + vec] = v_vec[vec]

                        # store s
                        S[bx, ty] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    # pylint: enable=invalid-name
    return merge_state_inplace


def _attention_prefill_ragged(h_kv, h_q, d, dtype):
    assert dtype == "float16", f"TIR attention kernel does not support dtype {dtype} right now"
    # pylint: disable=invalid-name,line-too-long
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    num_warps = 4
    tile_x, tile_y, tile_z = 32, d, 16
    L_per_cta = tile_x // group_size

    def mask(causal, row, col, kv_len, qo_len):
        return T.if_then_else(
            causal > 0,
            col < kv_len - qo_len + row + 1,
            col < kv_len,
        )

    # fmt: off
    @T.prim_func
    def batch_prefill_ragged_kv(  # pylint: disable=too-many-arguments,too-many-branches
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_k: T.handle, # [total_len, h_kv, d]
        var_v: T.handle, # [total_len, h_kv, d]
        var_kv_indptr: T.handle, # [batch_size + 1]
        var_q_rope_position: T.handle, # [total_q_len]
        var_k_rope_pos_offset: T.handle, # [b]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32")
        k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32")
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32")
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32")
        output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id = _var("int32")
                            batch_idx = _var("int32")
                            batch_tiles = _var("int32")
                            batch_rows = _var("int32")
                            iterator = _var("int32")
                            kv_chunk_len = _var("int32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                            m_new = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (32 * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    L_start: T.int32 = q_indptr[b_idx] + tile_id[0] * L_per_cta
                                    H_qo_start: T.int32 = by * group_size

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
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
                                            cur_L = L_start + i // group_size
                                            cur_H_qo = H_qo_start + i % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j)),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        L_kv_base: T.int32 = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (L_kv_base + cur_L, by, j)),
                                                        k[L_kv_base + cur_L, by, j]
                                                    )
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
                                                    V_smem[i, j] = v[L_kv_base + cur_L, by, j]
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
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    for j in T.serial(tile_z):
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
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
                                            if L_start + i // group_size < q_indptr[b_idx + 1]:
                                                output[L_start + i // group_size, H_qo_start + i % group_size, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with T.block("lse_store"):
                                            i = T.axis.remap("S", [li])
                                            if L_start + i // group_size < q_indptr[b_idx + 1]:
                                                lse[L_start + i // group_size, H_qo_start + i % group_size] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    sch = tir.Schedule(batch_prefill_ragged_kv)

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

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

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

    apply_to_md(sch, sch.get_block("lse_store"))
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)
