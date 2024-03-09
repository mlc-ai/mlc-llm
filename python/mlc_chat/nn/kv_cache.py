"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines
import enum
import math
from typing import Optional, Tuple

from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.target import Target

from mlc_chat.op.position_embedding import (
    llama_inplace_rope,
    llama_rope_with_position_map,
    rope_freq,
)

from ..support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


class PagedKVCache(Object):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(  # pylint: disable=too-many-arguments
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        rotary_dim: Optional[int] = None,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = head_dim
        return PagedKVCache(
            _expr=rx.Call(
                rx.extern("mlc.create_paged_kv_cache_generic"),
                args=[
                    rx.ShapeExpr(
                        [max_batch_size, max_total_seq_len, prefill_chunk_size, page_size]
                    ),
                    rx.PrimValue(num_hidden_layers),
                    rx.PrimValue(num_attention_heads),
                    rx.PrimValue(num_key_value_heads),
                    rx.PrimValue(head_dim),
                    rx.PrimValue(rope_mode),
                    rx.PrimValue(rope_scale),
                    rx.PrimValue(rope_theta),
                    rx.PrimValue(rotary_dim),
                    rx.DataTypeImm(dtype),
                ],
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )

    def attention(  # pylint: disable=invalid-name, too-many-arguments
        self,
        layer_id: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_score_scaling_factor: float = 1.0,
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
        q_shape = q.shape
        q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
        k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
        v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.paged_attention_kv_cache_attention",
                    [
                        self._expr,
                        rx.PrimValue(layer_id),  # type: ignore[arg-type]
                        rx.PrimValue(attn_score_scaling_factor),
                        q._expr,
                        k._expr,
                        v._expr,
                    ],
                    out_sinfo=q._expr.struct_info,
                )
            )
        ).reshape(*q_shape)
        # pylint: enable=protected-access

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
                    "vm.builtin.paged_attention_kv_cache_attention_with_fused_qkv",
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
                    "vm.builtin.paged_attention_kv_cache_get_query_positions",
                    self._expr,
                    sinfo_args=rx.TensorStructInfo((total_length,), "int32"),
                )
            )
        )

    # pylint: enable=protected-access


class FlashInferPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
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
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        """
        if rope_mode == RopeMode.INLINE:
            assert rotary_dim == head_dim, "FlashInfer RoPE does not support partial rotary dim."

        bb = rx.BlockBuilder.current()  # pylint: disable=invalid-name
        args = [
            rx.ShapeExpr([max_batch_size, max_total_seq_len, prefill_chunk_size, page_size]),
            rx.PrimValue(num_hidden_layers),
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
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, rotary_dim), "tir_split_rotary"),
            bb.add_func(llama_inplace_rope(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, target, rotary_dim), "tir_qk_rotary_inplace"),
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

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_mode: RopeMode,
        head_dim: int,
        rope_scale: int,
        rope_theta: int,
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
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        target : Target
            The target to build the model to.
        """

        bb = rx.BlockBuilder.current()
        args = [
            rx.ShapeExpr([max_batch_size, max_total_seq_len, prefill_chunk_size, page_size]),
            rx.PrimValue(num_hidden_layers),
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
            bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, head_dim, dtype, target), "tir_attention_prefill"),
            bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, head_dim, dtype, target), "tir_attention_decode"),
            bb.add_func(_attention_prefill_ragged(num_key_value_heads, num_attention_heads, head_dim, dtype, target), "tir_attention_prefill_ragged"),
            bb.add_func(_merge_state_inplace(num_key_value_heads, head_dim, dtype, target), "tir_attention_merge_state"),
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, rotary_dim), "tir_split_rotary"),
            bb.add_func(llama_inplace_rope(rope_theta, rope_scale, head_dim, num_attention_heads, num_key_value_heads, dtype, target, rotary_dim), "tir_qk_rotary_inplace"),
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
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
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
        seqlen = T.SizeVar("num_tokens_including_cache", "int64")
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


def _attention_prefill(h_kv, h_q, d, dtype, target: Target):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16
    L_per_cta = tile_x // group_size

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2
    check_thread_limits(target, bdx=bdx, bdy=num_warps, bdz=1, gdz=1)

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
        attn_score_scaling_factor: T.float32,
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
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
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

                            m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

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
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype),
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, 16)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, 16)  # type: ignore
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (page_no, 0, by, page_offset, j), dtype),
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, 16)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, 16)  # type: ignore
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
                                                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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
                                                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

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
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
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
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
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


def _attention_decode(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    target: Target,  # pylint: disable=unused-argument
):
    # pylint: disable=invalid-name
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    THREAD_LIMIT = 512
    TILE_SIZE_PER_BDX = 2
    if target.kind.name == "opencl" and "android" in str(target.host):
        THREAD_LIMIT = 64
        TILE_SIZE_PER_BDX = 1
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    thread_limit = min(max_num_threads_per_block, THREAD_LIMIT)

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = min(max(8 // qkv_dtype_bytes, D // 32), 4)
    bdx = D // VEC_SIZE
    bdy = GROUP_SIZE
    while bdx * bdy > thread_limit and bdy > 1:
        bdy //= 2
    gdz = GROUP_SIZE // bdy
    threads_per_CTA = max(thread_limit, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = TILE_SIZE_PER_BDX if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=bdz, gdz=1)

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
        attn_score_scaling_factor: T.float32,
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
            for fused_by_bz in T.thread_binding(H_kv * gdz, thread="blockIdx.y"):
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

                                by: T.int32 = fused_by_bz % H_kv
                                bz: T.int32 = fused_by_bz // H_kv
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
                                        _rope(Q, q_rope_position[batch_idx], head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec), qkv_dtype),
                                        Q[bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec]
                                    )

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    # load K from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j  # type: ignore
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, 16)]  # type: ignore
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, 16)  # type: ignore
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(pages, k_rope_pos_offset[batch_idx] + row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec), qkv_dtype),
                                                    pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec]
                                                )
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # load V from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j  # type: ignore
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, 16)]  # type: ignore
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, 16)  # type: ignore
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        # load K from shared memory to local memory
                                        for vec in T.vectorized(VEC_SIZE):
                                            K_local[vec] = K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        # compute S = Q * K * sm_scale
                                        S_reduce_local[0] = 0
                                        for vec in T.serial(VEC_SIZE):
                                            S_reduce_local[0] += T.cast(Q_local[vec], "float32") * T.cast(K_local[vec], "float32") * attn_score_scaling_factor * sm_scale

                                        with T.block("block_cross_thread"):
                                            T.reads(S_reduce_local[0])
                                            T.writes(t0[0])
                                            T.attr(
                                                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                                "reduce_scope",
                                                T.reinterpret("handle", T.uint64(0)),
                                            )
                                            T.tvm_thread_allreduce(T.uint32(1), S_reduce_local[0], True, t0[0], tx, dtype="handle")

                                        S_local[j] = -5e4
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j < kv_chunk_len[0]:
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
                                            O_local[vec] += T.cast(V_local[vec], "float32") * S_local[j]

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
                                    output[batch_idx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec] = O_local[vec]

                                # store lse to global memory
                                lse[batch_idx, by * GROUP_SIZE + bz * bdy + ty] = st_m[0] + T.log2(st_d[0])
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    return batch_decode_paged_kv


def _merge_state_inplace(
    num_heads, head_dim, v_dtype, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    v_dtype_bytes = 2
    VEC_SIZE = min(max(8 // v_dtype_bytes, head_dim // 32), 4)
    bdx = head_dim // VEC_SIZE
    bdy = num_heads
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    while bdx * bdy > max_num_threads_per_block and bdy > 1:
        bdy //= 2
    gdy = num_heads // bdy
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=1, gdz=1)

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
            for by in T.thread_binding(gdy, thread="blockIdx.y"):
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

                            s_val[0] = S[bx, ty + by * bdy]
                            s_other_val[0] = S_other[bx, ty + by * bdy]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])

                            # load v
                            for vec in T.vectorized(VEC_SIZE):
                                v_vec[vec] = V[bx, ty + by * bdy, tx * VEC_SIZE + vec]
                            # load v_other
                            for vec in T.vectorized(VEC_SIZE):
                                v_other_vec[vec] = V_other[bx, ty + by * bdy, tx * VEC_SIZE + vec]

                            # merge
                            for vec in T.serial(VEC_SIZE):
                                v_vec[vec] = (
                                    v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]
                                )

                            # store v
                            for vec in T.vectorized(VEC_SIZE):
                                V[bx, ty + by * bdy, tx * VEC_SIZE + vec] = v_vec[vec]

                            # store s
                            S[bx, ty + by * bdy] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    # pylint: enable=invalid-name
    return merge_state_inplace


def _attention_prefill_ragged(
    h_kv, h_q, d, dtype, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name,line-too-long
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16
    L_per_cta = tile_x // group_size

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2

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
        attn_score_scaling_factor: T.float32
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
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
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

                            m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

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
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype),
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
                                                        _rope(k, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (L_kv_base + cur_L, by, j), dtype),
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
                                                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
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
                                                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

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
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
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
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
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
