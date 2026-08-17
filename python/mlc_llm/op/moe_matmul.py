"""Mixture of Experts operators"""

from typing import Literal, Optional, Tuple  # noqa: UP035

from tvm import DataType, DataTypeCode, s_tir, tirx
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

# mypy: disable-error-code="attr-defined,valid-type,name-defined"


def gemv(x: Tensor, w: Tensor, indptr: Tensor) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The weight tensor of shape (local_experts, out_features, in_features), where `local_experts`
        is the total number of experts.

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    Returns
    -------
    out : Tensor
        The output tensor of shape (experts_per_tok, out_features), where `experts_per_tok` is the
        number of activated experts per token.
    """
    (local_experts, out_features, in_features), dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    x_leading_dim, _ = x.shape

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    # NOTE: Currently it assumes x.dtype == w.dtype, but the constraint can be relaxed easily.
    assert w.shape == [local_experts, out_features, in_features] and w.dtype == dtype
    assert x.shape == [x_leading_dim, in_features] and x.dtype == dtype
    assert indptr.shape == [1, experts_per_tok] and indptr.dtype == "int32"
    assert x_leading_dim in [1, experts_per_tok]

    @T.prim_func(private=True, s_tir=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), dtype),
        w: T.Buffer((local_experts, out_features, in_features), dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for e in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, e)
                T.reads(x[:, :], w[indptr[0, e], :, :], indptr[0, e])
                T.writes(o[e, :])
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), dtype)
                        o[e, i] += access_x(x, e, j) * w[indptr[0, e], i, j]

    return op.tensor_ir_op(
        _func,
        "moe_gemv",
        args=[x, w, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], dtype),
    )


def dequantize_gemv(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    quantize_dtype: str,
    group_size: int,
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized.
    It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features // n),
        where n is the number of elements per storage dtype, e.g. if the storage dtype is uint32,
        and the quantize dtype is int4, then n is 8.
        `local_experts` is the total number of experts including activated and non-active ones.

    scale : Tensor
        The scale tensor of shape (local_experts, out_features, in_features // group_size), where
        `local_experts` is the total number of experts including activated and non-active ones.

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    quantize_dtype : str
        The quantize dtype of the weight tensor, which is usually int3, int4 or fp8, etc.

    group_size : int
        The number of elements in each quantization group, e.g. 32 or 128.

    Returns
    -------
    out : Tensor
        The output tensor of shape (experts_per_tok, out_features), where `experts_per_tok` is the
        number of activated experts per token.
    """
    (x_leading_dim, in_features), model_dtype = x.shape, x.dtype
    (local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_group = (in_features + group_size - 1) // group_size
    num_storage = group_size // num_elem_per_storage * num_group

    def _dequantize(w, s, e, i, j):
        tir_bin_mask = tirx.const((2**quantize_dtype_bits) - 1, storage_dtype)
        tir_max_int = tirx.const((2 ** (quantize_dtype_bits - 1)) - 1, model_dtype)
        w = w[e, i, j // num_elem_per_storage]
        s = s[e, i, j // group_size]
        shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
        w = tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype(model_dtype)
        return (w - tir_max_int) * s

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    assert x.shape == [x_leading_dim, in_features] and x.dtype == model_dtype
    assert w.shape == [local_experts, out_features, num_storage] and w.dtype == storage_dtype
    assert scale.shape == [local_experts, out_features, num_group] and scale.dtype == model_dtype
    assert indptr.shape == [1, experts_per_tok] and indptr.dtype == "int32"
    assert x_leading_dim in [1, experts_per_tok]

    @T.prim_func(private=True, s_tir=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        scale: T.Buffer((local_experts, out_features, num_group), model_dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, scale, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    return op.tensor_ir_op(
        _func,
        "moe_dequantize_gemv",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
    )


def dequantize_float8_gemv(
    x: Tensor,
    w: Tensor,
    scale: Optional[Tensor],
    indptr: Tensor,
    quantize_dtype: Literal["float8_e5m2", "float8_e4m3fn"],
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized in
    fp8 e5m2 or e4m3. It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features)

    scale : Optional[Tensor]
        The optional scale tensor of shape (1,)

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    quantize_dtype : Literal["float8_e5m2", "float8_e4m3fn"]
        The quantize dtype of the weight tensor, which is either float8_e5m2 or float8_e4m3fn.
    """
    (x_leading_dim, in_features), model_dtype = x.shape, x.dtype
    (local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_storage = tirx.ceildiv(in_features, num_elem_per_storage)

    def _dequantize(w, s, e, i, j):
        if num_elem_per_storage == 1:
            w = tirx.reinterpret(quantize_dtype, w[e, i, j])
        else:
            assert DataType(storage_dtype).type_code == DataTypeCode.UINT
            tir_bin_mask = tirx.const((2**quantize_dtype_bits) - 1, storage_dtype)
            w = w[e, i, j // num_elem_per_storage]
            shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
            w = tirx.reinterpret(
                quantize_dtype,
                tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype("uint8"),
            )
        w = w.astype(model_dtype)
        if s is not None:
            w = w * s[0]
        return w

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    @T.prim_func(private=True, s_tir=True)
    def _func_with_scale(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        scale: T.Buffer((1,), "float32"),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, scale, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    @T.prim_func(private=True, s_tir=True)
    def _func_without_scale(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, None, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    if scale is not None:
        return op.tensor_ir_op(
            _func_with_scale,
            "moe_dequantize_gemv",
            args=[x, w, scale, indptr],
            out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
        )
    return op.tensor_ir_op(
        _func_without_scale,
        "moe_dequantize_gemv",
        args=[x, w, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
    )


def dequantize_block_scale_float8_gemv(
    x: Tensor,
    w: Tensor,
    w_scale: Tensor,
    expert_indices: Tensor,
    block_size: Tuple[int, int],  # noqa: UP006
    out_dtype: str,
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized in
    fp8 e5m2 or e4m3. It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features)

    w_scale : Tensor
        The scale tensor of shape
        (local_experts, out_features // block_size[0], in_features // block_size[1])

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    block_size : Tuple[int, int]
        The block size of the weight tensor.

    out_dtype : str
        The output dtype of the GEMV computation.
    """
    x_leading_dim, in_features = x.shape
    local_experts, out_features, k = w.shape
    _, experts_per_tok = expert_indices.shape
    model_dtype = x.dtype
    quantize_dtype = w.dtype

    assert out_features % block_size[0] == 0
    assert k % block_size[1] == 0

    def _dequantize(w, s, e, i, j):
        return w[e, i, j].astype(model_dtype) * s[e, i // block_size[0], j // block_size[1]].astype(
            model_dtype
        )

    def load_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    @T.prim_func(private=True, s_tir=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, k), quantize_dtype),
        w_scale: T.Buffer(
            (local_experts, out_features // block_size[0], k // block_size[1]),
            "float32",
        ),
        expert_indices: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), out_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, w_scale, expert_indices[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), out_dtype)
                        o[e, i] += (load_x(x, e, j) * y[i, j]).astype(out_dtype)

    return op.tensor_ir_op(
        _func,
        "moe_dequantize_gemv",
        args=[x, w, w_scale, expert_indices],
        out=Tensor.placeholder([experts_per_tok, out_features], out_dtype),
    )


def group_gemm(x: Tensor, w: Tensor, indptr: Tensor):
    """Group GEMM in MoE models.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (batch_size, in_features), where `batch_size` could be dynamic shape.

    w : Tensor
        Weight tensor of shape (num_local_experts, out_features, in_features).
        `w[i, :, :]` is the weight matrix for the `i`-th local expert.

    indptr : Tensor
        Index pointer tensor of shape (num_local_experts + 1, ).
        `x[indptr[a] : indptr[a + 1]]` is the input for the `i`-th local expert.

    Returns
    -------
    out : Tensor
        Output tensor of shape (batch_size, out_features).
    """
    # NOTE: Currently it assumes x.dtype == w.dtype, but the constraint can be relaxed easily.
    (num_local_experts, out_features, in_features), dtype = w.shape, w.dtype

    assert x.shape[1:] == [in_features] and x.dtype == dtype
    assert indptr.shape == [num_local_experts + 1] and indptr.dtype == "int32"

    Ne, N, K = num_local_experts, out_features, in_features
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    UNROLL = 64
    STORAGE_ALIGN = False
    assert BLK_K % 8 == 0
    tiles_per_row = (N + BLK_N - 1) // BLK_N
    zero = tirx.const(0, dtype)

    @T.prim_func(private=True, s_tir=True)
    def _func(
        var_x: T.handle,
        var_w: T.handle,
        var_indptr: T.handle,
        var_o: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        B = T.int32()
        X = T.match_buffer(var_x, (B, K), dtype)
        W = T.match_buffer(var_w, (Ne, N, K), dtype)
        indptr = T.match_buffer(var_indptr, (Ne + 1,), "int32")
        out = T.match_buffer(var_o, (B, N), dtype)

        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(indptr[:], X[:, :], W[:, :, :])
                T.writes(out[:, :])
                sum = T.sblock_alloc_buffer((2,), "int32", scope="local")
                row = T.sblock_alloc_buffer((2,), "int32", scope="local")
                cur_e = T.sblock_alloc_buffer((1,), "int32", scope="local")
                tile_id = T.sblock_alloc_buffer((1,), "int32", scope="local")
                sum[0] = 0
                sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = indptr[1] - indptr[0]
                cur_e[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_e[0] < Ne):
                    # move to the current group
                    while sum[1] <= tile_id[0] and cur_e[0] < Ne:
                        cur_e[0] += 1
                        if cur_e[0] < Ne:
                            e: T.int32 = cur_e[0]
                            delta: T.int32 = indptr[e + 1] - indptr[e]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    # sync threads to make sure all threads have the same tile position
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_e[0] < Ne):
                        # fetch current tile position
                        e: T.int32 = cur_e[0]
                        num_tiles: T.int32 = tile_id[0] - sum[0]
                        m_offset: T.int32 = BLK_M * T.floordiv(num_tiles, tiles_per_row) + row[0]
                        n_offset: T.int32 = BLK_N * T.floormod(num_tiles, tiles_per_row)
                        with T.sblock("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                W[e, n_offset : n_offset + BLK_N, :],
                            )
                            T.writes(
                                out[
                                    m_offset : m_offset + BLK_M,
                                    n_offset : n_offset + BLK_N,
                                ]
                            )
                            X_tile = T.sblock_alloc_buffer((BLK_M, K), dtype, scope="shared")
                            W_tile = T.sblock_alloc_buffer((BLK_N, K), dtype, scope="shared")
                            O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), dtype, scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.sblock("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(
                                        m_offset + i < row[1],
                                        X[m_offset + i, j],
                                        zero,
                                    )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.sblock("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    W_tile[i, j] = T.if_then_else(
                                        n_offset + i < N,
                                        W[e, n_offset + i, j],
                                        zero,
                                    )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.sblock("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.sblock("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        out[m_offset + i, n_offset + j] = O_tile[i, j]
                    # move to next tile
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = s_tir.Schedule(_func)

        def _cooperative_fetch(block, vec_len):
            num_loops = len(sch.get_loops(block))
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            ty, tx, _, vec = sch.split(
                sch.fuse(*loops),
                factors=[TY, TX, None, vec_len],
            )
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            if STORAGE_ALIGN:
                sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
            return block

        main_block = sch.get_sblock("compute")
        x, y, k = sch.get_loops(main_block)
        ty, yi = sch.split(y, [TY, None])
        tx, xi, vec_c = sch.split(x, [TX, None, VEC_DOT])
        ko, ki = sch.split(k, factors=[None, BLK_K])
        sch.reorder(ty, tx, ko, ki, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)
        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
        l2g = sch.get_sblock("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)
        _cooperative_fetch(sch.get_sblock("X_shared"), vec_len=VEC_X)
        _cooperative_fetch(sch.get_sblock("W_shared"), vec_len=VEC_W)
        sch.decompose_reduction(main_block, ko)
        return sch.mod["main"]

    return op.tensor_ir_op(
        _schedule(),
        "group_gemm",
        args=[x, w, indptr],
        out=Tensor.placeholder([x.shape[0], out_features], dtype),
    )


def dequantize_group_gemm(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    quantize_dtype: str,
    indptr_dtype: str,
    group_size: int,
):
    """Group GEMM in MoE models but the weight is quantized.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (batch_size, in_features), where `batch_size` could be dynamic shape.

    w : Tensor
        Weight tensor of shape (num_local_experts, out_features, in_features // n), where n is the
        number of elements per storage dtype, e.g. if the storage dtype is uint32, and the quantize
        dtype is int4, then n is 8.

    scale : Tensor
        The scale tensor of shape (num_local_experts, out_features, in_features // group_size).

    indptr : Tensor
        Index pointer tensor of shape (num_local_experts + 1, ). `x[indptr[a] : indptr[a + 1]]` is
        the input for the `i`-th local expert.

    group_size : int
        The number of elements in each quantization group, e.g. 32 or 128.

    quantize_dtype : str
        The quantize dtype of the weight tensor, which is usually int3, int4 or fp8, etc.

    indptr_dtype : str
        The dtype of the index pointer tensor, which can be int32 or int64.

    Returns
    -------
    out : Tensor
        Output tensor of shape (batch_size, out_features).
    """
    (_, in_features), model_dtype = x.shape, x.dtype
    (num_local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_group = (in_features + group_size - 1) // group_size
    num_storage = group_size // num_elem_per_storage * num_group

    def _dequantize(w, s, e, i, j):
        tir_bin_mask = tirx.const((1 << quantize_dtype_bits) - 1, storage_dtype)
        tir_max_int = tirx.const((2 ** (quantize_dtype_bits - 1)) - 1, model_dtype)
        w = w[e, i, j // num_elem_per_storage]
        s = s[e, i, j // group_size]
        shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
        w = tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype(model_dtype)
        return (w - tir_max_int) * s

    Ne, N, K = num_local_experts, out_features, in_features
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    UNROLL = 64
    STORAGE_ALIGN = False
    assert BLK_K % 8 == 0
    tiles_per_row = (N + BLK_N - 1) // BLK_N
    zero = tirx.const(0, model_dtype)
    if indptr_dtype == "int64":
        indptr = op.pad(indptr, [1, 0], "constant", 0)

    @T.prim_func(private=True, s_tir=True)
    def _func(
        var_x: T.handle,
        w: T.Buffer((Ne, N, num_storage), storage_dtype),
        scale: T.Buffer((Ne, N, num_group), model_dtype),
        indptr: T.Buffer((Ne + 1,), indptr_dtype),
        var_o: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        B = T.int32()
        X = T.match_buffer(var_x, (B, K), model_dtype)
        out = T.match_buffer(var_o, (B, N), model_dtype)
        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(X[:, :], w[:, :, :], scale[:, :, :], indptr[:])
                T.writes(out[:, :])
                sum = T.sblock_alloc_buffer((2,), indptr_dtype, scope="local")
                row = T.sblock_alloc_buffer((2,), indptr_dtype, scope="local")
                cur_e = T.sblock_alloc_buffer((1,), indptr_dtype, scope="local")
                tile_id = T.sblock_alloc_buffer((1,), indptr_dtype, scope="local")
                sum[0] = 0
                sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = indptr[1] - indptr[0]
                cur_e[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_e[0] < Ne):
                    # move to the current group
                    while sum[1] <= tile_id[0] and cur_e[0] < Ne:
                        cur_e[0] += 1
                        if cur_e[0] < Ne:
                            e = cur_e[0]
                            delta = indptr[e + 1] - indptr[e]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    # sync threads to make sure all threads have the same tile position
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_e[0] < Ne):
                        # fetch current tile position
                        e = cur_e[0]
                        num_tiles = tile_id[0] - sum[0]
                        m_offset = T.floordiv(num_tiles, tiles_per_row) * BLK_M + row[0]
                        n_offset = T.floormod(num_tiles, tiles_per_row) * BLK_N
                        with T.sblock("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                w[e, n_offset : n_offset + BLK_N, :],
                                scale[e, n_offset : n_offset + BLK_N, :],
                            )
                            T.writes(
                                out[
                                    m_offset : m_offset + BLK_M,
                                    n_offset : n_offset + BLK_N,
                                ]
                            )
                            X_tile = T.sblock_alloc_buffer((BLK_M, K), model_dtype, scope="shared")
                            W_tile = T.sblock_alloc_buffer((BLK_N, K), model_dtype, scope="shared")
                            O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), "float32", scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.sblock("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(
                                        m_offset + i < row[1],
                                        X[m_offset + i, j],
                                        zero,
                                    )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.sblock("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    W_tile[i, j] = T.if_then_else(
                                        n_offset + i < N,
                                        _dequantize(w, scale, e, n_offset + i, j),
                                        zero,
                                    )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.sblock("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.sblock("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        out[m_offset + i, n_offset + j] = O_tile[i, j]
                    # move to next tile
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = s_tir.Schedule(_func)

        def _cooperative_fetch(block, vec_len):
            num_loops = len(sch.get_loops(block))
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            ty, tx, _, vec = sch.split(
                sch.fuse(*loops),
                factors=[TY, TX, None, vec_len],
            )
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            if STORAGE_ALIGN:
                sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
            return block

        main_block = sch.get_sblock("compute")
        x, y, k = sch.get_loops(main_block)
        ty, yi = sch.split(y, [TY, None])
        tx, xi, vec_c = sch.split(x, [TX, None, VEC_DOT])
        ko, ki = sch.split(k, factors=[None, BLK_K])
        sch.reorder(ty, tx, ko, ki, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)
        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
        l2g = sch.get_sblock("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)
        _cooperative_fetch(sch.get_sblock("X_shared"), vec_len=VEC_X)
        _cooperative_fetch(sch.get_sblock("W_shared"), vec_len=VEC_W)
        sch.decompose_reduction(main_block, ko)
        return sch.mod["main"]

    return op.tensor_ir_op(
        _schedule(),
        "dequantize_group_gemm",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([x.shape[0], out_features], model_dtype),
    )




def dequantize_mxfp4_gemv(
    x: Tensor,
    w_blocks: Tensor,
    w_scales: Tensor,
    indptr: Tensor,
    group_size: int,
) -> Tensor:
    """GEMV for a single decode token against MXFP4-quantized MoE expert weights.
    Parameters
    ----------
    x : Tensor
        Shape ``(batch, x_leading_dim, in_features)``, dtype float16.
        ``x_leading_dim`` is 1 (broadcast) or ``experts_per_tok``.
    w_blocks : Tensor
        Shape ``(num_experts, in_features//8, out_features)``, dtype uint32.
        Each uint32 packs 8 Ã— FP4 nibbles along the K (in_features) dimension.
    w_scales : Tensor
        Shape ``(num_experts, in_features//group_size, out_features)``, dtype float16.
    indptr : Tensor
        Shape ``(batch, experts_per_tok)``, dtype int32.
        ``indptr[b, e]`` is the expert index for batch ``b``, slot ``e``.
    group_size : int
        Number of K elements sharing one scale factor (e.g. 32).
    Returns
    -------
    Tensor
        Shape ``(batch, experts_per_tok, out_features)``, dtype float16.
    """
    (bs, x_leading_dim, in_features), model_dtype = x.shape, x.dtype
    local_experts, kq, out_features = w_blocks.shape  # kq = in_features // 8
    _, experts_per_tok = indptr.shape
    storage_dtype = w_blocks.dtype  # uint32
    num_group = (in_features + group_size - 1) // group_size

    assert model_dtype == "float16"
    assert storage_dtype == "uint32"
    assert w_blocks.shape == [local_experts, in_features // 8, out_features]
    assert w_scales.shape == [local_experts, num_group, out_features]
    assert indptr.shape == [bs, experts_per_tok] and indptr.dtype == "int32"
    assert x_leading_dim in [1, experts_per_tok]

    def _dequantize_mxfp4(w_blocks, w_scales, e, n, k):
        """Branch-free MXFP4 dequantization via IEEE float16 bit reconstruction.
        All arithmetic is pure uint32 integer ops -- no LUT, no Select, no if.
        Steps:
          1. Extract 4-bit nibble from the packed uint32 weight block.
          2. Split into sign (bit 3) and magnitude (bits 2:0).
          3. Reconstruct float16 magnitude bits analytically:
               mantissa bit-9 = mag & 1
               biased exponent = (mag >> 1) + 14   (correct for mag in 1..7)
               zero-mask nz    = (mag + 7) >> 3    (0 iff mag==0, else 1)
               exponent_final  = biased_exp * nz   (zeroed for mag==0 -> +0.0)
          4. Assemble uint16 bits and reinterpret as float16.
          5. Multiply by the per-group scale.
        """
        # Step 1: extract nibble from packed uint32 (8 nibbles per word, 4 bits each)
        nibble = tir.bitwise_and(
            tir.shift_right(
                w_blocks[e, k // 8, n],
                tir.Cast("uint32", (k % 8) * 4),
            ),
            tir.const(0xF, "uint32"),
        )
        # Step 2: sign and magnitude
        sign = tir.shift_right(nibble, tir.const(3, "uint32"))  # 0 or 1
        mag = tir.bitwise_and(nibble, tir.const(0x7, "uint32"))  # 0..7
        # Step 3: float16 field reconstruction (branch-free)
        # mantissa: only bit-9 is ever set for MXFP4 magnitudes
        mant = tir.shift_left(
            tir.bitwise_and(mag, tir.const(1, "uint32")),
            tir.const(9, "uint32"),
        )
        # biased exponent: (mag >> 1) + 14  (correct for mag in 1..7)
        exp_r = tir.shift_right(mag, tir.const(1, "uint32")) + tir.const(14, "uint32")
        # zero-mask: (mag + 7) >> 3  ->  0 when mag==0, 1 when mag in [1..7]
        nz = tir.shift_right(
            mag + tir.const(7, "uint32"),
            tir.const(3, "uint32"),
        )
        # Step 4: assemble uint16 bits and reinterpret as float16
        fp16_bits = tir.bitwise_or(
            tir.bitwise_or(
                tir.shift_left(exp_r * nz, tir.const(10, "uint32")),
                mant,
            ),
            tir.shift_left(sign, tir.const(15, "uint32")),
        )
        fp16_val = tir.reinterpret("float16", tir.Cast("uint16", fp16_bits))
        # Step 5: apply per-group scale
        return fp16_val * w_scales[e, k // group_size, n]

    def access_x(x, b, e, k):
        return x[b, 0, k] if x_leading_dim == 1 else x[b, e, k]

    @T.prim_func(private=True)
    def _func(
        var_x: T.handle,
        w_blocks: T.Buffer((local_experts, in_features // 8, out_features), "uint32"),
        w_scales: T.Buffer((local_experts, num_group, out_features), model_dtype),
        var_indptr: T.handle,
        var_o: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})  # kOpaque
        batch_size = T.int32(is_size_var=True)
        x = T.match_buffer(var_x, (batch_size, x_leading_dim, in_features), model_dtype)
        indptr = T.match_buffer(var_indptr, (batch_size, experts_per_tok), "int32")
        o = T.match_buffer(var_o, (batch_size, experts_per_tok, out_features), model_dtype)
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            for _bs in range(batch_size):
                with T.sblock("gemv_o"):
                    e = T.axis.spatial(experts_per_tok, expert_id)
                    b = T.axis.spatial(batch_size, _bs)
                    y = T.alloc_buffer((out_features, in_features), model_dtype)
                    for i1, i2 in T.grid(out_features, in_features):
                        with T.sblock("dequantize"):
                            n, k = T.axis.remap("SS", [i1, i2])
                            y[n, k] = _dequantize_mxfp4(w_blocks, w_scales, indptr[b, e], n, k)
                    for i1, i2 in T.grid(out_features, in_features):
                        with T.sblock("gemv"):
                            n, k = T.axis.remap("SR", [i1, i2])
                            with T.init():
                                o[b, e, n] = T.cast(T.float16(0), model_dtype)
                            o[b, e, n] += access_x(x, b, e, k) * y[n, k]

    return op.tensor_ir_op(
        _func,
        "moe_dequantize_mxfp4_gemv",
        args=[x, w_blocks, w_scales, indptr],
        out=Tensor.placeholder([bs, experts_per_tok, out_features], model_dtype),
    )


def dequantize_mxfp4_group_gemm(
    x: Tensor,
    w_blocks: Tensor,
    w_scales: Tensor,
    bias: Tensor,
    indptr: Tensor,
    indptr_dtype: str,
    group_size: int,
) -> Tensor:
    """Group GEMM for prefill tokens against MXFP4-quantized MoE expert weights.
    The weight layout is **KN** (``w_blocks[e, k_packed, n]``) which matches
    the GptOss checkpoint layout.
    Parameters
    ----------
    x : Tensor
        Shape ``(batch_size, in_features)``, dtype float16.
        ``batch_size`` may be a dynamic shape variable.
    w_blocks : Tensor
        Shape ``(num_experts, in_features//8, out_features)``, dtype uint32.
    w_scales : Tensor
        Shape ``(num_experts, in_features//group_size, out_features)``, dtype float16.
    bias : Tensor
        Shape ``(num_experts, out_features)``, dtype float16.
        Per-expert bias fused into the store epilogue of each GEMM tile.
    indptr : Tensor
        Shape ``(num_experts + 1,)`` (exclusive) or ``(num_experts,)`` (inclusive).
        dtype is ``indptr_dtype``.
    indptr_dtype : str
        ``"int32"`` (exclusive indptr, length Ne+1) or
        ``"int64"`` (inclusive indptr, length Ne, CUTLASS path).
    group_size : int
        Number of K elements sharing one scale factor (e.g. 32).
    Returns
    -------
    Tensor
        Shape ``(batch_size, out_features)``, dtype float16.
    """
    (_, in_features), model_dtype = x.shape, x.dtype
    local_experts, kq, out_features = w_blocks.shape  # kq = in_features // 8
    storage_dtype = w_blocks.dtype  # uint32
    num_group = (in_features + group_size - 1) // group_size

    assert model_dtype == "float16"
    assert storage_dtype == "uint32"
    assert w_blocks.shape == [local_experts, in_features // 8, out_features]
    assert w_scales.shape == [local_experts, num_group, out_features]
    assert bias.shape == [local_experts, out_features] and bias.dtype == "float16"

    Ne, N, K = local_experts, out_features, in_features
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    assert BLK_K % 8 == 0
    UNROLL = 64
    STORAGE_ALIGN = False
    tiles_per_row = (N + BLK_N - 1) // BLK_N
    zero = tir.const(0, model_dtype)
    if indptr_dtype == "int64":
        indptr = op.pad(indptr, [1, 0], "constant", 0)

    def _dequantize_mxfp4(w_blocks, w_scales, e, n, k):
        """Branch-free MXFP4 dequantization via IEEE float16 bit reconstruction.
        All arithmetic is pure uint32 integer ops -- no LUT, no Select, no if.
          1. Extract 4-bit nibble from the packed uint32 weight block.
          2. Split into sign (bit 3) and magnitude (bits 2:0).
          3. Reconstruct float16 magnitude bits analytically:
               mantissa bit-9 = mag & 1
               biased exponent = (mag >> 1) + 14   (correct for mag in 1..7)
               zero-mask nz    = (mag + 7) >> 3    (0 iff mag==0, else 1)
               exponent_final  = biased_exp * nz   (zeroed for mag==0 -> +0.0)
          4. Assemble uint16 bits and reinterpret as float16.
          5. Multiply by the per-group scale.
        """
        nibble = tir.bitwise_and(
            tir.shift_right(
                w_blocks[e, k // 8, n],
                tir.Cast("uint32", (k % 8) * 4),
            ),
            tir.const(0xF, "uint32"),
        )
        sign = tir.shift_right(nibble, tir.const(3, "uint32"))
        mag = tir.bitwise_and(nibble, tir.const(0x7, "uint32"))
        mant = tir.shift_left(
            tir.bitwise_and(mag, tir.const(1, "uint32")),
            tir.const(9, "uint32"),
        )
        exp_r = tir.shift_right(mag, tir.const(1, "uint32")) + tir.const(14, "uint32")
        nz = tir.shift_right(
            mag + tir.const(7, "uint32"),
            tir.const(3, "uint32"),
        )
        fp16_bits = tir.bitwise_or(
            tir.bitwise_or(
                tir.shift_left(exp_r * nz, tir.const(10, "uint32")),
                mant,
            ),
            tir.shift_left(sign, tir.const(15, "uint32")),
        )
        fp16_val = tir.reinterpret("float16", tir.Cast("uint16", fp16_bits))
        return fp16_val * w_scales[e, k // group_size, n]

    @T.prim_func(private=True)
    def _func(
        var_x: T.handle,
        w_blocks: T.Buffer((Ne, K // 8, N), "uint32"),
        w_scales: T.Buffer((Ne, num_group, N), model_dtype),
        bias: T.Buffer((Ne, N), model_dtype),
        indptr: T.Buffer((Ne + 1,), indptr_dtype),
        var_o: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.is_scheduled": 1, "tir.noalias": True})
        B = T.int32(is_size_var=True)
        X = T.match_buffer(var_x, (B, K), model_dtype)
        O = T.match_buffer(var_o, (B, N), model_dtype)
        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(X[:, :], w_blocks[:, :, :], w_scales[:, :, :], bias[:, :], indptr[:])
                T.writes(O[:, :])
                # pylint: disable=redefined-builtin
                sum = T.alloc_buffer((2,), indptr_dtype, scope="local")
                row = T.alloc_buffer((2,), indptr_dtype, scope="local")
                cur_e = T.alloc_buffer((1,), indptr_dtype, scope="local")
                tile_id = T.alloc_buffer((1,), indptr_dtype, scope="local")
                # pylint: enable=redefined-builtin
                sum[0] = 0
                sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = indptr[1] - indptr[0]
                cur_e[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_e[0] < Ne):  # pylint: disable=no-member
                    while sum[1] <= tile_id[0] and cur_e[0] < Ne:
                        cur_e[0] += 1
                        if cur_e[0] < Ne:
                            e = cur_e[0]
                            delta = indptr[e + 1] - indptr[e]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_e[0] < Ne):  # pylint: disable=no-member
                        e = cur_e[0]  # type: ignore[no-redef]
                        num_tiles = tile_id[0] - sum[0]
                        m_offset = T.floordiv(num_tiles, tiles_per_row) * BLK_M + row[0]
                        n_offset = T.floormod(num_tiles, tiles_per_row) * BLK_N
                        with T.sblock("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                w_blocks[e, :, n_offset : n_offset + BLK_N],
                                w_scales[e, :, n_offset : n_offset + BLK_N],
                            )
                            T.writes(O[m_offset : m_offset + BLK_M, n_offset : n_offset + BLK_N])
                            X_tile = T.alloc_buffer((BLK_M, K), model_dtype, scope="shared")
                            W_tile = T.alloc_buffer((BLK_N, K), model_dtype, scope="shared")
                            O_tile = T.alloc_buffer((BLK_M, BLK_N), "float16", scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.sblock("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(
                                        m_offset + i < row[1], X[m_offset + i, j], zero
                                    )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.sblock("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    W_tile[i, j] = T.if_then_else(
                                        n_offset + i < N,
                                        _dequantize_mxfp4(w_blocks, w_scales, e, n_offset + i, j),
                                        zero,
                                    )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.sblock("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.sblock("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        # Fused bias add: accumulator + per-expert bias
                                        O[m_offset + i, n_offset + j] = (
                                            O_tile[i, j] + bias[e, n_offset + j]
                                        )
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = s_tir.Schedule(_func)

        main_block = sch.get_sblock("compute")
        x, y, k = sch.get_loops(main_block)
        yi, ty, vec_c = sch.split(y, [None, TY, VEC_O])
        tx, xi = sch.split(x, [TX, None])
        k0, k1, k2, k3 = sch.split(k, factors=[None, VEC_X, 4, 8])
        sch.reorder(ty, tx, k0, k1, k2, k3, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)
        sch.unroll(xi)

        inp_blk = sch.get_sblock("X_shared")
        sch.compute_at(inp_blk, k0)
        x, y = sch.get_loops(inp_blk)[-2:]
        tx2, xi2 = sch.split(x, [TX, None])
        yi2, ty2, vec_c2 = sch.split(y, [None, TY, VEC_X])
        sch.reorder(ty2, tx2, yi2, xi2, vec_c2)
        sch.bind(ty2, "threadIdx.y")
        sch.bind(tx2, "threadIdx.x")
        sch.vectorize(vec_c2)

        dequant_block = sch.get_sblock("W_shared")
        sch.compute_at(dequant_block, k3)
        sch.set_scope(dequant_block, 0, "local")
        yy = sch.get_loops(dequant_block)[-1]
        _, tyy, y_vec = sch.split(yy, [None, TY, VEC_O])
        sch.bind(tyy, "threadIdx.y")
        sch.vectorize(y_vec)
        sch.unroll(k3)

        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.get_sblock("store")
        x2, y2 = sch.get_loops(l2g)
        yi3, ty3, vec_c3 = sch.split(y2, [None, TY, VEC_O])
        tx3, xi3 = sch.split(x2, [TX, None])
        sch.reorder(ty3, tx3, yi3, xi3, vec_c3)
        sch.bind(ty3, "threadIdx.y")
        sch.bind(tx3, "threadIdx.x")
        sch.vectorize(vec_c3)

        sch.decompose_reduction(main_block, k0)
        return sch.mod["main"]

    return op.tensor_ir_op(
        _schedule(),
        "dequantize_mxfp4_group_gemm",
        args=[x, w_blocks, w_scales, bias, indptr],
        out=Tensor.placeholder([x.shape[0], out_features], model_dtype),
    )
