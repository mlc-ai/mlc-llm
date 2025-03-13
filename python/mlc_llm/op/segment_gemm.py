from typing import Literal, Optional
import functools

from tvm import relax as rx
from tvm import DataType, DataTypeCode, tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T
from tvm.target import Target
import tvm


@functools.lru_cache(maxsize=None)
def segment_tir_func_impl(K, dtype):
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    UNROLL = 64
    STORAGE_ALIGN = False
    assert BLK_K % 8 == 0
    zero = tir.const(0, dtype)

    @T.prim_func(private=True)
    def _func(  # pylint: disable=too-many-statements
        var_x: T.handle,
        var_w: T.handle,
        var_seq_lens: T.handle,
        var_weight_indices: T.handle,
        var_o: T.handle,
    ):
        S = T.int32(is_size_var=True)
        max_batch_size = T.int32(is_size_var=True)
        batch_size = T.int32(is_size_var=True)
        seqlen_elem_offset = T.int32(is_size_var=True)
        weight_indices_elem_offset = T.int32(is_size_var=True)
        NL = T.int32(is_size_var=True)
        N = T.int32(is_size_var=True)

        X = T.match_buffer(var_x, (S, K), dtype)
        W = T.match_buffer(var_w, (NL, N, K), dtype)
        seq_lens = T.match_buffer(var_seq_lens, (max_batch_size,), "int32", elem_offset=seqlen_elem_offset)
        weight_indices = T.match_buffer(var_weight_indices, (batch_size,), "int32", elem_offset=weight_indices_elem_offset)
        O = T.match_buffer(var_o, (S, N), dtype)

        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.block("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(seq_lens[:], X[:, :], W[:, :, :], weight_indices[:])
                T.writes(O[:, :])
                # pylint: disable=redefined-builtin
                sum = T.alloc_buffer((2,), "int32", scope="local")
                row = T.alloc_buffer((2,), "int32", scope="local")
                cur_b = T.alloc_buffer((1,), "int32", scope="local")
                tile_id = T.alloc_buffer((1,), "int32", scope="local")
                tiles_per_row = T.ceildiv(N + BLK_N - 1, BLK_N)
                # pylint: enable=redefined-builtin
                sum[0] = 0
                sum[1] = T.ceildiv(seq_lens[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = seq_lens[0]
                cur_b[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_b[0] < batch_size):  # pylint: disable=no-member
                    # move to the current group
                    while sum[1] <= tile_id[0] and cur_b[0] < batch_size:
                        cur_b[0] += 1
                        if cur_b[0] < batch_size:
                            delta: T.int32 = seq_lens[cur_b[0]]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    # sync threads to make sure all threads have the same tile position
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_b[0] < batch_size):  # pylint: disable=no-member
                        # fetch current tile position
                        weight_index: T.int32 = weight_indices[cur_b[0]]
                        num_tiles: T.int32 = tile_id[0] - sum[0]
                        m_offset: T.int32 = BLK_M * T.floordiv(num_tiles, tiles_per_row) + row[0]
                        n_offset: T.int32 = BLK_N * T.floormod(num_tiles, tiles_per_row)
                        with T.block("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                W[T.if_then_else(weight_index < 0, zero, weight_index,), n_offset : n_offset + BLK_N, :],
                            )
                            T.writes(O[m_offset : m_offset + BLK_M, n_offset : n_offset + BLK_N])
                            X_tile = T.alloc_buffer((BLK_M, K), dtype, scope="shared")
                            W_tile = T.alloc_buffer((BLK_N, K), dtype, scope="shared")
                            O_tile = T.alloc_buffer((BLK_M, BLK_N), dtype, scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.block("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if weight_index >= 0:
                                        X_tile[i, j] = T.if_then_else(
                                            m_offset + i < row[1],
                                            X[m_offset + i, j],
                                            zero,
                                        )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.block("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if weight_index >= 0:
                                        W_tile[i, j] = T.if_then_else(
                                            n_offset + i < N,
                                            W[weight_index, n_offset + i, j],
                                            zero,
                                        )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.block("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    if weight_index >= 0:
                                        O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.block("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        O[m_offset + i, n_offset + j] = O_tile[i, j]
                    # move to next tile
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = tir.Schedule(_func)

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

        main_block = sch.get_block("compute")
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
        l2g = sch.get_block("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)
        _cooperative_fetch(sch.get_block("X_shared"), vec_len=VEC_X)
        _cooperative_fetch(sch.get_block("W_shared"), vec_len=VEC_W)
        sch.decompose_reduction(main_block, ko)
        return sch.mod["main"].with_attr("tir.is_scheduled", 1)
    
    return _schedule()


def segment_gemm(x: Tensor, w: Tensor, seq_lens: Tensor, weight_indices: Tensor):  # pylint: disable=too-many-statements
    """Segment GEMM in Lora models.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (s, in_features), here s is the sum of all sequence lengths.
    w : Tensor
        Weight tensor of shape (num_lora, out_features, in_features).
        "w[i, :, :]" is the weight matrix for the 'i'-th lora weight.
    seq_lens : Tensor
        Sequence length tensor of shape (batch_size, ),
        seq_lens[i]' is the input for 'i'-th batch.
    weight_indices : Tensor
        Sequence length tensor of shape (max_batch_size, ),
        "w[weight_indices[i], :, :]" is the weight for 'i'-th batch input.

    Returns
    -------
    out : Tensor
        Output tensor of shape (s, out_features), here s is the sum of all sequence lengths.
    """
    s, in_features = x.shape
    (num_lora, out_features, _), dtype = w.shape, w.dtype
    assert w.shape[-1] == in_features and x.dtype == dtype

    return op.tensor_ir_op(
        segment_tir_func_impl(in_features, dtype),
        "segment_gemm",
        args=[x, w, seq_lens, weight_indices],
        out=Tensor.placeholder((s, out_features), dtype),
    )