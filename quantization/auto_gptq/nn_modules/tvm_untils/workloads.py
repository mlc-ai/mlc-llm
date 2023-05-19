import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os


        
def _apply_gemv_schedule(ir_module, bits, K, num_warps=4):
    num_warps = num_warps
    warp_size = 32
    vec = 8
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)
    i, j, k = sch.get_loops(block_b)
    block_shared_local_A = sch.cache_read(block_b, 0, "local")
    block_local_C = sch.cache_write(block_b, 0, "local")
    bx, j = sch.split(
        j, factors=[None, num_warps])
    k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
    sch.reorder(bx, j, i, k, tx)

    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.bind(j, "threadIdx.y")

    sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
    block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
    sch.vectorize(block_local_a_v)
    ctx = tvm.cuda(0)
    cuda_mod = tvm.build(sch.mod, target="cuda")
    return cuda_mod

def _apply_gemm_schedule(ir_module, bits, K, config):
    from tvm.tir.tensor_intrin.cuda import (
        WMMA_FILL_16x16x16_F16_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
        WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
        WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
    )

    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    warp_size = 32
    BM = config['BM']
    BN = config['BN']
    BK = config['BK']
    block_row_warps = config['block_row_warps']
    block_col_warps = config['block_col_warps']
    raster = config['raster']
    stage = config['stage']
    warp_row_tiles = BM // (wmma_m * block_row_warps)
    warp_col_tiles = BN // (wmma_n * block_col_warps)
    chunk = BK // (wmma_k)
    vec = 8
    shared_pad = 8
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    block_shared_A = sch.cache_read(block_b, 0, "shared")
    block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
    block_shared_B = sch.cache_read(block_b, 1, "shared")
    block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
    block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)

    (i, j, k) = sch.get_loops(block_b)
    i, kernel_i = sch.split(i, factors=[None, wmma_m])
    j, kernel_j = sch.split(j, factors=[None, wmma_n])
    k, kernel_k = sch.split(k, factors=[None, wmma_k])
    block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
    block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
    if raster > 0:
        block_j, block_k = sch.split(block_j, factors=[None, raster])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_k, block_i, block_j, i, j, ko, ki,
                    ii, jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_k, "blockIdx.z")
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
    else:
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii,
                    jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")


    # cache read A from global memory to shared_memory
    sch.compute_at(block_shared_local_A, ki)
    sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
    sch.compute_at(block_shared_local_B, ki)
    sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j)


    A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
    A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
        A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
    sch.vectorize(A_shared_vi)
    sch.bind(A_shared_tx, "threadIdx.x")
    sch.bind(A_shared_ty, "threadIdx.y")
    sch.bind(A_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)

    B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
    B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
        B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, 1])
    sch.vectorize(B_shared_vi)
    sch.bind(B_shared_tx, "threadIdx.x")
    sch.bind(B_shared_ty, "threadIdx.y")
    sch.bind(B_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)



    A_local_i, A_local_j = sch.get_loops(block_shared_local_A)[-2:]
    A_local_i, A_local_kernel_i = sch.split(A_local_i, factors=[None, wmma_m])
    A_local_j, A_local_kernel_j = sch.split(A_local_j, factors=[None, wmma_k])
    sch.reorder(A_local_i, A_local_j, A_local_kernel_i, A_local_kernel_j)

    B_local_i, B_local_j = sch.get_loops(block_shared_local_B)[-2:]
    B_local_i, B_local_kernel_i = sch.split(B_local_i, factors=[None, wmma_n])
    B_local_j, B_local_kernel_j = sch.split(B_local_j, factors=[None, wmma_k])
    sch.reorder(B_local_i, B_local_j, B_local_kernel_i, B_local_kernel_j)

    C_local_i, C_local_j = sch.get_loops(block_local_C)[-2:]
    C_local_i, C_local_kernel_i = sch.split(C_local_i, factors=[None, wmma_m])
    C_local_j, C_local_kernel_j = sch.split(C_local_j, factors=[None, wmma_n])
    sch.reorder(C_local_i, C_local_j, C_local_kernel_i, C_local_kernel_j)

    # decompose reduction
    init_block_b = sch.decompose_reduction(block_b, ko)

    # transpose layout

    init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
    sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)

    block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(
        block_shared_local_A)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_A)
                [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)

    block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(
        block_shared_local_B)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_B)
                [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
    sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

    sch.tensorize(sch.get_loops(block_local_C)
                [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)

    if stage > 1:

        sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
        sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        sch.annotate(ko, ann_key="software_pipeline_stage",
                    ann_val=[0, 0, 0, stage - 1, 0])
        sch.annotate(ko, ann_key="software_pipeline_order",
                    ann_val=[0, 1, 3, 2, 4])
        sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])


    ctx = tvm.cuda(0)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        cuda_mod = tvm.build(sch.mod, target="cuda")
    
    return cuda_mod


def get_gemm_workloads(bits, N, K, transposed=True):
    M = 512
    N = N
    K = K
    group_stride = 32 * bits // 8
    mask = (1 << bits) - 1
    if transposed:
        @tvm.script.ir_module
        class MyModule:
            @T.prim_func
            def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                A = T.match_buffer(a, [M, K], dtype="float16")
                B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
                C = T.match_buffer(c, [M, N], dtype="float16")
                Scales = T.match_buffer(scales, [N], dtype="float16")
                Zeros = T.match_buffer(zeros, [N], dtype="float16")

                B_decompress = T.alloc_buffer([N, K], dtype="float16")
                B_rescale = T.alloc_buffer([N, K], dtype="float16")

                for i, j in T.grid(N, K):
                    with T.block("B_decompress"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> ((vj % 32) * bits) % 8) & (
                            1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
                
                for i, j in T.grid(N, K):
                    with T.block("B_rescale"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_rescale[vi, vj] = B_decompress[vi, vj] * \
                            Scales[vi].astype('float16') - Zeros[vi].astype('float16')
                            
                for i, j, k in T.grid(M, N, K):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = T.float16(0)
                        C[vi, vj] = C[vi, vj] + \
                            A[vi, vk].astype("float16") * \
                            B_rescale[vj, vk].astype("float16")
    else:
        @tvm.script.ir_module
        class MyModule:
            @T.prim_func
            def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                A = T.match_buffer(a, [M, K], dtype="float16")
                B = T.match_buffer(b, [K // 8 * bits, N], dtype="int8")
                C = T.match_buffer(c, [M, N], dtype="float16")
                Scales = T.match_buffer(scales, [N], dtype="float16")
                Zeros = T.match_buffer(zeros, [N], dtype="float16")

                B_decompress = T.alloc_buffer([K, N], dtype="float16")
                B_rescale = T.alloc_buffer([K, N], dtype="float16")

                for i, j in T.grid(K, N):
                    with T.block("B_decompress"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_decompress[vi, vj] = T.Select(((vi % 32) * bits) % 8 <= 5, ((B[(vi // 32) * group_stride + (vi % 32) * bits // 8, vj] >> (((vi % 32) * bits) % 8) & mask)).astype("float16"), (((B[(vi // 32) * group_stride + (vi % 32) * bits // 8, vj] >> ((vi % 32) * bits) % 8) & (
                            1 << (8 - ((vi % 32) * bits) % 8)) - 1).astype("int8") | ((B[(vi // 32) * group_stride + (vi % 32) * bits // 8 + 1, vj] << (8 - ((vi % 32) * bits) % 8)) & (mask << (8 - ((vi % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
                
                for i, j in T.grid(K, N):
                    with T.block("B_rescale"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_rescale[vi, vj] = B_decompress[vi, vj] * \
                            Scales[vj].astype('float16') - Zeros[vj].astype('float16')
                            
                for i, j, k in T.grid(M, N, K):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = T.float16(0)
                        C[vi, vj] = C[vi, vj] + \
                            A[vi, vk].astype("float16") * \
                            B_rescale[vk, vj].astype("float16")
    return MyModule

def get_gemv_workloads(bits, N, K, group_size=-1):
    group_stride = 32 * bits // 8
    M = 1
    N = N
    K = K
    group_size = K if group_size == -1 else group_size
    
    mask = (1 << bits) - 1
    vec = 8 if bits == 3 else 8
    num_warps = 4
    warp_size = 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype="float16")
            B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
            C = T.match_buffer(c, [M, N], dtype="float16")
            Scales = T.match_buffer(scales, [N, K // group_size], dtype="float16")
            Zeros = T.match_buffer(zeros, [N, K // group_size], dtype="float16")
            
            B_decompress = T.alloc_buffer([N, K], dtype="float16")
            B_rescale= T.alloc_buffer([N, K], dtype="float16")

            for i, j  in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> ((vj % 32) * bits) % 8) & (1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16")) 
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * \
                        Scales[vi, vj // group_size].astype('float16') - Zeros[vi, vj // group_size].astype('float16')
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")

    return MyModule

def _apply_dynamic_gemm_schedule(bits, M, N, K, group_size, config):
    from tvm.tir.tensor_intrin.cuda import (
        WMMA_FILL_16x16x16_F16_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
        WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
        WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
    )
    group_size = K if group_size == -1 else group_size
    group_stride = 32 * bits // 8
    mask = (1 << bits) - 1
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    warp_size = 32
    BM = config['BM']
    BN = config['BN']
    BK = config['BK']
    block_row_warps = config['block_row_warps']
    block_col_warps = config['block_col_warps']
    raster = config['raster']
    stage = config['stage']
    warp_row_tiles = BM // (wmma_m * block_row_warps)
    warp_col_tiles = BN // (wmma_n * block_col_warps)
    chunk = BK // (wmma_k)
    vec = 8
    shared_pad = 8
    MPAD = (M + block_row_warps * warp_row_tiles * wmma_m - 1) // (
    block_row_warps * warp_row_tiles * wmma_m
    ) * block_row_warps * warp_row_tiles * wmma_m
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle, m: T.int32):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [m, K], dtype="float16")
            B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
            C = T.match_buffer(c, [M, N], dtype="float16")
            Scales = T.match_buffer(scales, [K // group_size, N], dtype="float16")
            Zeros = T.match_buffer(zeros, [K // group_size, N], dtype="float16")
            APad = T.alloc_buffer([MPAD, K], dtype="float16")

            B_decompress = T.alloc_buffer([N, K], dtype="float16")
            B_rescale = T.alloc_buffer([N, K], dtype="float16")

            for i, k in T.grid(MPAD, K):
                with T.block("APad"):
                    vi, vk = T.axis.remap("SS", [i, k])
                    APad[vi, vk] = T.if_then_else(vi < m , A[vi, vk], T.float16(0), dtype="float16")
                    
            for i, j in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> ((vj % 32) * bits) % 8) & (
                        1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * \
                        Scales[vj // group_size, vi].astype('float16') - Zeros[vj // group_size, vi].astype('float16')
                        
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] = C[vi, vj] + \
                        APad[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")
    ir_module = MyModule
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_apad = sch.get_block("APad")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    block_shared_A = sch.cache_read(block_b, 0, "shared")
    block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
    block_shared_B = sch.cache_read(block_b, 1, "shared")
    block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
    block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

    sch.compute_inline(block_apad)
    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)
    (i, j, k) = sch.get_loops(block_b)
    i, kernel_i = sch.split(i, factors=[None, wmma_m])
    j, kernel_j = sch.split(j, factors=[None, wmma_n])
    k, kernel_k = sch.split(k, factors=[None, wmma_k])
    block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
    block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
    if raster > 0:
        block_j, block_k = sch.split(block_j, factors=[None, raster])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_k, block_i, block_j, i, j, ko, ki,
                    ii, jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_k, "blockIdx.z")
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
    else:
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii,
                    jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")

    # cache read A from global memory to shared_memory
    sch.compute_at(block_shared_local_A, ki)
    sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
    sch.compute_at(block_shared_local_B, ki)
    sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j)

    A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
    A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
        A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
    sch.vectorize(A_shared_vi)
    sch.bind(A_shared_tx, "threadIdx.x")
    sch.bind(A_shared_ty, "threadIdx.y")
    sch.bind(A_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)
    
    B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
    B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
        B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size,  vec // (8 // bits) if bits==4 else 1])
    sch.vectorize(B_shared_vi)
    sch.bind(B_shared_tx, "threadIdx.x")
    sch.bind(B_shared_ty, "threadIdx.y")
    sch.bind(B_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)

    A_local_i, A_local_j = sch.get_loops(block_shared_local_A)[-2:]
    A_local_i, A_local_kernel_i = sch.split(A_local_i, factors=[None, wmma_m])
    A_local_j, A_local_kernel_j = sch.split(A_local_j, factors=[None, wmma_k])
    sch.reorder(A_local_i, A_local_j, A_local_kernel_i, A_local_kernel_j)

    B_local_i, B_local_j = sch.get_loops(block_shared_local_B)[-2:]
    B_local_i, B_local_kernel_i = sch.split(B_local_i, factors=[None, wmma_n])
    B_local_j, B_local_kernel_j = sch.split(B_local_j, factors=[None, wmma_k])
    sch.reorder(B_local_i, B_local_j, B_local_kernel_i, B_local_kernel_j)

    C_local_i, C_local_j = sch.get_loops(block_local_C)[-2:]
    C_local_i, C_local_kernel_i = sch.split(C_local_i, factors=[None, wmma_m])
    C_local_j, C_local_kernel_j = sch.split(C_local_j, factors=[None, wmma_n])
    sch.reorder(C_local_i, C_local_j, C_local_kernel_i, C_local_kernel_j)

    # decompose reduction
    init_block_b = sch.decompose_reduction(block_b, ko)

    # transpose layout

    init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
    sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)

    block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(block_shared_local_A)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_A)
                [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)

    block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(block_shared_local_B)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_B)
                [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
    sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

    sch.tensorize(sch.get_loops(block_local_C)[-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)

    if stage > 1:

        sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
        sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        sch.annotate(ko, ann_key="software_pipeline_stage",
                    ann_val=[0, 0, 0, stage - 1, 0])
        sch.annotate(ko, ann_key="software_pipeline_order",
                    ann_val=[0, 1, 3, 2, 4])
        sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])

    ctx = tvm.cuda(0)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        cuda_mod = tvm.build(sch.mod, target="cuda")
    return cuda_mod

if __name__ == '__main__':
    import nni
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=-1)
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    group_size = args.group_size
    group_size = K if group_size == -1 else group_size
    ctx = tvm.cuda(0)
    params = nni.get_next_parameters()
    # params = {
    #             "block_col_warps": 4,
    #             "BN": 64,
    #             "BK": 64,
    #             "raster": 0,
    #             "stage": 1,
    #             "block_row_warps": 1,
    #             "BM": 16    
    #         }
    if args.M == 1:
        # no need to generate gemv kernel.
        pass
    else:
        # ir_module = get_gemm_workloads(args.bits, args.N, args.K)
        # cuda_mod = _apply_gemm_schedule(ir_module, args.bits, args.K, params)
        cuda_mod = _apply_dynamic_gemm_schedule(args.bits, args.M, args.N, args.K, group_size, params)
        a_np = (np.random.rand(M, K)).astype("float16")
        b_np = np.random.randint(0, 2**args.bits, size=(N, K // 8 * args.bits)).astype("int8")
        c_np = np.zeros((M, N)).astype("float16")
        scales_np = np.random.rand(N * (K // group_size)).reshape((N, K // group_size)).astype("float16")
        zeros_np = np.random.rand(N * (K // group_size)).reshape((N, K // group_size)).astype("float16")
        cuda_a = tvm.nd.array(a_np, device=ctx)
        cuda_b = tvm.nd.array(b_np, device=ctx)
        cuda_c = tvm.nd.array(c_np, device=ctx)
        scales = tvm.nd.array(scales_np, device=ctx)
        zeros = tvm.nd.array(zeros_np, device=ctx)
        cuda_mod(cuda_a, cuda_b, cuda_c, scales, zeros, M)
        timer_cuda_mod = cuda_mod.time_evaluator(cuda_mod.entry_name, dev=ctx, number=10)
        t = timer_cuda_mod(cuda_a, cuda_b, cuda_c, scales, zeros, M).mean
        nni.report_final_result(t * 1e3)
