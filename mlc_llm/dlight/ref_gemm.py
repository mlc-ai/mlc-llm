# pylint: disable=missing-docstring
import os

import numpy as np
import tvm
from tvm import tir
from tvm.script import ir as I
from tvm.script import tir as T

M = N = K = 16384
DTYPE = "float32"
TARGET = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
SMEM_TRANS_A = True
SMEM_TRANS_B = False


@tvm.register_func
def tvm_callback_cuda_postproc(code, _):
    if not os.path.exists("/tmp/"):
        os.mkdir("/tmp/")
    with open("/tmp/generated.cu", "w", encoding="utf-8") as o_f:
        o_f.write(code)
    return code


# pylint: disable=invalid-name,too-few-public-methods
@I.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # M, N, K = T.int64(), T.int64(), T.int64()
        A = T.match_buffer(a, [M, K], dtype=DTYPE)
        # AT = T.match_buffer(a, [K, M], dtype=DTYPE)
        B = T.match_buffer(b, [K, N], dtype=DTYPE)
        # BT = T.match_buffer(b, [N, K], dtype=DTYPE)
        C = T.match_buffer(c, [M, N], dtype=DTYPE)

        for i, j, k in T.grid(M, N, K):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                # C[vi, vj] = C[vi, vj] + AT[vk, vi] * B[vk, vj]
                # C[vi, vj] = C[vi, vj] + A[vi, vk] * BT[vj, vk]
                # C[vi, vj] = C[vi, vj] + AT[vk, vi] * BT[vj, vk]


# fmt: off

# fmt: on

Block_Size_X = 16
Block_Size_Y = 16
V_Thread_X = 2
V_Thread_Y = 2
INNER_Size_X = 4
INNER_Size_Y = 4
VECTOR_SIZE = 4
BK = 16

# pylint: enable=invalid-name,too-few-public-methods


def do_sch(sch: tir.Schedule):
    # pylint: disable=invalid-name

    matmul = sch.get_block("matmul")
    sch.pad_einsum(
        matmul,
        [
            V_Thread_X * Block_Size_X * INNER_Size_X,
            V_Thread_Y * Block_Size_Y * INNER_Size_Y,
            BK,
        ],
    )

    (x, y, k) = sch.get_loops(matmul)
    bx, vx, tx, xi = sch.split(
        x, factors=[None, V_Thread_X, Block_Size_X, INNER_Size_X]
    )
    by, vy, ty, yi = sch.split(
        y, factors=[None, V_Thread_Y, Block_Size_Y, INNER_Size_Y]
    )
    ko, ki = sch.split(k, [None, BK])
    sch.reorder(bx, by, vy, vx, ty, tx, ko, ki, yi, xi)
    sch.bind(bx, "blockIdx.x")
    sch.bind(by, "blockIdx.y")
    sch.bind(vy, "vthread.y")
    sch.bind(vx, "vthread.x")
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=256)
    sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

    block_cl = sch.cache_write(matmul, 0, "local")
    sch.reverse_compute_at(block_cl, tx, preserve_unit_loops=True)

    def _cooperative_fetch(index, vec_len, transpose_shared):
        block = sch.cache_read(matmul, index, "shared")
        if transpose_shared:
            sch.transform_layout(block, ("write", 0), index_map=lambda x, y: (y, x))
            sch.transform_layout(block, ("read", 0), index_map=lambda x, y: (y, x))
        sch.compute_at(block, ko)
        x, y = sch.get_loops(block)[-2:]
        print(sch.get(x).extent, sch.get(y).extent)
        _, ty, tx, vec = sch.split(
            sch.fuse(x, y),
            factors=[None, Block_Size_Y, Block_Size_X, vec_len],
        )
        sch.vectorize(vec)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def _shared_to_local(index):
        block = sch.cache_read(matmul, index, "local")
        sch.compute_at(block, ki)
        sch.vectorize(sch.get_loops(block)[-1])

    _cooperative_fetch(0, vec_len=VECTOR_SIZE, transpose_shared=SMEM_TRANS_A)
    _cooperative_fetch(1, vec_len=VECTOR_SIZE, transpose_shared=SMEM_TRANS_B)
    _shared_to_local(0)
    _shared_to_local(1)

    try:
        sch.compute_inline(sch.get_block("AT_pad"))
    except:
        pass
    try:
        sch.compute_inline(sch.get_block("A_pad"))
    except:
        pass
    try:
        sch.compute_inline(sch.get_block("B_pad"))
    except:
        pass
    try:
        sch.reverse_compute_inline(sch.get_block("C_pad"))
    except:
        pass

    sch.decompose_reduction(matmul, ko)
    sch.mod.show(black_format=False)
    # pylint: enable=invalid-name


def profile(mod):
    cuda_mod = tvm.build(mod, target=TARGET)

    device = tvm.cuda(0)
    cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(DTYPE), device)
    cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(DTYPE), device)
    cuda_c = tvm.nd.array(np.zeros((M, N)).astype(DTYPE), device)
    cuda_mod(cuda_a, cuda_b, cuda_c)

    num_runs = 10
    timer_cuda_mod = cuda_mod.time_evaluator(
        cuda_mod.entry_name, device, number=num_runs
    )
    time = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean
    gflops = 2 * M * N * K / (time * 1e3) / 1e6
    print(
        "average time cost of %d runs = %g ms, %g GFLOPS."
        % (num_runs, time * 1e3, gflops)
    )


def main():
    sch = tvm.tir.Schedule(MyModule)
    do_sch(sch)
    profile(sch.mod)


if __name__ == "__main__":
    main()
