import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os

def apply_gemv_schedule(ir_module, bits, K):
    num_warps = 4
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

    code = cuda_mod.imported_modules[0].get_source()
    code = code.replace(
        "main_kernel0", f"tir_halfxint{bits}_simt_bn{num_warps}_k{K}")

    return cuda_mod


def get_source_code(M:int, K:int, bits:int = 3):
    # get_workload -> ir_module
    # apply schedule -> cuda_mod
    # code parser -> cuda_code
    return ""



