# pylint: disable=missing-docstring,invalid-name
import argparse
import os
import sys
from typing import Callable, List, Optional

from tvm import meta_schedule as ms
from tvm import tir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./log_db/RedPajama-INCITE-Chat-3B-v1/",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/tmp/mlc_llm_tuning_work_dir/",
    )
    parser.add_argument("--target", type=str, default="auto")
    args = parser.parse_args()
    sys.path.insert(0, f"{args.path}/debug")
    from mod_tir_static import (  # type: ignore  # pylint: disable=import-outside-toplevel,import-error
        Module,
    )

    from mlc_llm import utils  # pylint: disable=import-outside-toplevel

    args.mod = Module
    utils.parse_target(args)
    os.makedirs(args.work_dir, exist_ok=True)
    print("target:", args.target)
    return args


def wrap(sch_func):
    return ms.space_generator.ScheduleFn(sch_func)


def wrap_without_postproc(sch_func):
    return ms.space_generator.ScheduleFn(sch_func, postprocs=[])


def sch_fused_decode_gemv(
    name_gemv: str = "matmul",
    name_decode: str = "decode",
    name_epilogues: Optional[List[str]] = None,
) -> Callable:
    def sch_func(sch: tir.Schedule) -> None:
        gemv = sch.get_block(name_gemv)
        decode = sch.get_block(name_decode)
        epilogues = (
            [sch.get_block(n) for n in name_epilogues]
            if name_epilogues is not None
            else []
        )
        # Step 1. Schedule GEMV
        # [b=1, i=1, j, k]
        # split j           => [b=1, i=1, (bx, vx, tx), k]
        # fuse (b, i, bx)   => [bx, vx, tx, (k)]
        # split k           => [bx, vx, tx, (ko, ki * 8)]
        rb = sch.cache_write(gemv, 0, "local")
        b, i, j, k = sch.get_loops(gemv)
        assert sch.get(b).extent.value == 1
        assert sch.get(i).extent.value == 1
        len_bx, len_vx, len_tx = sch.sample_perfect_tile(
            j, n=3, max_innermost_factor=-1
        )
        bx, vx, tx = sch.split(j, [len_bx, len_vx, len_tx])
        bx = sch.fuse(b, i, bx)
        k_o, k_8 = sch.split(k, [None, 8])
        k_o, k_i = sch.split(
            k_o,
            sch.sample_perfect_tile(k_o, n=2, max_innermost_factor=-1),
        )
        k_i = sch.fuse(k_i, k_8)
        sch.unroll(k_i)
        sch.bind(bx, thread_axis="blockIdx.x")
        sch.bind(vx, thread_axis="vthread.x")
        sch.bind(tx, thread_axis="threadIdx.x")
        sch.reorder(bx, vx, tx, k_o, k_i)
        # Step 2. Schedule decode: move to under threadIdx.x and fetch separately for each thread
        sch.compute_at(decode, k_o, preserve_unit_loops=True)
        sch.set_scope(decode, 0, "local")
        _, unroll = sch.split(sch.get_loops(decode)[-2], [None, 8])
        sch.unroll(unroll)

        # Step 3. Cooperative fetch GEMV
        def cooperative_fetch(block, tx):
            block = sch.cache_read(block, 0, "shared")
            sch.compute_at(block, tx, preserve_unit_loops=True)
            l = sch.fuse(*sch.get_loops(block)[-2:])
            len_vector = sch.sample_categorical(
                [1, 2, 3, 4],
                probs=[0.25, 0.25, 0.25, 0.25],
            )
            _, tx, vec = sch.split(l, [None, len_tx, len_vector])
            sch.bind(tx, thread_axis="threadIdx.x")
            sch.vectorize(vec)
            sch.storage_align(block, buffer_index=0, axis=-2, factor=32, offset=8)

        cooperative_fetch(gemv, tx)
        # Step 4. Schedule epilogue
        for epilogue in epilogues[:-1]:
            sch.compute_inline(epilogue)
        if epilogues:
            sch.reverse_compute_inline(epilogues[-1])
        sch.reverse_compute_at(rb, tx, preserve_unit_loops=True)
        # Step 5. Postprocess: decompose reduction
        sch.decompose_reduction(gemv, k_o)
        sch.show(black_format=False)

    return wrap(sch_func)


def sch_decode(
    name_decode: str = "decode",
    name_transpose: Optional[str] = "T_transpose",
):
    def sch_func(sch: tir.Schedule) -> None:
        decode = sch.get_block(name_decode)
        # Step 1. Tile the decoding
        i, j = sch.get_loops(decode)
        i, i_8 = sch.split(i, factors=[None, 8])
        len_by, len_ty, len_i = sch.sample_perfect_tile(i, n=3, max_innermost_factor=4)
        len_bx, len_tx = sch.sample_perfect_tile(j, n=2, max_innermost_factor=16)
        by, ty, i = sch.split(i, [len_by, len_ty, len_i])
        bx, tx = sch.split(j, [len_bx, len_tx])
        sch.reorder(by, bx, ty, tx, i, i_8)
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.unroll(i_8)
        # Step 2. Cache results in shared memory
        rb = sch.cache_write(decode, 0, storage_scope="shared")
        if name_transpose is None:
            epilogue = rb
        else:
            sch.compute_inline(rb)
            epilogue = sch.get_block(name_transpose)
        # Step 3. Schedule the shared memory write back
        sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
        l = sch.fuse(*sch.get_loops(epilogue)[-2:])
        _, ty, tx, vec = sch.split(l, factors=[None, len_ty, len_tx, 4])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        sch.storage_align(decode, buffer_index=0, axis=0, factor=32, offset=1)
        sch.mod.show(black_format=False)

    return wrap(sch_func)


def sch_fused_decode_gemv_multi_k(
    name_decode: str = "decode",
    name_gemv="matmul",
    name_epilogues: Optional[List[str]] = None,
):
    def sch_func(sch: tir.Schedule) -> None:
        gemv = sch.get_block(name_gemv)
        decode = sch.get_block(name_decode)
        epilogues = (
            [sch.get_block(n) for n in name_epilogues]
            if name_epilogues is not None
            else []
        )
        # Step 1. Schedule GEMV
        # [b=1, i=1, j, k]
        # split j           => [b=1, i=1, (bx, vx, tx), k]
        # fuse (b, i, bx)   => [bx, vx, tx, (k)]
        # split k           => [bx, vx, tx, (ko, ki * 8)]
        rb = sch.cache_write(gemv, 0, "local")
        b, i, j, k = sch.get_loops(gemv)
        assert sch.get(b).extent.value == 1
        assert sch.get(i).extent.value == 1
        len_bx, len_vx, len_tx = sch.sample_perfect_tile(
            j, n=3, max_innermost_factor=-1
        )
        bx, vx, tx = sch.split(j, [len_bx, len_vx, len_tx])
        bx = sch.fuse(b, i, bx)
        k_o, k_8 = sch.split(k, [None, 8])
        k_o, k_m, k_i = sch.split(
            k_o,
            sch.sample_perfect_tile(k_o, n=3, max_innermost_factor=-1),
        )
        k_i = sch.fuse(k_i, k_8)
        sch.unroll(k_i)
        sch.bind(bx, thread_axis="blockIdx.x")
        sch.bind(vx, thread_axis="vthread.x")
        sch.bind(tx, thread_axis="threadIdx.x")
        sch.reorder(bx, vx, tx, k_o, k_m, k_i)
        # Step 2. Schedule decode: move to under threadIdx.x and fetch separately for each thread
        sch.compute_at(decode, k_m, preserve_unit_loops=True)
        sch.set_scope(decode, 0, "local")
        _, unroll = sch.split(sch.get_loops(decode)[-2], [None, 8])
        sch.unroll(unroll)

        # Step 3. Cooperative fetch GEMV
        def cooperative_fetch(block, tx):
            block = sch.cache_read(block, 0, "shared")
            sch.compute_at(block, tx, preserve_unit_loops=True)
            l = sch.fuse(*sch.get_loops(block)[-2:])
            len_vector = sch.sample_categorical(
                [1, 2, 3, 4],
                probs=[0.25, 0.25, 0.25, 0.25],
            )
            _, tx, vec = sch.split(l, [None, len_tx, len_vector])
            sch.bind(tx, thread_axis="threadIdx.x")
            sch.vectorize(vec)
            sch.storage_align(block, buffer_index=0, axis=-2, factor=32, offset=8)

        cooperative_fetch(gemv, k_o)
        # Step 4. Schedule epilogue
        for epilogue in epilogues[:-1]:
            sch.compute_inline(epilogue)
        if epilogues:
            sch.reverse_compute_inline(epilogues[-1])
        sch.reverse_compute_at(rb, tx, preserve_unit_loops=True)
        # Step 5. Postprocess: decompose reduction
        sch.decompose_reduction(gemv, k_o)

    return sch_func


def sch_fused_NT_matmul3_add6_gelu1_cast11(sch: tir.Schedule) -> None:
    # fmt: off
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="compute", func_name="main")
    b4 = sch.get_block(name="T_multiply_1", func_name="main")
    b5 = sch.get_block(name="T_add_1", func_name="main")
    b6 = sch.get_block(name="T_multiply_2", func_name="main")
    b7 = sch.get_block(name="compute_1", func_name="main")
    b8 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l9, l10, l11, l12 = sch.get_loops(block=b0)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64)
    l18, l19, l20, l21, l22 = sch.split(loop=l9, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True)
    v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64)
    l28, l29, l30, l31, l32 = sch.split(loop=l10, factors=[v23, v24, v25, v26, v27], preserve_unit_iters=True)
    v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l11, n=5, max_innermost_factor=64)
    l38, l39, l40, l41, l42 = sch.split(loop=l11, factors=[v33, v34, v35, v36, v37], preserve_unit_iters=True)
    v43, v44, v45 = sch.sample_perfect_tile(loop=l12, n=3, max_innermost_factor=64)
    l46, l47, l48 = sch.split(loop=l12, factors=[v43, v44, v45], preserve_unit_iters=True)
    sch.reorder(l18, l28, l38, l19, l29, l39, l20, l30, l40, l46, l47, l21, l31, l41, l48, l22, l32, l42)
    l49 = sch.fuse(l18, l28, l38, preserve_unit_iters=True)
    sch.bind(loop=l49, thread_axis="blockIdx.x")
    l50 = sch.fuse(l19, l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="vthread.x")
    l51 = sch.fuse(l20, l30, l40, preserve_unit_iters=True)
    sch.bind(loop=l51, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b52 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b52, loop=l51, preserve_unit_loops=True, index=-1)
    b53 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b53, loop=l46, preserve_unit_loops=True, index=-1)
    l54, l55, l56, l57, l58, l59, l60 = sch.get_loops(block=b53)
    l61 = sch.fuse(l58, l59, l60, preserve_unit_iters=True)
    v62 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25])
    sch.annotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch", ann_val=v62)
    b63 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b63, loop=l46, preserve_unit_loops=True, index=-1)
    l64, l65, l66, l67, l68, l69 = sch.get_loops(block=b63)
    l70 = sch.fuse(l68, l69, preserve_unit_iters=True)
    v71 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25])
    sch.annotate(block_or_loop=b63, ann_key="meta_schedule.cooperative_fetch", ann_val=v71)
    sch.reverse_compute_inline(block=b7)
    sch.compute_inline(block=b5)
    sch.compute_inline(block=b4)
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b2)
    sch.compute_inline(block=b1)
    sch.reverse_compute_inline(block=b6)
    v72 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2])
    sch.annotate(block_or_loop=b8, ann_key="meta_schedule.unroll_explicit", ann_val=v72)
    # fmt: on


def sch_fused_NT_matmul2_add2_gelu(sch: tir.Schedule) -> None:
    # fmt: off
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="compute", func_name="main")
    b4 = sch.get_block(name="T_multiply_1", func_name="main")
    b5 = sch.get_block(name="T_add_1", func_name="main")
    b6 = sch.get_block(name="T_multiply_2", func_name="main")
    b7 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l8, l9, l10, l11 = sch.get_loops(block=b0)
    v12, v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l17, l18, l19, l20, l21 = sch.split(loop=l8, factors=[v12, v13, v14, v15, v16], preserve_unit_iters=True)
    v22, v23, v24, v25, v26 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[8, 4, 2, 1, 2])
    l27, l28, l29, l30, l31 = sch.split(loop=l9, factors=[v22, v23, v24, v25, v26], preserve_unit_iters=True)
    v32, v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64, decision=[1, 16, 4, 4, 40])
    l37, l38, l39, l40, l41 = sch.split(loop=l10, factors=[v32, v33, v34, v35, v36], preserve_unit_iters=True)
    v42, v43, v44 = sch.sample_perfect_tile(loop=l11, n=3, max_innermost_factor=64, decision=[320, 4, 2])
    l45, l46, l47 = sch.split(loop=l11, factors=[v42, v43, v44], preserve_unit_iters=True)
    sch.reorder(l17, l27, l37, l18, l28, l38, l19, l29, l39, l45, l46, l20, l30, l40, l47, l21, l31, l41)
    l48 = sch.fuse(l17, l27, l37, preserve_unit_iters=True)
    sch.bind(loop=l48, thread_axis="blockIdx.x")
    l49 = sch.fuse(l18, l28, l38, preserve_unit_iters=True)
    sch.bind(loop=l49, thread_axis="vthread.x")
    l50 = sch.fuse(l19, l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b51 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b51, loop=l50, preserve_unit_loops=True, index=-1)
    b52 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b52, loop=l45, preserve_unit_loops=True, index=-1)
    l53, l54, l55, l56, l57, l58, l59 = sch.get_loops(block=b52)
    l60 = sch.fuse(l57, l58, l59, preserve_unit_iters=True)
    v61 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v61)
    b62 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b62, loop=l45, preserve_unit_loops=True, index=-1)
    l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b62)
    l69 = sch.fuse(l67, l68, preserve_unit_iters=True)
    v70 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b62, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
    sch.compute_inline(block=b5)
    sch.compute_inline(block=b4)
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b2)
    sch.compute_inline(block=b1)
    sch.reverse_compute_inline(block=b6)
    v71 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
    sch.annotate(block_or_loop=b7, ann_key="meta_schedule.unroll_explicit", ann_val=v71)
    # fmt: on


def sch_fused_decode4_matmul3(sch: tir.Schedule) -> None:
    # fmt: off
    b0 = sch.get_block(name="compute", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="T_multiply_1", func_name="main")
    b4 = sch.get_block(name="T_add_1", func_name="main")
    b5 = sch.get_block(name="T_add_2", func_name="main")
    b6 = sch.get_block(name="matmul", func_name="main")
    b7 = sch.get_block(name="root", func_name="main")
    sch.pad_einsum(b6, [1, 64, 64])
    b8 = sch.get_producers(b6)[-1]
    b9 = sch.get_consumers(b6)[-1]
    sch.annotate(block_or_loop=b6, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l8, l9, l10 = sch.get_loops(block=b6)
    v11, v12, v13, v14, v15 = 1, 1, 1, 1, 1
    l16, l17, l18, l19, l20 = sch.split(loop=l8, factors=[v11, v12, v13, v14, v15], preserve_unit_iters=True)
    v21, v22, v23, v24, v25 = None, 1, 64, 1, 2
    l26, l27, l28, l29, l30 = sch.split(loop=l9, factors=[v21, v22, v23, v24, v25], preserve_unit_iters=True)
    v31, v32, v33 = None, 4, 4
    l34, l35, l36 = sch.split(loop=l10, factors=[v31, v32, v33], preserve_unit_iters=True)
    sch.reorder(l16, l26, l17, l27, l18, l28, l34, l35, l19, l29, l36, l20, l30)
    l37 = sch.fuse(l16, l26, preserve_unit_iters=True)
    sch.bind(loop=l37, thread_axis="blockIdx.x")
    l38 = sch.fuse(l17, l27, preserve_unit_iters=True)
    sch.bind(loop=l38, thread_axis="vthread.x")
    l39 = sch.fuse(l18, l28, preserve_unit_iters=True)
    sch.bind(loop=l39, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b6, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b6, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b40 = sch.cache_write(block=b6, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b40, loop=l39, preserve_unit_loops=True, index=-1)
    b41 = sch.cache_read(block=b6, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b6])
    sch.compute_at(block=b41, loop=l34, preserve_unit_loops=True, index=-1)
    l42, l43, l44, l45, l46, l47 = sch.get_loops(block=b41)
    l48 = sch.fuse(l46, l47, preserve_unit_iters=True)
    v49 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b41, ann_key="meta_schedule.cooperative_fetch", ann_val=v49)
    b50 = sch.cache_read(block=b6, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b6])
    sch.compute_at(block=b50, loop=l34, preserve_unit_loops=True, index=-1)
    l51, l52, l53, l54, l55, l56 = sch.get_loops(block=b50)
    l57 = sch.fuse(l55, l56, preserve_unit_iters=True)
    v58 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b50, ann_key="meta_schedule.cooperative_fetch", ann_val=v58)
    sch.compute_inline(block=b5)
    sch.compute_inline(block=b4)
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b2)
    sch.compute_inline(block=b1)
    sch.compute_inline(block=b0)
    v59 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b7, ann_key="meta_schedule.unroll_explicit", ann_val=v59)
    sch.unannotate(block_or_loop=b41, ann_key="meta_schedule.cooperative_fetch")
    l60, l61, l62, l63, l64 = sch.get_loops(block=b41)
    l65, l66 = sch.split(loop=l64, factors=[None, 64], preserve_unit_iters=True)
    sch.bind(loop=l66, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b50, ann_key="meta_schedule.cooperative_fetch")
    l67, l68, l69, l70, l71 = sch.get_loops(block=b50)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    b75 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b75, ann_key="meta_schedule.unroll_explicit")
    _, b76, b77, b78, b79, _ = sch.get_child_blocks(b75)
    l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b76)
    l86, l87, l88, l89, l90, l91, l92 = sch.get_loops(block=b77)
    l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b78)
    sch.annotate(block_or_loop=l93, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l93, ann_key="pragma_unroll_explicit", ann_val=1)
    b108 = sch.get_block(name="matmul", func_name="main")
    l109, l110, l111, l112, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b108)
    b119 = sch.decompose_reduction(block=b108, loop=l112)
    sch.compute_inline(b8)
    sch.reverse_compute_inline(b9)
    # fmt: on


def main():
    args = _parse_args()
    path_workload = os.path.join(args.path, "database_workload.json")
    path_tuning_record = os.path.join(args.path, "database_tuning_record.json")
    if os.path.exists(path_workload):
        os.remove(path_workload)
    if os.path.exists(path_tuning_record):
        os.remove(path_tuning_record)

    args.mod.show(black_format=False)
    database = ms.tir_integration.tune_tir(
        mod=args.mod,
        target=args.target,
        work_dir=args.work_dir,
        max_trials_global=2000,
        # max_trials_per_task=2,
        runner=ms.runner.LocalRunner(timeout_sec=10),
        # runner=ms.runner.RPCRunner(
        #     ms.runner.RPCConfig(
        #         tracker_host="192.168.10.1",
        #         tracker_port=9191,
        #         tracker_key="m2-mac-mini",
        #     ),
        # ),
        special_space={
            "fused_decode3_matmul1": sch_fused_decode_gemv(
                name_epilogues=[],
            ),
            "fused_decode4_fused_matmul7_add3": sch_fused_decode_gemv(
                name_epilogues=[
                    "T_add",
                ],
            ),
            "fused_decode4_fused_matmul7_add3_add4": sch_fused_decode_gemv(
                name_epilogues=[
                    "T_add",
                    "T_add_1",
                ],
            ),
            "fused_decode5_fused_matmul9_add5_gelu1": sch_fused_decode_gemv(
                name_epilogues=[
                    "T_add",
                    "T_multiply",
                    "compute",
                    "T_multiply_1",
                    "T_add_1",
                    "T_multiply_2",
                ],
            ),
            "fused_decode6_fused_matmul10_add3_cast1_add4": sch_fused_decode_gemv_multi_k(
                name_epilogues=[
                    "T_add",
                    "compute",
                    "T_add_1",
                ],
            ),
            "decode": sch_decode(),
            "decode1": sch_decode(),
            "decode2": sch_decode(),
            "fused_NT_matmul2_add2_gelu": wrap(sch_fused_NT_matmul2_add2_gelu),
            "fused_decode4_matmul3": wrap_without_postproc(sch_fused_decode4_matmul3),
        },
    )
    if os.path.exists(path_workload):
        os.remove(path_workload)
    if os.path.exists(path_tuning_record):
        os.remove(path_tuning_record)
    database.dump_pruned(
        ms.database.JSONDatabase(
            path_workload=path_workload,
            path_tuning_record=path_tuning_record,
        )
    )


if __name__ == "__main__":
    main()
