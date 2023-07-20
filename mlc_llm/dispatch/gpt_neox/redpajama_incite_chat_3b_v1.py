# pylint: disable=missing-docstring,line-too-long,invalid-name,too-many-statements,too-many-locals
import tvm
from tvm import tir
from tvm.script import tir as T

from .redpajama_incite_chat_3b_v1_mod import Module as MOD

# fmt: off

def fused_NT_matmul1_add4(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 2, 32, 1, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[80, 1, 4, 4, 2])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[128, 5, 4])
    l40, l41, l42 = sch.split(loop=l6, factors=[v37, v38, v39], preserve_unit_iters=True)
    sch.reorder(l12, l22, l32, l13, l23, l33, l14, l24, l34, l40, l41, l15, l25, l35, l42, l16, l26, l36)
    l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="blockIdx.x")
    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="vthread.x")
    l45 = sch.fuse(l14, l24, l34, preserve_unit_iters=True)
    sch.bind(loop=l45, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b46 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b46, loop=l45, preserve_unit_loops=True, index=-1)
    b47 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b47, loop=l40, preserve_unit_loops=True, index=-1)
    l52, l53, l54 = sch.get_loops(block=b47)[-3:]
    sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    l62, l63 = sch.get_loops(block=b57)[-2:]
    sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    l71 = sch.get_loops(block=b47)[-1]
    _, l73, l74 = sch.split(loop=l71, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    l79 = sch.get_loops(block=b57)[-1]
    _, l81 = sch.split(loop=l79, factors=[None, 64], preserve_unit_iters=True)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b82 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv9_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b85, _ = sch.get_child_blocks(b82)
    l100 = sch.get_loops(block=b85)[0]
    sch.annotate(block_or_loop=l100, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l100, ann_key="pragma_unroll_explicit", ann_val=1)
    b118 = sch.get_block(name="NT_matmul", func_name="main")
    l122 = sch.get_loops(block=b118)[4]
    sch.decompose_reduction(block=b118, loop=l122)


def fused_NT_matmul1_add4_add5(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="T_add_1", func_name="main")
    b3 = sch.get_block(name="root", func_name="main")

    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l4, l5, l6, l7 = sch.get_loops(block=b0)
    v8, v9, v10, v11, v12 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l13, l14, l15, l16, l17 = sch.split(loop=l4, factors=[v8, v9, v10, v11, v12], preserve_unit_iters=True)
    v18, v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[2, 8, 4, 2, 1])
    l23, l24, l25, l26, l27 = sch.split(loop=l5, factors=[v18, v19, v20, v21, v22], preserve_unit_iters=True)
    v28, v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[40, 2, 16, 1, 2])
    l33, l34, l35, l36, l37 = sch.split(loop=l6, factors=[v28, v29, v30, v31, v32], preserve_unit_iters=True)
    v38, v39, v40 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64, decision=[160, 4, 4])
    l41, l42, l43 = sch.split(loop=l7, factors=[v38, v39, v40], preserve_unit_iters=True)
    sch.reorder(l13, l23, l33, l14, l24, l34, l15, l25, l35, l41, l42, l16, l26, l36, l43, l17, l27, l37)

    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="blockIdx.x")
    l45 = sch.fuse(l14, l24, l34, preserve_unit_iters=True)
    sch.bind(loop=l45, thread_axis="vthread.x")
    l46 = sch.fuse(l15, l25, l35, preserve_unit_iters=True)
    sch.bind(loop=l46, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b47 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b47, loop=l46, preserve_unit_loops=True, index=-1)
    b48 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b48, loop=l41, preserve_unit_loops=True, index=-1)
    l53, l54, l55 = sch.get_loops(block=b48)[-3:]
    sch.fuse(l53, l54, l55, preserve_unit_iters=True)
    v57 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch", ann_val=v57)
    b58 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b58, loop=l41, preserve_unit_loops=True, index=-1)
    l63, l64 = sch.get_loops(block=b58)[-2:]
    sch.fuse(l63, l64, preserve_unit_iters=True)
    v66 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v66)
    sch.reverse_compute_inline(block=b2)
    sch.reverse_compute_inline(block=b1)
    v67 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2], decision=3)
    sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v67)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch")
    l72 = sch.get_loops(block=b48)[-1]
    _, l74, l75 = sch.split(loop=l72, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l75)
    sch.bind(loop=l74, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
    l80 = sch.get_loops(block=b58)[-1]
    _, l82, l83 = sch.split(loop=l80, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l83)
    sch.bind(loop=l82, thread_axis="threadIdx.x")
    b84 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b84, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv49_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b87, _ = sch.get_child_blocks(b84)
    l103 = sch.get_loops(block=b87)[0]
    sch.annotate(block_or_loop=l103, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l103, ann_key="pragma_unroll_explicit", ann_val=1)

    b121 = sch.get_block(name="NT_matmul", func_name="main")
    l125 = sch.get_loops(block=b121)[4]
    sch.decompose_reduction(block=b121, loop=l125)


def fused_NT_matmul2_divide1_maximum1_minimum1_cast9(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 1, 32, 32, 1])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l3, [None, 32])
    l8, l9 = sch.split(l4, [None, 32])
    sch.reorder(l6, l8, l1, l2, l7, l9, l5)

    b1 = sch.get_block(name="T_divide", func_name="main")
    b2 = sch.get_block(name="T_maximum", func_name="main")
    b3 = sch.get_block(name="T_minimum", func_name="main")
    b4 = sch.get_block(name="compute", func_name="main")
    b5 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, _, l6, l7, l8, l9, l10 = sch.get_loops(block=b0)
    v11, v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l16, l17, l18, l19, l20 = sch.split(loop=l6, factors=[v11, v12, v13, v14, v15], preserve_unit_iters=True)
    v21, v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[16, 1, 1, 1, 2])
    l26, l27, l28, l29, l30 = sch.split(loop=l7, factors=[v21, v22, v23, v24, v25], preserve_unit_iters=True)
    v31, v32, v33, v34, v35 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[1, 1, 16, 2, 4])
    l36, l37, l38, l39, l40 = sch.split(loop=l8, factors=[v31, v32, v33, v34, v35], preserve_unit_iters=True)
    v41, v42, v43, v44, v45 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[8, 1, 16, 1, 1])
    l46, l47, l48, l49, l50 = sch.split(loop=l9, factors=[v41, v42, v43, v44, v45], preserve_unit_iters=True)
    v51, v52, v53 = sch.sample_perfect_tile(loop=l10, n=3, max_innermost_factor=64, decision=[4, 20, 1])
    l54, l55, l56 = sch.split(loop=l10, factors=[v51, v52, v53], preserve_unit_iters=True)
    sch.reorder(l16, l26, l36, l46, l17, l27, l37, l47, l18, l28, l38, l48, l54, l55, l19, l29, l39, l49, l56, l20, l30, l40, l50)
    l57 = sch.fuse(l16, l26, l36, l46, preserve_unit_iters=True)
    sch.bind(loop=l57, thread_axis="blockIdx.x")
    l58 = sch.fuse(l17, l27, l37, l47, preserve_unit_iters=True)
    sch.bind(loop=l58, thread_axis="vthread.x")
    l59 = sch.fuse(l18, l28, l38, l48, preserve_unit_iters=True)
    sch.bind(loop=l59, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b60 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b60, loop=l59, preserve_unit_loops=True, index=-1)
    b61 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b61, loop=l54, preserve_unit_loops=True, index=-1)
    l66, l67, l68, l69 = sch.get_loops(block=b61)[-4: ]
    sch.fuse(l66, l67, l68, l69, preserve_unit_iters=True)
    v71 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v71)
    b72 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b72, loop=l54, preserve_unit_loops=True, index=-1)
    l77, l78, l79, l80 = sch.get_loops(block=b72)[-4:]
    sch.fuse(l77, l78, l79, l80, preserve_unit_iters=True)
    v82 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b72, ann_key="meta_schedule.cooperative_fetch", ann_val=v82)
    sch.reverse_compute_inline(block=b4)
    sch.reverse_compute_inline(block=b3)
    sch.reverse_compute_inline(block=b2)
    sch.reverse_compute_inline(block=b1)
    v83 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
    sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v83)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch")
    l88 = sch.get_loops(block=b61)[-1]
    _, l90, l91 = sch.split(loop=l88, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l91)
    sch.bind(loop=l90, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b72, ann_key="meta_schedule.cooperative_fetch")
    l96 = sch.get_loops(block=b72)[-1]
    _, l98, l99 = sch.split(loop=l96, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l99)
    sch.bind(loop=l98, thread_axis="threadIdx.x")
    b100 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b100, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv36_pad")
    sch.compute_inline(b1)
    b1 = sch.get_block("lv37_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    b140 = sch.get_block(name="NT_matmul", func_name="main")
    l144 = sch.get_loops(block=b140)[5]
    sch.decompose_reduction(block=b140, loop=l144)


def fused_NT_matmul3_add6_gelu1_cast11(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="compute", func_name="main")
    b4 = sch.get_block(name="T_multiply_1", func_name="main")
    b5 = sch.get_block(name="T_add_1", func_name="main")
    b6 = sch.get_block(name="T_multiply_2", func_name="main")
    b7 = sch.get_block(name="compute_1", func_name="main")
    b8 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l9, l10, l11, l12 = sch.get_loops(block=b0)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l18, l19, l20, l21, l22 = sch.split(loop=l9, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True)
    v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64, decision=[1, 1, 32, 4, 1])
    l28, l29, l30, l31, l32 = sch.split(loop=l10, factors=[v23, v24, v25, v26, v27], preserve_unit_iters=True)
    v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l11, n=5, max_innermost_factor=64, decision=[320, 1, 4, 8, 1])
    l38, l39, l40, l41, l42 = sch.split(loop=l11, factors=[v33, v34, v35, v36, v37], preserve_unit_iters=True)
    v43, v44, v45 = sch.sample_perfect_tile(loop=l12, n=3, max_innermost_factor=64, decision=[80, 32, 1])
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
    l58, l59, l60 = sch.get_loops(block=b53)[-3:]
    sch.fuse(l58, l59, l60, preserve_unit_iters=True)
    v62 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch", ann_val=v62)
    b63 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b63, loop=l46, preserve_unit_loops=True, index=-1)
    l68, l69 = sch.get_loops(block=b63)[-2:]
    sch.fuse(l68, l69, preserve_unit_iters=True)
    v71 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b63, ann_key="meta_schedule.cooperative_fetch", ann_val=v71)
    sch.reverse_compute_inline(block=b7)
    sch.compute_inline(block=b5)
    sch.compute_inline(block=b4)
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b2)
    sch.compute_inline(block=b1)
    sch.reverse_compute_inline(block=b6)
    v72 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b8, ann_key="meta_schedule.unroll_explicit", ann_val=v72)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch")
    l77 = sch.get_loops(block=b53)[-1]
    _, l79, l80 = sch.split(loop=l77, factors=[None, 32, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l80)
    sch.bind(loop=l79, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b63, ann_key="meta_schedule.cooperative_fetch")
    l85 = sch.get_loops(block=b63)[-1]
    _, l87, l88 = sch.split(loop=l85, factors=[None, 32, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l88)
    sch.bind(loop=l87, thread_axis="threadIdx.x")
    b89 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b89, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv57_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b92, _ = sch.get_child_blocks(b89)
    l108 = sch.get_loops(block=b92)[0]
    sch.annotate(block_or_loop=l108, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l108, ann_key="pragma_unroll_explicit", ann_val=1)

    b126 = sch.get_block(name="NT_matmul", func_name="main")
    l130 = sch.get_loops(block=b126)[4]
    sch.decompose_reduction(block=b126, loop=l130)


def fused_NT_matmul4_add7_cast8_cast12_add5(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="compute", func_name="main")
    b3 = sch.get_block(name="compute_1", func_name="main")
    b4 = sch.get_block(name="T_add_1", func_name="main")
    b5 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l6, l7, l8, l9 = sch.get_loops(block=b0)
    v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l15, l16, l17, l18, l19 = sch.split(loop=l6, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
    v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[2, 4, 16, 1, 1])
    l25, l26, l27, l28, l29 = sch.split(loop=l7, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
    v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[40, 1, 8, 1, 8])
    l35, l36, l37, l38, l39 = sch.split(loop=l8, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
    v40, v41, v42 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[320, 32, 1])
    l43, l44, l45 = sch.split(loop=l9, factors=[v40, v41, v42], preserve_unit_iters=True)
    sch.reorder(l15, l25, l35, l16, l26, l36, l17, l27, l37, l43, l44, l18, l28, l38, l45, l19, l29, l39)
    l46 = sch.fuse(l15, l25, l35, preserve_unit_iters=True)
    sch.bind(loop=l46, thread_axis="blockIdx.x")
    l47 = sch.fuse(l16, l26, l36, preserve_unit_iters=True)
    sch.bind(loop=l47, thread_axis="vthread.x")
    l48 = sch.fuse(l17, l27, l37, preserve_unit_iters=True)
    sch.bind(loop=l48, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b49 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b49, loop=l48, preserve_unit_loops=True, index=-1)
    b50 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b50, loop=l43, preserve_unit_loops=True, index=-1)
    l55, l56, l57 = sch.get_loops(block=b50)[-3:]
    sch.fuse(l55, l56, l57, preserve_unit_iters=True)
    v59 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b50, ann_key="meta_schedule.cooperative_fetch", ann_val=v59)
    b60 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b60, loop=l43, preserve_unit_loops=True, index=-1)
    l65, l66 = sch.get_loops(block=b60)[-2:]
    sch.fuse(l65, l66, preserve_unit_iters=True)
    v68 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v68)
    sch.reverse_compute_inline(block=b4)
    sch.reverse_compute_inline(block=b3)
    sch.reverse_compute_inline(block=b2)
    sch.reverse_compute_inline(block=b1)
    v69 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v69)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b50, ann_key="meta_schedule.cooperative_fetch")
    l74 = sch.get_loops(block=b50)[-1]
    _, l76, l77 = sch.split(loop=l74, factors=[None, 128, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l77)
    sch.bind(loop=l76, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
    l82 = sch.get_loops(block=b60)[-1]
    _, l84, l85 = sch.split(loop=l82, factors=[None, 128, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l85)
    sch.bind(loop=l84, thread_axis="threadIdx.x")
    b86 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b86, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv63_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b89, _ = sch.get_child_blocks(b86)
    l105 = sch.get_loops(block=b89)[0]
    sch.annotate(block_or_loop=l105, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l105, ann_key="pragma_unroll_explicit", ann_val=1)
    b123 = sch.get_block(name="NT_matmul", func_name="main")
    l127 = sch.get_loops(block=b123)[4]
    sch.decompose_reduction(block=b123, loop=l127)


def fused_NT_matmul4_add7_cast8_cast12_add5_cast7(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="compute", func_name="main")
    b3 = sch.get_block(name="compute_1", func_name="main")
    b4 = sch.get_block(name="T_add_1", func_name="main")
    b5 = sch.get_block(name="compute_2", func_name="main")
    b6 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l7, l8, l9, l10 = sch.get_loops(block=b0)
    v11, v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l16, l17, l18, l19, l20 = sch.split(loop=l7, factors=[v11, v12, v13, v14, v15], preserve_unit_iters=True)
    v21, v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[2, 1, 16, 2, 2])
    l26, l27, l28, l29, l30 = sch.split(loop=l8, factors=[v21, v22, v23, v24, v25], preserve_unit_iters=True)
    v31, v32, v33, v34, v35 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[64, 2, 10, 1, 2])
    l36, l37, l38, l39, l40 = sch.split(loop=l9, factors=[v31, v32, v33, v34, v35], preserve_unit_iters=True)
    v41, v42, v43 = sch.sample_perfect_tile(loop=l10, n=3, max_innermost_factor=64, decision=[256, 20, 2])
    l44, l45, l46 = sch.split(loop=l10, factors=[v41, v42, v43], preserve_unit_iters=True)
    sch.reorder(l16, l26, l36, l17, l27, l37, l18, l28, l38, l44, l45, l19, l29, l39, l46, l20, l30, l40)
    l47 = sch.fuse(l16, l26, l36, preserve_unit_iters=True)
    sch.bind(loop=l47, thread_axis="blockIdx.x")
    l48 = sch.fuse(l17, l27, l37, preserve_unit_iters=True)
    sch.bind(loop=l48, thread_axis="vthread.x")
    l49 = sch.fuse(l18, l28, l38, preserve_unit_iters=True)
    sch.bind(loop=l49, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b50 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b50, loop=l49, preserve_unit_loops=True, index=-1)
    b51 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b51, loop=l44, preserve_unit_loops=True, index=-1)
    l56, l57, l58 = sch.get_loops(block=b51)[-3:]
    sch.fuse(l56, l57, l58, preserve_unit_iters=True)
    v60 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b51, ann_key="meta_schedule.cooperative_fetch", ann_val=v60)
    b61 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b61, loop=l44, preserve_unit_loops=True, index=-1)
    l66, l67 = sch.get_loops(block=b61)[-2:]
    sch.fuse(l66, l67, preserve_unit_iters=True)
    v69 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v69)
    sch.reverse_compute_inline(block=b5)
    sch.reverse_compute_inline(block=b4)
    sch.reverse_compute_inline(block=b3)
    sch.reverse_compute_inline(block=b2)
    sch.reverse_compute_inline(block=b1)
    v70 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b6, ann_key="meta_schedule.unroll_explicit", ann_val=v70)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b51, ann_key="meta_schedule.cooperative_fetch")
    l75 = sch.get_loops(block=b51)[-1]
    _, l77, l78 = sch.split(loop=l75, factors=[None, 80, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l78)
    sch.bind(loop=l77, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch")
    l83 = sch.get_loops(block=b61)[-1]
    _, l85, l86 = sch.split(loop=l83, factors=[None, 80, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l86)
    sch.bind(loop=l85, thread_axis="threadIdx.x")
    b87 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b87, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv2047_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b90, _ = sch.get_child_blocks(b87)
    l106 = sch.get_loops(block=b90)[0]
    sch.annotate(block_or_loop=l106, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l106, ann_key="pragma_unroll_explicit", ann_val=1)
    b124 = sch.get_block(name="NT_matmul", func_name="main")
    l128 = sch.get_loops(block=b124)[4]
    sch.decompose_reduction(block=b124, loop=l128)


def fused_NT_matmul_divide_maximum_minimum_cast2(sch: tir.Schedule):
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    sch.pad_einsum(b0, [1, 1, 1, 32, 1])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l4, [None, 32])
    sch.reorder(l6, l1, l2, l3, l7, l5)

    b1 = sch.get_block(name="T_divide", func_name="main")
    b2 = sch.get_block(name="T_maximum", func_name="main")
    b3 = sch.get_block(name="T_minimum", func_name="main")
    b4 = sch.get_block(name="compute", func_name="main")
    b5 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l6, l7, l8, l9, l10 = sch.get_loops(block=b0)
    v11, v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l16, l17, l18, l19, l20 = sch.split(loop=l6, factors=[v11, v12, v13, v14, v15], preserve_unit_iters=True)
    v21, v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[4, 1, 8, 1, 1])
    l26, l27, l28, l29, l30 = sch.split(loop=l7, factors=[v21, v22, v23, v24, v25], preserve_unit_iters=True)
    v31, v32, v33, v34, v35 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l36, l37, l38, l39, l40 = sch.split(loop=l8, factors=[v31, v32, v33, v34, v35], preserve_unit_iters=True)
    v41, v42, v43, v44, v45 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[4, 1, 16, 2, 1])
    l46, l47, l48, l49, l50 = sch.split(loop=l9, factors=[v41, v42, v43, v44, v45], preserve_unit_iters=True)
    v51, v52, v53 = sch.sample_perfect_tile(loop=l10, n=3, max_innermost_factor=64, decision=[5, 8, 2])
    l54, l55, l56 = sch.split(loop=l10, factors=[v51, v52, v53], preserve_unit_iters=True)
    sch.reorder(l16, l26, l36, l46, l17, l27, l37, l47, l18, l28, l38, l48, l54, l55, l19, l29, l39, l49, l56, l20, l30, l40, l50)
    l57 = sch.fuse(l16, l26, l36, l46, preserve_unit_iters=True)
    sch.bind(loop=l57, thread_axis="blockIdx.x")
    l58 = sch.fuse(l17, l27, l37, l47, preserve_unit_iters=True)
    sch.bind(loop=l58, thread_axis="vthread.x")
    l59 = sch.fuse(l18, l28, l38, l48, preserve_unit_iters=True)
    sch.bind(loop=l59, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b60 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b60, loop=l59, preserve_unit_loops=True, index=-1)
    b61 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b61, loop=l54, preserve_unit_loops=True, index=-1)
    l66, l67, l68, l69 = sch.get_loops(block=b61)[-4:]
    sch.fuse(l66, l67, l68, l69, preserve_unit_iters=True)
    v71 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v71)
    b72 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b72, loop=l54, preserve_unit_loops=True, index=-1)
    l77, l78, l79, l80 = sch.get_loops(block=b72)[-4:]
    sch.fuse(l77, l78, l79, l80, preserve_unit_iters=True)
    v82 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b72, ann_key="meta_schedule.cooperative_fetch", ann_val=v82)
    sch.reverse_compute_inline(block=b4)
    sch.reverse_compute_inline(block=b3)
    sch.reverse_compute_inline(block=b2)
    sch.reverse_compute_inline(block=b1)
    v83 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v83)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch")
    l88 = sch.get_loops(block=b61)[-1]
    _, l90, l91 = sch.split(loop=l88, factors=[None, 128, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l91)
    sch.bind(loop=l90, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b72, ann_key="meta_schedule.cooperative_fetch")
    l96 = sch.get_loops(block=b72)[-1]
    _, l98, l99 = sch.split(loop=l96, factors=[None, 128, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l99)
    sch.bind(loop=l98, thread_axis="threadIdx.x")
    b100 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b100, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("lv2095_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    _, _, b103, _ = sch.get_child_blocks(b100)
    l119 = sch.get_loops(block=b103)[0]
    sch.annotate(block_or_loop=l119, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l119, ann_key="pragma_unroll_explicit", ann_val=1)

    b140 = sch.get_block(name="NT_matmul", func_name="main")
    l144 = sch.get_loops(block=b140)[4]
    sch.decompose_reduction(block=b140, loop=l144)


def fused_layer_norm1_cast8(sch: tir.Schedule):
    b0 = sch.get_block(name="A_red_temp", func_name="main")
    b1 = sch.get_block(name="T_layer_norm", func_name="main")
    b2 = sch.get_block(name="compute", func_name="main")
    b3 = sch.get_block(name="root", func_name="main")
    sch.reverse_compute_inline(block=b2)
    v4 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], decision=5)
    l5, l6, l7 = sch.get_loops(block=b0)
    l8, l9 = sch.split(loop=l7, factors=[None, v4], preserve_unit_iters=True)
    sch.bind(loop=l9, thread_axis="threadIdx.x")
    v10 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v10)
    l11, l12, l13 = sch.get_loops(block=b1)
    l14 = sch.fuse(l11, l12, l13, preserve_unit_iters=True)
    l15, l16, l17 = sch.split(loop=l14, factors=[None, 256, 256], preserve_unit_iters=True)
    sch.reorder(l16, l17, l15)
    sch.bind(loop=l16, thread_axis="blockIdx.x")
    sch.bind(loop=l17, thread_axis="threadIdx.x")
    l18, l19, l20, l21 = sch.get_loops(block=b0)
    l22 = sch.fuse(l18, l19, preserve_unit_iters=True)
    sch.bind(loop=l22, thread_axis="blockIdx.x")
    sch.enter_postproc()
    b23 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b23, ann_key="meta_schedule.unroll_explicit")
    b24, b25 = sch.get_child_blocks(b23)
    l26, l27, l28 = sch.get_loops(block=b24)
    sch.annotate(block_or_loop=l26, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l26, ann_key="pragma_unroll_explicit", ann_val=1)
    l29, l30, l31 = sch.get_loops(block=b25)


def layer_norm1(sch: tir.Schedule):
    b0 = sch.get_block(name="A_red_temp", func_name="main")
    b1 = sch.get_block(name="T_layer_norm", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    v3 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], decision=1)
    l4, l5, l6 = sch.get_loops(block=b0)
    l7, l8 = sch.split(loop=l6, factors=[None, v3], preserve_unit_iters=True)
    sch.bind(loop=l8, thread_axis="threadIdx.x")
    v9 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v9)
    l10, l11, l12 = sch.get_loops(block=b1)
    l13 = sch.fuse(l10, l11, l12, preserve_unit_iters=True)
    l14, l15, l16 = sch.split(loop=l13, factors=[None, 256, 256], preserve_unit_iters=True)
    sch.reorder(l15, l16, l14)
    sch.bind(loop=l15, thread_axis="blockIdx.x")
    sch.bind(loop=l16, thread_axis="threadIdx.x")
    l17, l18, l19, l20 = sch.get_loops(block=b0)
    l21 = sch.fuse(l17, l18, preserve_unit_iters=True)
    sch.bind(loop=l21, thread_axis="blockIdx.x")
    sch.enter_postproc()
    b22 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.unroll_explicit")
    b23, b24 = sch.get_child_blocks(b22)
    l25, l26, l27 = sch.get_loops(block=b23)
    sch.annotate(block_or_loop=l25, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l25, ann_key="pragma_unroll_explicit", ann_val=1)
    l28, l29, l30 = sch.get_loops(block=b24)


def matmul3(sch: tir.Schedule):
    b0 = sch.get_block(name="matmul", func_name="main")
    sch.pad_einsum(b0, [1, 1, 1, 1, 32])
    l1, l2, l3, l4, k = sch.get_loops(b0)
    k0, k1 = sch.split(k, [None, 32])
    sch.reorder(l1, l2, l3, k0, l4, k1)

    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l2, l3, l4, _, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 16, 1, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[8, 1, 10, 1, 1])
    l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
    v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[1, 32, 1])
    l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
    sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, k0, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
    l53 = sch.fuse(l12, l22, l32, l42, preserve_unit_iters=True)
    sch.bind(loop=l53, thread_axis="blockIdx.x")
    l54 = sch.fuse(l13, l23, l33, l43, preserve_unit_iters=True)
    sch.bind(loop=l54, thread_axis="vthread.x")
    l55 = sch.fuse(l14, l24, l34, l44, preserve_unit_iters=True)
    sch.bind(loop=l55, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b56 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b56, loop=l55, preserve_unit_loops=True, index=-1)
    b57 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l50, preserve_unit_loops=True, index=-1)
    l62, l63, l64, l65 = sch.get_loops(block=b57)[-4:]
    sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
    v67 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
    b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
    l73, l74, l75, l76 = sch.get_loops(block=b68)[-4:]
    sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
    v78 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
    v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    l84 = sch.get_loops(block=b57)[-1]
    _, l86, l87 = sch.split(loop=l84, factors=[None, 160, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l87)
    sch.bind(loop=l86, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
    l92 = sch.get_loops(block=b68)[-1]
    _, l94, l95 = sch.split(loop=l92, factors=[None, 160, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l95)
    sch.bind(loop=l94, thread_axis="threadIdx.x")
    b96 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b96, ann_key="meta_schedule.unroll_explicit")

    b1 = sch.get_block("A_pad")
    sch.compute_inline(b1)
    b1 = sch.get_block("B_pad")
    sch.compute_inline(b1)

    _, _, b99, _ = sch.get_child_blocks(b96)
    l115 = sch.get_loops(block=b99)[0]
    sch.annotate(block_or_loop=l115, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l115, ann_key="pragma_unroll_explicit", ann_val=1)
    b136 = sch.get_block(name="matmul", func_name="main")
    l140 = sch.get_loops(block=b136)[3]
    sch.decompose_reduction(block=b136, loop=l140)


def matmul9(sch: tir.Schedule):
    b0 = sch.get_block(name="matmul", func_name="main")
    sch.pad_einsum(b0, [1, 1, 32, 1, 32])
    l1, l2, l3, l4, k = sch.get_loops(b0)
    s0, s1 = sch.split(l3, [None, 32])
    k0, k1 = sch.split(k, [None, 32])
    sch.reorder(s0, l1, l2, s1, k0, l4, k1)

    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l2, l3, l4, _, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 1, 1, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[8, 1, 8, 2, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[2, 1, 5, 2, 4])
    l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
    v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[16, 1, 2])
    l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
    sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, k0, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
    l53 = sch.fuse(l12, l22, l32, l42, preserve_unit_iters=True)
    sch.bind(loop=l53, thread_axis="blockIdx.x")
    l54 = sch.fuse(l13, l23, l33, l43, preserve_unit_iters=True)
    sch.bind(loop=l54, thread_axis="vthread.x")
    l55 = sch.fuse(l14, l24, l34, l44, preserve_unit_iters=True)
    sch.bind(loop=l55, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b56 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b56, loop=l55, preserve_unit_loops=True, index=-1)
    b57 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l50, preserve_unit_loops=True, index=-1)
    l62, l63, l64, l65 = sch.get_loops(block=b57)[-4:]
    sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
    v67 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
    b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
    l73, l74, l75, l76 = sch.get_loops(block=b68)[-4:]
    sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
    v78 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
    v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)

    b1 = sch.get_block("A_pad")
    sch.compute_inline(b1)
    b1 = sch.get_block("B_pad")
    sch.compute_inline(b1)
    b1 = sch.get_block("matmul_pad")
    sch.reverse_compute_inline(b1)

    sch.enter_postproc()
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    l84 = sch.get_loops(block=b57)[-1]
    _, l86, l87 = sch.split(loop=l84, factors=[None, 40, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l87)
    sch.bind(loop=l86, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
    l92 = sch.get_loops(block=b68)[-1]
    _, l94, l95 = sch.split(loop=l92, factors=[None, 40, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l95)
    sch.bind(loop=l94, thread_axis="threadIdx.x")
    b96 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b96, ann_key="meta_schedule.unroll_explicit")
    b136 = sch.get_block(name="matmul", func_name="main")
    l140 = sch.get_loops(block=b136)[4]
    sch.decompose_reduction(block=b136, loop=l140)


def softmax_1xn(sch: tir.Schedule):
    has_cast = True
    if has_cast:
        b_cast = sch.get_block("compute")
        sch.reverse_compute_inline(b_cast)

    b0 = sch.get_block("T_softmax_exp")
    sch.compute_inline(b0)
    b1 = sch.get_block("T_softmax_norm")
    l2, l3, l4, l5 = sch.get_loops(b1)
    _, l7 = sch.split(l5, [None, 128])
    sch.bind(l7, "threadIdx.x")
    b8 = sch.get_block("T_softmax_expsum")
    sch.compute_at(b8, l4)
    sch.set_scope(b8, 0, "shared")
    _, _, _, l12 = sch.get_loops(b8)
    _, l14 = sch.split(l12, [None, 128])
    sch.bind(l14, "threadIdx.x")
    b15 = sch.get_block("T_softmax_maxelem")
    sch.compute_at(b15, l4)
    sch.set_scope(b15, 0, "shared")
    _, _, _, l19 = sch.get_loops(b15)
    _, l21 = sch.split(l19, [None, 128])
    sch.bind(l21, "threadIdx.x")
    l22 = sch.fuse(l2, l3, l4)
    sch.bind(l22, "blockIdx.x")


def fused_min_max_triu_te_broadcast_to(sch: tir.Schedule):
    b0 = sch.get_block("T_broadcast_to")
    sch.reverse_compute_inline(b0)
    b1 = sch.get_block("make_diag_mask_te")
    i, j = sch.get_loops(b1)
    i = sch.fuse(i, j)
    i, j = sch.split(i, [None, 128])
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")


@T.prim_func
def softmax_mxn_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    m = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, m))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], rxplaceholder[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(rxplaceholder[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_mxn_after(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_maxelem_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(m + T.int64(127)) // T.int64(128) * T.int64(128)])
            T.writes(T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_maxelem_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (m + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((m + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i])
                        T.writes(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.max(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i], T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < m, A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T.float32(-3.4028234663852886e+38)))
            for i0_i1_i2_1_fused_0 in range(T.int64(8)):
                for i0_i1_i2_1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem_cache_write"):
                        v_i1_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) // T.int64(32))
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32))
                        T.where(v_i2_o * T.int64(32) + (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32) < n)
                        T.reads(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i] = T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i]
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_expsum_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(m + T.int64(127)) // T.int64(128) * T.int64(128)], T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T.writes(T_softmax_expsum[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_expsum_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (m + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((m + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T.writes(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float32(0)
                        T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] + T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < m, T.exp(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i] - T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i]), T.float32(0))
            for i0_i1_i2_1_fused_0 in range(T.int64(8)):
                for i0_i1_i2_1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum_cache_write"):
                        v_i1_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) // T.int64(32))
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32))
                        T.where(v_i2_o * T.int64(32) + (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32) < n)
                        T.reads(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i]
    for i0_i1_i2_fused_i3_fused_0 in T.thread_binding((n * T.int64(32) * m + T.int64(255)) // T.int64(256), thread="blockIdx.x"):
        for i0_i1_i2_fused_i3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("T_softmax_norm"):
                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_i1 = T.axis.spatial(T.int64(32), (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // m // n)
                v_i2 = T.axis.spatial(n, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // m % n)
                v_i3 = T.axis.spatial(m, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) % m)
                T.where(i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1 < n * T.int64(32) * m)
                T.reads(T_softmax_expsum[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]) / T_softmax_expsum[v_i0, v_i1, v_i2]



def _get_dict():
    # tvm.ir.assert_structural_equal(MOD["softmax"], softmax_mxn_before)
    func_dict = {
        # softmax_mxn_before: softmax_mxn_after,
    }
    for name, func in [
        # fmt: off
        ("fused_layer_norm1_cast8", fused_layer_norm1_cast8),
        ("fused_NT_matmul1_add4_add5", fused_NT_matmul1_add4_add5),
        ("fused_NT_matmul2_divide1_maximum1_minimum1_cast9", fused_NT_matmul2_divide1_maximum1_minimum1_cast9),
        ("fused_NT_matmul4_add7_cast8_cast12_add5", fused_NT_matmul4_add7_cast8_cast12_add5),
        ("fused_NT_matmul3_add6_gelu1_cast11", fused_NT_matmul3_add6_gelu1_cast11),
        ("fused_NT_matmul_divide_maximum_minimum_cast2", fused_NT_matmul_divide_maximum_minimum_cast2),
        ("matmul3", matmul3),
        ("fused_NT_matmul1_add4", fused_NT_matmul1_add4),
        ("matmul9", matmul9),
        ("layer_norm1", layer_norm1),
        ("fused_NT_matmul4_add7_cast8_cast12_add5_cast7", fused_NT_matmul4_add7_cast8_cast12_add5_cast7),
        ("fused_min_max_triu_te_broadcast_to", fused_min_max_triu_te_broadcast_to),
        ("fused_softmax_cast3", softmax_1xn),
        # fmt: on
    ]:
        # print(f"############### {name} ###############")
        sch = tir.Schedule(MOD[name])
        func(sch)
        # sch.mod["main"].show(black_format=False)
        func_dict[MOD[name]] = sch.mod["main"]
    return {
        (tvm.ir.structural_hash(k), k): v.with_attr("tir.is_scheduled", True)
        for k, v in func_dict.items()
    }


DICT = _get_dict()


def lookup(func):
    for (hash_value, func_before), f_after in DICT.items():
        if tvm.ir.structural_hash(func) == hash_value and tvm.ir.structural_equal(
            func, func_before
        ):
            return f_after
    return None
