import tvm
from tvm import IRModule
from tvm.script import tir as T


@T.prim_func
def fused_decode4_matmul3(
    lv1587: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv1588: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    lv1583: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1587[v_i // T.int64(8), v_j], lv1588[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1587[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1588[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1583[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1583[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )



def sch_fused_decode4_matmul3(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[32, 64, 2]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[128, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l41, l42, l43, l44, l45 = sch.get_loops(block=b22)
    l46, l47, l48 = sch.split(loop=l45, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l48)
    sch.bind(loop=l47, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode6_fused_matmul7_add1(
    lv1623: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv1624: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    lv200: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    lv198: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1623[v_i // T.int64(8), v_j], lv1624[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1623[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1624[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv200[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv200[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv198[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv198[v_ax0, v_ax1, v_ax2]
                + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )



def sch_fused_decode6_fused_matmul7_add1(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[8, 256, 2]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[344, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21
    )
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    sch.reverse_compute_inline(block=b28)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 256, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode5_fused_matmul6_multiply1(
    lv1617: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv1618: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    lv1622: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1617[v_i // T.int64(8), v_j], lv1618[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1617[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1618[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1622[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1622[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv4[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv4[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )



def sch_fused_decode5_fused_matmul6_multiply1(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[43, 64, 4]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[128, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    b41 = sch.get_block(name="T_multiply", func_name="main")
    sch.reverse_compute_inline(block=b41)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l42, l43, l44, l45, l46 = sch.get_loops(block=b22)
    l47, l48, l49 = sch.split(loop=l46, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l49)
    sch.bind(loop=l48, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode5_fused_matmul6_silu1(
    lv1611: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv1612: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    lv1622: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1611[v_i // T.int64(8), v_j], lv1612[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1611[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1612[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1622[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1622[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(
                var_matmul_intermediate[v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2],
                compute[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
                * compute[v_ax0, v_ax1, v_ax2]
            )



def sch_fused_decode5_fused_matmul6_silu1(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[43, 64, 4]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[128, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    b41 = sch.get_block(name="compute", func_name="main")
    sch.compute_inline(block=b41)
    b42 = sch.get_block(name="T_multiply", func_name="main")
    sch.reverse_compute_inline(block=b42)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l43, l44, l45, l46, l47 = sch.get_loops(block=b22)
    l48, l49, l50 = sch.split(loop=l47, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l50)
    sch.bind(loop=l49, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode4_fused_matmul4_add1(
    lv1605: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv1606: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    lv197: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv1581: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1605[v_i // T.int64(8), v_j], lv1606[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1605[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1606[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv197[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv197[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv1581[v_ax0, v_ax1, v_ax2],
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv1581[v_ax0, v_ax1, v_ax2]
                + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )



def sch_fused_decode4_fused_matmul4_add1(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[32, 64, 2]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[128, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    b41 = sch.get_block(name="T_add", func_name="main")
    sch.reverse_compute_inline(block=b41)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l42, l43, l44, l45, l46 = sch.get_loops(block=b22)
    l47, l48, l49 = sch.split(loop=l46, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l49)
    sch.bind(loop=l48, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode3_fused_matmul1_cast2(
    lv1576: T.Buffer((T.int64(512), T.int64(32000)), "uint32"),
    lv1577: T.Buffer((T.int64(128), T.int64(32000)), "float16"),
    lv1575: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1576[v_i // T.int64(8), v_j], lv1577[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1576[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1577[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1575[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1575[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_matmul_intermediate[v_i0, v_i1, v_i2]
            )



def sch_fused_decode3_fused_matmul1_cast2(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[80, 100, 4]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[512, 8, 1]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    b41 = sch.get_block(name="compute", func_name="main")
    sch.reverse_compute_inline(block=b41)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l42, l43, l44, l45, l46 = sch.get_loops(block=b22)
    l47, l48, l49 = sch.split(
        loop=l46, factors=[None, 100, 2], preserve_unit_iters=True
    )
    sch.vectorize(loop=l49)
    sch.bind(loop=l48, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode2_fused_NT_matmul3_add(
    lv50: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv51: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    p_lv5: T.handle,
    p_lv3: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv5 = T.match_buffer(p_lv5, (T.int64(1), n, T.int64(11008)), "float16")
    lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(4096), T.int64(11008)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv50[v_i // T.int64(8), v_j], lv51[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv50[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv51[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv5[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv5[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv3[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv3[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )



@T.prim_func
def fused_decode2_fused_NT_matmul3_add_after(
    lv50: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv51: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    p_lv5: T.handle,
    p_lv3: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv5 = T.match_buffer(p_lv5, (T.int64(1), n, T.int64(11008)), "float16")
    lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    decode_local = T.alloc_buffer(
        (T.int64(11008), T.int64(4096)), "float16", scope="local"
    )
    lv50_local = T.alloc_buffer((T.int64(1376), T.int64(4096)), "uint32", scope="local")
    lv51_local = T.alloc_buffer((T.int64(344), T.int64(4096)), "float16", scope="local")
    lv5_pad_local = T.alloc_buffer(
        (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(11008)),
        "float16",
        scope="local",
    )
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
        (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)),
        "float16",
        scope="local",
    )
    for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
        (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
    ):
        for i2_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                    for i0_i1_fused_1_2_init in range(T.int64(4)):
                        for i2_2_init in T.vectorized(T.int64(8)):
                            with T.block("NT_matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(
                                    (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32)
                                    + i0_i1_fused_1_1 * T.int64(4)
                                    + i0_i1_fused_1_2_init,
                                )
                                v_i2 = T.axis.spatial(
                                    T.int64(4096),
                                    i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init,
                                )
                                T.reads()
                                T.writes(
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ]
                                )
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ] = T.float16(0)
                    for k_0 in range(T.int64(344)):
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(8)):
                                with T.block("lv51_local"):
                                    v0 = T.axis.spatial(T.int64(344), k_0 + ax0)
                                    v1 = T.axis.spatial(
                                        T.int64(4096),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1,
                                    )
                                    T.reads(lv51[v0, v1])
                                    T.writes(lv51_local[v0, v1])
                                    lv51_local[v0, v1] = lv51[v0, v1]
                        for k_1 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv50_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(1376), k_0 * T.int64(4) + k_1 + ax0
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv50[v0, v1])
                                        T.writes(lv50_local[v0, v1])
                                        lv50_local[v0, v1] = lv50[v0, v1]
                            for k_2 in range(T.int64(8)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("decode"):
                                            v_i = T.axis.spatial(
                                                T.int64(11008),
                                                k_0 * T.int64(32)
                                                + k_1 * T.int64(8)
                                                + k_2
                                                + ax0,
                                            )
                                            v_j = T.axis.spatial(
                                                T.int64(4096),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(
                                                lv50_local[v_i // T.int64(8), v_j],
                                                lv51_local[v_i // T.int64(32), v_j],
                                            )
                                            T.writes(decode_local[v_i, v_j])
                                            decode_local[v_i, v_j] = (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv50_local[
                                                                v_i // T.int64(8), v_j
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_i % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            ) * lv51_local[v_i // T.int64(32), v_j]
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                    for ax2 in T.vectorized(T.int64(1)):
                                        with T.block("lv5_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(
                                                (n + T.int64(31))
                                                // T.int64(32)
                                                * T.int64(32),
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused
                                                * T.int64(32)
                                                + i0_i1_fused_1_1 * T.int64(4)
                                                + ax1,
                                            )
                                            v2 = T.axis.spatial(
                                                T.int64(11008),
                                                k_0 * T.int64(32)
                                                + k_1 * T.int64(8)
                                                + k_2
                                                + ax2,
                                            )
                                            T.reads(lv5[v0, v1, v2])
                                            T.writes(lv5_pad_local[v0, v1, v2])
                                            lv5_pad_local[v0, v1, v2] = T.if_then_else(
                                                v1 < n, lv5[v0, v1, v2], T.float16(0)
                                            )
                                for i0_i1_fused_1_2 in range(T.int64(4)):
                                    for i2_2 in T.vectorized(T.int64(8)):
                                        with T.block("NT_matmul_update"):
                                            v_i0 = T.axis.spatial(
                                                T.int64(1), T.int64(0)
                                            )
                                            v_i1 = T.axis.spatial(
                                                (n + T.int64(31))
                                                // T.int64(32)
                                                * T.int64(32),
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused
                                                * T.int64(32)
                                                + i0_i1_fused_1_1 * T.int64(4)
                                                + i0_i1_fused_1_2,
                                            )
                                            v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + i2_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(11008),
                                                k_0 * T.int64(32)
                                                + k_1 * T.int64(8)
                                                + k_2,
                                            )
                                            T.reads(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv5_pad_local[v_i0, v_i1, v_k],
                                                decode_local[v_k, v_i2],
                                            )
                                            T.writes(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ] = (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                                + lv5_pad_local[v_i0, v_i1, v_k]
                                                * decode_local[v_k, v_i2]
                                            )
                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                        for ax2 in T.vectorized(T.int64(8)):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(
                                    (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32)
                                    + i0_i1_fused_1_1 * T.int64(4)
                                    + ax1,
                                )
                                v2 = T.axis.spatial(
                                    T.int64(4096),
                                    i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                )
                                T.reads(
                                    lv3[v0, v1, v2],
                                    var_NT_matmul_intermediate_pad_local[v0, v1, v2],
                                )
                                T.writes(p_output0_intermediate[v0, v1, v2])
                                if v1 < n:
                                    p_output0_intermediate[v0, v1, v2] = (
                                        lv3[v0, v1, v2]
                                        + var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ]
                                    )


@T.prim_func
def fused_decode_NT_matmul(
    lv8: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv9: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    p_lv6: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(4096), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv8[v_i // T.int64(8), v_j], lv9[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv8[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv9[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv6[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv6[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )


@T.prim_func
def fused_decode_NT_matmul_after(
    lv8: T.Buffer((512, 4096), "uint32"),
    lv9: T.Buffer((128, 4096), "float16"),
    p_lv6: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int32()
    lv6 = T.match_buffer(p_lv6, (1, n, 4096), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (1, n, 4096), "float16")
    # with T.block("root"):
    decode_local = T.alloc_buffer((4096, 4096), "float16", scope="local")
    lv8_local = T.alloc_buffer((512, 4096), "uint32", scope="local")
    lv9_local = T.alloc_buffer((128, 4096), "float16", scope="local")
    lv6_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
        (n + 31) // 32, thread="blockIdx.y"
    ):
        for i2_0 in T.thread_binding(32, thread="blockIdx.x"):
            for i0_i1_fused_1_1 in T.thread_binding(8, thread="threadIdx.y"):
                for i2_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i0_i1_fused_1_2_init in range(4):
                        for i2_2_init in T.vectorized(8):
                            with T.block("NT_matmul_init"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + i0_i1_fused_1_2_init,
                                )
                                v_i2 = T.axis.spatial(
                                    4096, i2_0 * 128 + i2_1 * 8 + i2_2_init
                                )
                                T.reads()
                                T.writes(
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ]
                                )
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ] = T.float16(0)
                    for k_0 in range(128):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("lv9_local"):
                                    v0 = T.axis.spatial(128, k_0 + ax0)
                                    v1 = T.axis.spatial(
                                        4096, i2_0 * 128 + i2_1 * 8 + ax1
                                    )
                                    T.reads(lv9[v0, v1])
                                    T.writes(lv9_local[v0, v1])
                                    lv9_local[v0, v1] = lv9[v0, v1]
                        for k_1 in range(4):
                            for ax0 in range(1):
                                for ax1 in T.vectorized(8):
                                    with T.block("lv8_local"):
                                        v0 = T.axis.spatial(512, k_0 * 4 + k_1 + ax0)
                                        v1 = T.axis.spatial(
                                            4096, i2_0 * 128 + i2_1 * 8 + ax1
                                        )
                                        T.reads(lv8[v0, v1])
                                        T.writes(lv8_local[v0, v1])
                                        lv8_local[v0, v1] = lv8[v0, v1]
                            for k_2 in range(8):
                                for ax0 in range(1):
                                    for ax1 in T.vectorized(8):
                                        with T.block("decode"):
                                            v_i = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax0
                                            )
                                            v_j = T.axis.spatial(
                                                4096, i2_0 * 128 + i2_1 * 8 + ax1
                                            )
                                            T.reads(
                                                lv8_local[v_i // 8, v_j],
                                                lv9_local[v_i // 32, v_j],
                                            )
                                            T.writes(decode_local[v_i, v_j])
                                            decode_local[v_i, v_j] = (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv8_local[v_i // 8, v_j],
                                                            T.Cast("uint32", v_i % 8)
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            ) * lv9_local[v_i // 32, v_j]
                                for ax0, ax1 in T.grid(1, 4):
                                    for ax2 in T.vectorized(1):
                                        with T.block("lv6_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + ax1,
                                            )
                                            v2 = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax2
                                            )
                                            T.reads(lv6[v0, v1, v2])
                                            T.writes(lv6_pad_local[v0, v1, v2])
                                            lv6_pad_local[v0, v1, v2] = T.if_then_else(
                                                v1 < n, lv6[v0, v1, v2], T.float16(0)
                                            )
                                for i0_i1_fused_1_2 in range(4):
                                    for i2_2 in T.vectorized(8):
                                        with T.block("NT_matmul_update"):
                                            v_i0 = T.axis.spatial(1, 0)
                                            v_i1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + i0_i1_fused_1_2,
                                            )
                                            v_i2 = T.axis.spatial(
                                                4096, i2_0 * 128 + i2_1 * 8 + i2_2
                                            )
                                            v_k = T.axis.reduce(
                                                4096, k_0 * 32 + k_1 * 8 + k_2
                                            )
                                            T.reads(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv6_pad_local[v_i0, v_i1, v_k],
                                                decode_local[v_k, v_i2],
                                            )
                                            T.writes(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ] = (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                                + lv6_pad_local[v_i0, v_i1, v_k]
                                                * decode_local[v_k, v_i2]
                                            )
                    for ax0, ax1 in T.grid(1, 4):
                        for ax2 in T.vectorized(8):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + ax1,
                                )
                                v2 = T.axis.spatial(4096, i2_0 * 128 + i2_1 * 8 + ax2)
                                T.reads(
                                    var_NT_matmul_intermediate_pad_local[v0, v1, v2]
                                )
                                T.writes(var_NT_matmul_intermediate[v0, v1, v2])
                                if v1 < n:
                                    var_NT_matmul_intermediate[
                                        v0, v1, v2
                                    ] = var_NT_matmul_intermediate_pad_local[v0, v1, v2]


@T.prim_func
def fused_decode1_fused_NT_matmul2_silu(
    lv36: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv37: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    p_lv45: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv45 = T.match_buffer(p_lv45, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(11008)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(11008), T.int64(4096)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(11008)), "float16"
    )
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv36[v_i // T.int64(8), v_j], lv37[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv36[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv37[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv45[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv45[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
                compute[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
                * compute[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_decode1_fused_NT_matmul2_silu_after(
    lv36: T.Buffer((512, 11008), "uint32"),
    lv37: T.Buffer((128, 11008), "float16"),
    p_lv45: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int32()
    lv45 = T.match_buffer(p_lv45, (1, n, 4096), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (1, n, 11008), "float16")
    # with T.block("root"):
    decode_local = T.alloc_buffer((4096, 11008), "float16", scope="local")
    lv36_local = T.alloc_buffer((512, 11008), "uint32", scope="local")
    lv37_local = T.alloc_buffer((128, 11008), "float16", scope="local")
    lv45_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 11008), "float16", scope="local"
    )
    for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
        (n + 31) // 32, thread="blockIdx.y"
    ):
        for i2_0 in T.thread_binding(86, thread="blockIdx.x"):
            for i0_i1_fused_1_1 in T.thread_binding(8, thread="threadIdx.y"):
                for i2_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i0_i1_fused_1_2_init in range(4):
                        for i2_2_init in T.vectorized(8):
                            with T.block("NT_matmul_init"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + i0_i1_fused_1_2_init,
                                )
                                v_i2 = T.axis.spatial(
                                    11008, i2_0 * 128 + i2_1 * 8 + i2_2_init
                                )
                                T.reads()
                                T.writes(
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ]
                                )
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ] = T.float16(0)
                    for k_0 in range(128):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("lv37_local"):
                                    v0 = T.axis.spatial(128, k_0 + ax0)
                                    v1 = T.axis.spatial(
                                        11008, i2_0 * 128 + i2_1 * 8 + ax1
                                    )
                                    T.reads(lv37[v0, v1])
                                    T.writes(lv37_local[v0, v1])
                                    lv37_local[v0, v1] = lv37[v0, v1]
                        for k_1 in range(4):
                            for ax0 in range(1):
                                for ax1 in T.vectorized(8):
                                    with T.block("lv36_local"):
                                        v0 = T.axis.spatial(512, k_0 * 4 + k_1 + ax0)
                                        v1 = T.axis.spatial(
                                            11008, i2_0 * 128 + i2_1 * 8 + ax1
                                        )
                                        T.reads(lv36[v0, v1])
                                        T.writes(lv36_local[v0, v1])
                                        lv36_local[v0, v1] = lv36[v0, v1]
                            for k_2 in range(8):
                                for ax0 in range(1):
                                    for ax1 in T.vectorized(8):
                                        with T.block("decode"):
                                            v_i = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax0
                                            )
                                            v_j = T.axis.spatial(
                                                11008, i2_0 * 128 + i2_1 * 8 + ax1
                                            )
                                            T.reads(
                                                lv36_local[v_i // 8, v_j],
                                                lv37_local[v_i // 32, v_j],
                                            )
                                            T.writes(decode_local[v_i, v_j])
                                            decode_local[v_i, v_j] = (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv36_local[v_i // 8, v_j],
                                                            T.Cast("uint32", v_i % 8)
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            ) * lv37_local[v_i // 32, v_j]
                                for ax0, ax1 in T.grid(1, 4):
                                    for ax2 in T.vectorized(1):
                                        with T.block("lv45_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + ax1,
                                            )
                                            v2 = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax2
                                            )
                                            T.reads(lv45[v0, v1, v2])
                                            T.writes(lv45_pad_local[v0, v1, v2])
                                            lv45_pad_local[v0, v1, v2] = T.if_then_else(
                                                v1 < n, lv45[v0, v1, v2], T.float16(0)
                                            )
                                for i0_i1_fused_1_2 in range(4):
                                    for i2_2 in T.vectorized(8):
                                        with T.block("NT_matmul_update"):
                                            v_i0 = T.axis.spatial(1, 0)
                                            v_i1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + i0_i1_fused_1_2,
                                            )
                                            v_i2 = T.axis.spatial(
                                                11008, i2_0 * 128 + i2_1 * 8 + i2_2
                                            )
                                            v_k = T.axis.reduce(
                                                4096, k_0 * 32 + k_1 * 8 + k_2
                                            )
                                            T.reads(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv45_pad_local[v_i0, v_i1, v_k],
                                                decode_local[v_k, v_i2],
                                            )
                                            T.writes(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ] = (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                                + lv45_pad_local[v_i0, v_i1, v_k]
                                                * decode_local[v_k, v_i2]
                                            )
                    for ax0, ax1 in T.grid(1, 4):
                        for ax2 in T.vectorized(8):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + ax1,
                                )
                                v2 = T.axis.spatial(11008, i2_0 * 128 + i2_1 * 8 + ax2)
                                T.reads(
                                    var_NT_matmul_intermediate_pad_local[v0, v1, v2]
                                )
                                T.writes(p_output0_intermediate[v0, v1, v2])
                                if v1 < n:
                                    p_output0_intermediate[
                                        v0, v1, v2
                                    ] = var_NT_matmul_intermediate_pad_local[
                                        v0, v1, v2
                                    ] * T.sigmoid(
                                        var_NT_matmul_intermediate_pad_local[v0, v1, v2]
                                    )


@T.prim_func
def fused_decode1_fused_NT_matmul2_multiply(
    lv43: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv44: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    p_lv45: T.handle,
    p_lv132: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv45 = T.match_buffer(p_lv45, (T.int64(1), n, T.int64(4096)), "float16")
    lv132 = T.match_buffer(p_lv132, (T.int64(1), n, T.int64(11008)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(11008)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(11008), T.int64(4096)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(11008)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv43[v_i // T.int64(8), v_j], lv44[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv43[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv44[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv45[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv45[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv132[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv132[v_ax0, v_ax1, v_ax2]
                * var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_decode1_fused_NT_matmul2_multiply_after(lv43: T.Buffer((512, 11008), "uint32"), lv44: T.Buffer((128, 11008), "float16"), p_lv45: T.handle, p_lv132: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int32()
    lv45 = T.match_buffer(p_lv45, (1, n, 4096), "float16")
    lv132 = T.match_buffer(p_lv132, (1, n, 11008), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (1, n, 11008), "float16")
    # with T.block("root"):
    decode_local = T.alloc_buffer((4096, 11008), "float16", scope="local")
    lv43_local = T.alloc_buffer((512, 11008), "uint32", scope="local")
    lv44_local = T.alloc_buffer((128, 11008), "float16", scope="local")
    lv45_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 11008), "float16", scope="local"
    )
    for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
        (n + 31) // 32, thread="blockIdx.y"
    ):
        for i2_0 in T.thread_binding(86, thread="blockIdx.x"):
            for i0_i1_fused_1_1 in T.thread_binding(8, thread="threadIdx.y"):
                for i2_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i0_i1_fused_1_2_init in range(4):
                        for i2_2_init in T.vectorized(8):
                            with T.block("NT_matmul_init"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + i0_i1_fused_1_2_init,
                                )
                                v_i2 = T.axis.spatial(
                                    11008, i2_0 * 128 + i2_1 * 8 + i2_2_init
                                )
                                T.reads()
                                T.writes(
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ]
                                )
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ] = T.float16(0)
                    for k_0 in range(128):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("lv44_local"):
                                    v0 = T.axis.spatial(128, k_0 + ax0)
                                    v1 = T.axis.spatial(
                                        11008, i2_0 * 128 + i2_1 * 8 + ax1
                                    )
                                    T.reads(lv44[v0, v1])
                                    T.writes(lv44_local[v0, v1])
                                    lv44_local[v0, v1] = lv44[v0, v1]
                        for k_1 in range(4):
                            for ax0 in range(1):
                                for ax1 in T.vectorized(8):
                                    with T.block("lv43_local"):
                                        v0 = T.axis.spatial(512, k_0 * 4 + k_1 + ax0)
                                        v1 = T.axis.spatial(
                                            11008, i2_0 * 128 + i2_1 * 8 + ax1
                                        )
                                        T.reads(lv43[v0, v1])
                                        T.writes(lv43_local[v0, v1])
                                        lv43_local[v0, v1] = lv43[v0, v1]
                            for k_2 in range(8):
                                for ax0 in range(1):
                                    for ax1 in T.vectorized(8):
                                        with T.block("decode"):
                                            v_i = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax0
                                            )
                                            v_j = T.axis.spatial(
                                                11008, i2_0 * 128 + i2_1 * 8 + ax1
                                            )
                                            T.reads(
                                                lv43_local[v_i // 8, v_j],
                                                lv44_local[v_i // 32, v_j],
                                            )
                                            T.writes(decode_local[v_i, v_j])
                                            decode_local[v_i, v_j] = (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv43_local[v_i // 8, v_j],
                                                            T.Cast("uint32", v_i % 8)
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            ) * lv44_local[v_i // 32, v_j]
                                for ax0, ax1 in T.grid(1, 4):
                                    for ax2 in T.vectorized(1):
                                        with T.block("lv45_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + ax1,
                                            )
                                            v2 = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax2
                                            )
                                            T.reads(lv45[v0, v1, v2])
                                            T.writes(lv45_pad_local[v0, v1, v2])
                                            lv45_pad_local[v0, v1, v2] = T.if_then_else(
                                                v1 < n, lv45[v0, v1, v2], T.float16(0)
                                            )
                                for i0_i1_fused_1_2 in range(4):
                                    for i2_2 in T.vectorized(8):
                                        with T.block("NT_matmul_update"):
                                            v_i0 = T.axis.spatial(1, 0)
                                            v_i1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + i0_i1_fused_1_2,
                                            )
                                            v_i2 = T.axis.spatial(
                                                11008, i2_0 * 128 + i2_1 * 8 + i2_2
                                            )
                                            v_k = T.axis.reduce(
                                                4096, k_0 * 32 + k_1 * 8 + k_2
                                            )
                                            T.reads(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv45_pad_local[v_i0, v_i1, v_k],
                                                decode_local[v_k, v_i2],
                                            )
                                            T.writes(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ] = (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                                + lv45_pad_local[v_i0, v_i1, v_k]
                                                * decode_local[v_k, v_i2]
                                            )
                    for ax0, ax1 in T.grid(1, 4):
                        for ax2 in T.vectorized(8):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + ax1,
                                )
                                v2 = T.axis.spatial(11008, i2_0 * 128 + i2_1 * 8 + ax2)
                                T.reads(
                                    lv132[v0, v1, v2],
                                    var_NT_matmul_intermediate_pad_local[v0, v1, v2],
                                )
                                T.writes(p_output0_intermediate[v0, v1, v2])
                                if v1 < n:
                                    p_output0_intermediate[v0, v1, v2] = (
                                        lv132[v0, v1, v2]
                                        * var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ]
                                    )


@T.prim_func
def fused_decode_fused_NT_matmul_add(
    lv29: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv30: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    p_lv41: T.handle,
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(4096), T.int64(4096)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv29[v_i // T.int64(8), v_j], lv30[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv29[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv30[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv41[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv41[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv2[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv2[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_decode_fused_NT_matmul_add_after(lv29: T.Buffer((512, 4096), "uint32"), lv30: T.Buffer((128, 4096), "float16"), p_lv41: T.handle, p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int32()
    lv41 = T.match_buffer(p_lv41, (1, n, 4096), "float16")
    lv2 = T.match_buffer(p_lv2, (1, n, 4096), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (1, n, 4096), "float16")
    # with T.block("root"):
    decode_local = T.alloc_buffer((4096, 4096), "float16", scope="local")
    lv29_local = T.alloc_buffer((512, 4096), "uint32", scope="local")
    lv30_local = T.alloc_buffer((128, 4096), "float16", scope="local")
    lv41_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
        (1, (n + 31) // 32 * 32, 4096), "float16", scope="local"
    )
    for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
        (n + 31) // 32, thread="blockIdx.y"
    ):
        for i2_0 in T.thread_binding(32, thread="blockIdx.x"):
            for i0_i1_fused_1_1 in T.thread_binding(8, thread="threadIdx.y"):
                for i2_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i0_i1_fused_1_2_init in range(4):
                        for i2_2_init in T.vectorized(8):
                            with T.block("NT_matmul_init"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + i0_i1_fused_1_2_init,
                                )
                                v_i2 = T.axis.spatial(
                                    4096, i2_0 * 128 + i2_1 * 8 + i2_2_init
                                )
                                T.reads()
                                T.writes(
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ]
                                )
                                var_NT_matmul_intermediate_pad_local[
                                    v_i0, v_i1, v_i2
                                ] = T.float16(0)
                    for k_0 in range(128):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("lv30_local"):
                                    v0 = T.axis.spatial(128, k_0 + ax0)
                                    v1 = T.axis.spatial(
                                        4096, i2_0 * 128 + i2_1 * 8 + ax1
                                    )
                                    T.reads(lv30[v0, v1])
                                    T.writes(lv30_local[v0, v1])
                                    lv30_local[v0, v1] = lv30[v0, v1]
                        for k_1 in range(4):
                            for ax0 in range(1):
                                for ax1 in T.vectorized(8):
                                    with T.block("lv29_local"):
                                        v0 = T.axis.spatial(512, k_0 * 4 + k_1 + ax0)
                                        v1 = T.axis.spatial(
                                            4096, i2_0 * 128 + i2_1 * 8 + ax1
                                        )
                                        T.reads(lv29[v0, v1])
                                        T.writes(lv29_local[v0, v1])
                                        lv29_local[v0, v1] = lv29[v0, v1]
                            for k_2 in range(8):
                                for ax0 in range(1):
                                    for ax1 in T.vectorized(8):
                                        with T.block("decode"):
                                            v_i = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax0
                                            )
                                            v_j = T.axis.spatial(
                                                4096, i2_0 * 128 + i2_1 * 8 + ax1
                                            )
                                            T.reads(
                                                lv29_local[v_i // 8, v_j],
                                                lv30_local[v_i // 32, v_j],
                                            )
                                            T.writes(decode_local[v_i, v_j])
                                            decode_local[v_i, v_j] = (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv29_local[v_i // 8, v_j],
                                                            T.Cast("uint32", v_i % 8)
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            ) * lv30_local[v_i // 32, v_j]
                                for ax0, ax1 in T.grid(1, 4):
                                    for ax2 in T.vectorized(1):
                                        with T.block("lv41_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + ax1,
                                            )
                                            v2 = T.axis.spatial(
                                                4096, k_0 * 32 + k_1 * 8 + k_2 + ax2
                                            )
                                            T.reads(lv41[v0, v1, v2])
                                            T.writes(lv41_pad_local[v0, v1, v2])
                                            lv41_pad_local[v0, v1, v2] = T.if_then_else(
                                                v1 < n, lv41[v0, v1, v2], T.float16(0)
                                            )
                                for i0_i1_fused_1_2 in range(4):
                                    for i2_2 in T.vectorized(8):
                                        with T.block("NT_matmul_update"):
                                            v_i0 = T.axis.spatial(1, 0)
                                            v_i1 = T.axis.spatial(
                                                (n + 31) // 32 * 32,
                                                i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                                + i0_i1_fused_1_1 * 4
                                                + i0_i1_fused_1_2,
                                            )
                                            v_i2 = T.axis.spatial(
                                                4096, i2_0 * 128 + i2_1 * 8 + i2_2
                                            )
                                            v_k = T.axis.reduce(
                                                4096, k_0 * 32 + k_1 * 8 + k_2
                                            )
                                            T.reads(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv41_pad_local[v_i0, v_i1, v_k],
                                                decode_local[v_k, v_i2],
                                            )
                                            T.writes(
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_NT_matmul_intermediate_pad_local[
                                                v_i0, v_i1, v_i2
                                            ] = (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ]
                                                + lv41_pad_local[v_i0, v_i1, v_k]
                                                * decode_local[v_k, v_i2]
                                            )
                    for ax0, ax1 in T.grid(1, 4):
                        for ax2 in T.vectorized(8):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(1, ax0)
                                v1 = T.axis.spatial(
                                    (n + 31) // 32 * 32,
                                    i0_i1_fused_0_i0_i1_fused_1_0_fused * 32
                                    + i0_i1_fused_1_1 * 4
                                    + ax1,
                                )
                                v2 = T.axis.spatial(4096, i2_0 * 128 + i2_1 * 8 + ax2)
                                T.reads(
                                    lv2[v0, v1, v2],
                                    var_NT_matmul_intermediate_pad_local[v0, v1, v2],
                                )
                                T.writes(p_output0_intermediate[v0, v1, v2])
                                if v1 < n:
                                    p_output0_intermediate[v0, v1, v2] = (
                                        lv2[v0, v1, v2]
                                        + var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ]
                                    )


@T.prim_func
def fused_decode4_fused_matmul6_add4(lv1363: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv1364: T.Buffer((T.int64(80), T.int64(2560)), "float16"), lv2067: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias192: T.Buffer((T.int64(2560),), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1363[v_i // T.int64(8), v_j], lv1364[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1363[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1364[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2067[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2067[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias192[v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias192[v_ax2]


def sch_fused_decode4_fused_matmul6_add4(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[160, 8, 2])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    sch.reverse_compute_inline(block=b28)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 256, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode6_fused_matmul9_add7_cast8_cast12_add5(lv1393: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv1394: T.Buffer((T.int64(320), T.int64(2560)), "float16"), lv2121: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), linear_bias197: T.Buffer((T.int64(2560),), "float32"), lv329: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1393[v_i // T.int64(8), v_j], lv1394[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1393[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1394[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2121[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv2121[v_i0, v_i1, v_k]) * T.Cast("float32", var_decode_intermediate[v_k, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias197[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias197[v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[v_i0, v_i1, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv329[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv329[v_ax0, v_ax1, v_ax2]


def sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[640, 2, 8])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    bb1 = sch.get_block(name="compute", func_name="main")
    bb2 = sch.get_block(name="compute_1", func_name="main")
    bb3 = sch.get_block(name="T_add_1", func_name="main")
    sch.compute_inline(block=b28)
    sch.compute_inline(block=bb1)
    sch.compute_inline(block=bb2)
    sch.reverse_compute_inline(block=bb3)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 256, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode5_fused_matmul8_add6_gelu1_cast11(lv1387: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), lv1388: T.Buffer((T.int64(80), T.int64(10240)), "float16"), lv2115: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias196: T.Buffer((T.int64(10240),), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_multiply = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    for i, j in T.grid(T.int64(2560), T.int64(10240)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1387[v_i // T.int64(8), v_j], lv1388[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1387[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1388[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(10240), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2115[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv2115[v_i0, v_i1, v_k]) * T.Cast("float32", var_decode_intermediate[v_k, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias196[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias196[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_multiply_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_multiply_intermediate[v_i0, v_i1, v_i2])


def sch_fused_decode5_fused_matmul8_add6_gelu1_cast11(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 4])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[80, 4, 8])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    bb1 = sch.get_block(name="T_multiply", func_name="main")
    bb2 = sch.get_block(name="compute", func_name="main")
    bb3 = sch.get_block(name="T_multiply_1", func_name="main")
    bb4 = sch.get_block(name="T_add_1", func_name="main")
    bb5 = sch.get_block(name="T_multiply_2", func_name="main")
    bb6 = sch.get_block(name="compute_1", func_name="main")
    sch.compute_inline(block=b28)
    sch.compute_inline(block=bb1)
    sch.compute_inline(block=bb2)
    sch.compute_inline(block=bb3)
    sch.compute_inline(block=bb4)
    sch.compute_inline(block=bb5)
    sch.reverse_compute_inline(block=bb6)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 256, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode4_fused_matmul6_add4_add5(lv1381: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv1382: T.Buffer((T.int64(80), T.int64(2560)), "float16"), lv328: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias195: T.Buffer((T.int64(2560),), "float16"), lv2062: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1381[v_i // T.int64(8), v_j], lv1382[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1381[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1382[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv328[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv328[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias195[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias195[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], lv2062[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] + lv2062[v_ax0, v_ax1, v_ax2]


def sch_fused_decode4_fused_matmul6_add4_add5(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[160, 8, 2])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    bb4 = sch.get_block(name="T_add_1", func_name="main")
    sch.compute_inline(block=b28)
    sch.reverse_compute_inline(block=bb4)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 256, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode3_matmul3(lv2515: T.Buffer((T.int64(320), T.int64(50432)), "uint32"), lv2516: T.Buffer((T.int64(80), T.int64(50432)), "float32"), lv705: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(50432)))
    for i, j in T.grid(T.int64(2560), T.int64(50432)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2515[v_i // T.int64(8), v_j], lv2516[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.Cast("float16", T.bitwise_and(T.shift_right(lv2515[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2516[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50432), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv705[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv705[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


def sch_fused_decode3_matmul3(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[197, 128, 2])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[80, 4, 8])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 128, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(lv2509: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv2510: T.Buffer((T.int64(320), T.int64(2560)), "float16"), lv4105: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), linear_bias383: T.Buffer((T.int64(2560),), "float32"), lv701: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2509[v_i // T.int64(8), v_j], lv2510[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2509[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv2510[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv4105[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv4105[v_i0, v_i1, v_k]) * T.Cast("float32", var_decode_intermediate[v_k, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias383[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias383[v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[v_i0, v_i1, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv701[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv701[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_2"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2])


def sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=4, decision=[5, 256, 2])
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=8, decision=[320, 4, 8])
    l16, l17, l18 = sch.split(loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True)
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    v21 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v21)
    l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
    sch.vectorize(loop=l26)
    sch.vectorize(loop=l12)
    b27 = sch.decompose_reduction(block=b1, loop=l16)
    b28 = sch.get_block(name="T_add", func_name="main")
    bb1 = sch.get_block(name="compute", func_name="main")
    bb2 = sch.get_block(name="compute_1", func_name="main")
    bb3 = sch.get_block(name="T_add_1", func_name="main")
    bb4 = sch.get_block(name="compute_2", func_name="main")
    sch.compute_inline(block=b28)
    sch.compute_inline(block=bb1)
    sch.compute_inline(block=bb2)
    sch.compute_inline(block=bb3)
    sch.reverse_compute_inline(block=bb4)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(loop=l33, factors=[None, 256, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode2_fused_NT_matmul3_add6_gelu1_cast11(lv36: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), lv37: T.Buffer((T.int64(80), T.int64(10240)), "float16"), p_lv57: T.handle, linear_bias4: T.Buffer((T.int64(10240),), "float32"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv57 = T.match_buffer(p_lv57, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    T_multiply = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    compute = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    T_multiply_1 = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    T_add = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
    for i, j in T.grid(T.int64(2560), T.int64(10240)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv36[v_i // T.int64(8), v_j], lv37[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv36[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv37[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(10240), T.int64(2560)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv57[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv57[v_i0, v_i1, v_k]) * T.Cast("float32", var_T_transpose_intermediate[v_i2, v_k])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias4[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias4[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_multiply_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_multiply_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_decode2_fused_NT_matmul3_add6_gelu1_cast11_after(lv36: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), lv37: T.Buffer((T.int64(80), T.int64(10240)), "float16"), p_lv57: T.handle, linear_bias4: T.Buffer((T.int64(10240),), "float32"), p_output0: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv57 = T.match_buffer(p_lv57, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16", scope="local")
        lv36_local = T.alloc_buffer((T.int64(320), T.int64(10240)), "uint32", scope="local")
        lv37_local = T.alloc_buffer((T.int64(80), T.int64(10240)), "float16", scope="local")
        lv57_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "float16", scope="local")
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(10240)), scope="local")
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
            for i2_0 in T.thread_binding(T.int64(80), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2_init)
                                    v_i2 = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init)
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv37_local"):
                                        v0 = T.axis.spatial(T.int64(80), k_0_0 * T.int64(4) + k_0_1 + ax0)
                                        v1 = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv37[v0, v1])
                                        T.writes(lv37_local[v0, v1])
                                        lv37_local[v0, v1] = lv37[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv36_local"):
                                            v0 = T.axis.spatial(T.int64(320), k_0_0 * T.int64(16) + k_0_1 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv36[v0, v1])
                                            T.writes(lv36_local[v0, v1])
                                            lv36_local[v0, v1] = lv36[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax0)
                                                v_j = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                                T.reads(lv36_local[v_i // T.int64(8), v_j], lv37_local[v_i // T.int64(32), v_j])
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv36_local[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv37_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv57_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                T.reads(lv57[v0, v1, v2])
                                                T.writes(lv57_pad_local[v0, v1, v2])
                                                lv57_pad_local[v0, v1, v2] = T.if_then_else(v1 < n, lv57[v0, v1, v2], T.float16(0))
                                    for i0_i1_fused_1_2 in range(T.int64(4)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("NT_matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2)
                                                v_i2 = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv57_pad_local[v_i0, v_i1, v_k], decode_local[v_k, v_i2])
                                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + T.Cast("float32", lv57_pad_local[v_i0, v_i1, v_k]) * T.Cast("float32", decode_local[v_k, v_i2])
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                    v2 = T.axis.spatial(T.int64(10240), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2], linear_bias4[v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = T.Cast("float16", (var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias4[v2]) * (T.float32(0.5) + T.erf((var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias4[v2]) * T.float32(0.70710678118654757)) * T.float32(0.5)))


@T.prim_func
def fused_decode1_fused_NT_matmul1_add4(lv8: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv9: T.Buffer((T.int64(80), T.int64(2560)), "float16"), p_lv9: T.handle, linear_bias: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv9_1 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv8[v_i // T.int64(8), v_j], lv9[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv8[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv9[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv9_1[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv9_1[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias[v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias[v_ax2]


@T.prim_func
def fused_decode1_fused_NT_matmul1_add4_after(lv8: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv9: T.Buffer((T.int64(80), T.int64(2560)), "float16"), p_lv9: T.handle, linear_bias: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv9_1 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16", scope="local")
        lv8_local = T.alloc_buffer((T.int64(320), T.int64(2560)), "uint32", scope="local")
        lv9_local = T.alloc_buffer((T.int64(80), T.int64(2560)), "float16", scope="local")
        lv9_1_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "float16", scope="local")
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "float16", scope="local")
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2_init)
                                    v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init)
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(T.int64(80), k_0_0 * T.int64(4) + k_0_1 + ax0)
                                        v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(T.int64(320), k_0_0 * T.int64(16) + k_0_1 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax0)
                                                v_j = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                                T.reads(lv8_local[v_i // T.int64(8), v_j], lv9_local[v_i // T.int64(32), v_j])
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv8_local[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv9_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv9_1_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                T.reads(lv9_1[v0, v1, v2])
                                                T.writes(lv9_1_pad_local[v0, v1, v2])
                                                lv9_1_pad_local[v0, v1, v2] = T.if_then_else(v1 < n, lv9_1[v0, v1, v2], T.float16(0))
                                    for i0_i1_fused_1_2 in range(T.int64(4)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("NT_matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2)
                                                v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv9_1_pad_local[v_i0, v_i1, v_k], decode_local[v_k, v_i2])
                                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv9_1_pad_local[v_i0, v_i1, v_k] * decode_local[v_k, v_i2]
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                    v2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2], linear_bias[v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias[v2]


@T.prim_func
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5(lv43: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv44: T.Buffer((T.int64(320), T.int64(2560)), "float16"), p_lv63: T.handle, linear_bias5: T.Buffer((T.int64(2560),), "float32"), p_lv7: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv63 = T.match_buffer(p_lv63, (T.int64(1), n, T.int64(10240)), "float16")
    lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv43[v_i // T.int64(8), v_j], lv44[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv43[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv44[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(2560), T.int64(10240)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv63[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv63[v_i0, v_i1, v_k]) * T.Cast("float32", var_T_transpose_intermediate[v_i2, v_k])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias5[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias5[v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[v_i0, v_i1, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv7[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv7[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_after(lv43: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv44: T.Buffer((T.int64(320), T.int64(2560)), "float16"), p_lv63: T.handle, linear_bias5: T.Buffer((T.int64(2560),), "float32"), p_lv7: T.handle, p_output0: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv63 = T.match_buffer(p_lv63, (T.int64(1), n, T.int64(10240)), "float16")
    lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16", scope="local")
        lv43_local = T.alloc_buffer((T.int64(1280), T.int64(2560)), "uint32", scope="local")
        lv44_local = T.alloc_buffer((T.int64(320), T.int64(2560)), "float16", scope="local")
        lv63_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(10240)), "float16", scope="local")
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), scope="local")
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2_init)
                                    v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init)
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(80), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv44_local"):
                                        v0 = T.axis.spatial(T.int64(320), k_0_0 * T.int64(4) + k_0_1 + ax0)
                                        v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv44[v0, v1])
                                        T.writes(lv44_local[v0, v1])
                                        lv44_local[v0, v1] = lv44[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv43_local"):
                                            v0 = T.axis.spatial(T.int64(1280), k_0_0 * T.int64(16) + k_0_1 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv43[v0, v1])
                                            T.writes(lv43_local[v0, v1])
                                            lv43_local[v0, v1] = lv43[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax0)
                                                v_j = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                                T.reads(lv43_local[v_i // T.int64(8), v_j], lv44_local[v_i // T.int64(32), v_j])
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv43_local[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv44_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv63_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                T.reads(lv63[v0, v1, v2])
                                                T.writes(lv63_pad_local[v0, v1, v2])
                                                lv63_pad_local[v0, v1, v2] = T.if_then_else(v1 < n, lv63[v0, v1, v2], T.float16(0))
                                    for i0_i1_fused_1_2 in range(T.int64(4)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("NT_matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2)
                                                v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv63_pad_local[v_i0, v_i1, v_k], decode_local[v_k, v_i2])
                                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + T.Cast("float32", lv63_pad_local[v_i0, v_i1, v_k]) * T.Cast("float32", decode_local[v_k, v_i2])
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                    v2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2], linear_bias5[v2], lv7[v0, v1, v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = T.Cast("float16", var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias5[v2]) + lv7[v0, v1, v2]


@T.prim_func
def fused_decode1_fused_NT_matmul1_add4_add5(lv29: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv30: T.Buffer((T.int64(80), T.int64(2560)), "float16"), p_lv49: T.handle, linear_bias3: T.Buffer((T.int64(2560),), "float16"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv29[v_i // T.int64(8), v_j], lv30[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv29[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv30[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv49[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv49[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias3[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias3[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], lv2[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] + lv2[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode1_fused_NT_matmul1_add4_add5_after(lv29: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv30: T.Buffer((T.int64(80), T.int64(2560)), "float16"), p_lv49: T.handle, linear_bias3: T.Buffer((T.int64(2560),), "float16"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16", scope="local")
        lv29_local = T.alloc_buffer((T.int64(320), T.int64(2560)), "uint32", scope="local")
        lv30_local = T.alloc_buffer((T.int64(80), T.int64(2560)), "float16", scope="local")
        lv49_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "float16", scope="local")
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), "float16", scope="local")
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2_init)
                                    v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init)
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv30_local"):
                                        v0 = T.axis.spatial(T.int64(80), k_0_0 * T.int64(4) + k_0_1 + ax0)
                                        v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv30[v0, v1])
                                        T.writes(lv30_local[v0, v1])
                                        lv30_local[v0, v1] = lv30[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv29_local"):
                                            v0 = T.axis.spatial(T.int64(320), k_0_0 * T.int64(16) + k_0_1 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv29[v0, v1])
                                            T.writes(lv29_local[v0, v1])
                                            lv29_local[v0, v1] = lv29[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax0)
                                                v_j = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                                T.reads(lv29_local[v_i // T.int64(8), v_j], lv30_local[v_i // T.int64(32), v_j])
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv29_local[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv30_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv49_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                T.reads(lv49[v0, v1, v2])
                                                T.writes(lv49_pad_local[v0, v1, v2])
                                                lv49_pad_local[v0, v1, v2] = T.if_then_else(v1 < n, lv49[v0, v1, v2], T.float16(0))
                                    for i0_i1_fused_1_2 in range(T.int64(4)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("NT_matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2)
                                                v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(2560), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv49_pad_local[v_i0, v_i1, v_k], decode_local[v_k, v_i2])
                                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv49_pad_local[v_i0, v_i1, v_k] * decode_local[v_k, v_i2]
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                    v2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2], linear_bias3[v2], lv2[v0, v1, v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias3[v2] + lv2[v0, v1, v2]


@T.prim_func
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7(lv1345: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv1346: T.Buffer((T.int64(320), T.int64(2560)), "float16"), p_lv2047: T.handle, linear_bias191: T.Buffer((T.int64(2560),), "float32"), p_lv317: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv2047 = T.match_buffer(p_lv2047, (T.int64(1), n, T.int64(10240)), "float16")
    lv317 = T.match_buffer(p_lv317, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1345[v_i // T.int64(8), v_j], lv1346[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1345[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1346[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(2560), T.int64(10240)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2047[v_i0, v_i1, v_k], var_T_transpose_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv2047[v_i0, v_i1, v_k]) * T.Cast("float32", var_T_transpose_intermediate[v_i2, v_k])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias191[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias191[v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[v_i0, v_i1, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv317[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv317[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_2"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2])


@T.prim_func
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7_after(lv1345: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv1346: T.Buffer((T.int64(320), T.int64(2560)), "float16"), p_lv2047: T.handle, linear_bias191: T.Buffer((T.int64(2560),), "float32"), p_lv317: T.handle, p_output0: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv2047 = T.match_buffer(p_lv2047, (T.int64(1), n, T.int64(10240)), "float16")
    lv317 = T.match_buffer(p_lv317, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16", scope="local")
        lv1345_local = T.alloc_buffer((T.int64(1280), T.int64(2560)), "uint32", scope="local")
        lv1346_local = T.alloc_buffer((T.int64(320), T.int64(2560)), "float16", scope="local")
        lv2047_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(10240)), "float16", scope="local")
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)), scope="local")
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2_init)
                                    v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2_init)
                                    T.reads()
                                    T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(80), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv1346_local"):
                                        v0 = T.axis.spatial(T.int64(320), k_0_0 * T.int64(4) + k_0_1 + ax0)
                                        v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv1346[v0, v1])
                                        T.writes(lv1346_local[v0, v1])
                                        lv1346_local[v0, v1] = lv1346[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv1345_local"):
                                            v0 = T.axis.spatial(T.int64(1280), k_0_0 * T.int64(16) + k_0_1 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv1345[v0, v1])
                                            T.writes(lv1345_local[v0, v1])
                                            lv1345_local[v0, v1] = lv1345[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax0)
                                                v_j = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax1)
                                                T.reads(lv1345_local[v_i // T.int64(8), v_j], lv1346_local[v_i // T.int64(32), v_j])
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1345_local[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1346_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv2047_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                                v2 = T.axis.spatial(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                T.reads(lv2047[v0, v1, v2])
                                                T.writes(lv2047_pad_local[v0, v1, v2])
                                                lv2047_pad_local[v0, v1, v2] = T.if_then_else(v1 < n, lv2047[v0, v1, v2], T.float16(0))
                                    for i0_i1_fused_1_2 in range(T.int64(4)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("NT_matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + i0_i1_fused_1_2)
                                                v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(10240), k_0_0 * T.int64(128) + k_0_1 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv2047_pad_local[v_i0, v_i1, v_k], decode_local[v_k, v_i2])
                                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + T.Cast("float32", lv2047_pad_local[v_i0, v_i1, v_k]) * T.Cast("float32", decode_local[v_k, v_i2])
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i0_i1_fused_0_i0_i1_fused_1_0_fused * T.int64(32) + i0_i1_fused_1_1 * T.int64(4) + ax1)
                                    v2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2], linear_bias191[v2], lv317[v0, v1, v2])
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = T.Cast("float32", T.Cast("float16", var_NT_matmul_intermediate_pad_local[v0, v1, v2] + linear_bias191[v2]) + lv317[v0, v1, v2])

def get_dict_key(func):
    return tvm.ir.structural_hash(func), func


tir_dispatch_dict = {
    get_dict_key(fused_decode4_matmul3): sch_fused_decode4_matmul3(
        fused_decode4_matmul3
    ),
    get_dict_key(
        fused_decode6_fused_matmul7_add1
    ): sch_fused_decode6_fused_matmul7_add1(fused_decode6_fused_matmul7_add1),
    get_dict_key(
        fused_decode5_fused_matmul6_multiply1
    ): sch_fused_decode5_fused_matmul6_multiply1(fused_decode5_fused_matmul6_multiply1),
    get_dict_key(
        fused_decode5_fused_matmul6_silu1
    ): sch_fused_decode5_fused_matmul6_silu1(fused_decode5_fused_matmul6_silu1),
    get_dict_key(
        fused_decode4_fused_matmul4_add1
    ): sch_fused_decode4_fused_matmul4_add1(fused_decode4_fused_matmul4_add1),
    get_dict_key(
        fused_decode3_fused_matmul1_cast2
    ): sch_fused_decode3_fused_matmul1_cast2(fused_decode3_fused_matmul1_cast2),
    get_dict_key(
        fused_decode2_fused_NT_matmul3_add
    ): fused_decode2_fused_NT_matmul3_add_after,
    get_dict_key(fused_decode_NT_matmul): fused_decode_NT_matmul_after,
    get_dict_key(fused_decode1_fused_NT_matmul2_silu): fused_decode1_fused_NT_matmul2_silu_after,
    get_dict_key(fused_decode1_fused_NT_matmul2_multiply): fused_decode1_fused_NT_matmul2_multiply_after,
    get_dict_key(fused_decode_fused_NT_matmul_add): fused_decode_fused_NT_matmul_add_after,
    get_dict_key(fused_decode4_fused_matmul6_add4): sch_fused_decode4_fused_matmul6_add4(fused_decode4_fused_matmul6_add4),
    get_dict_key(fused_decode6_fused_matmul9_add7_cast8_cast12_add5): sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5(fused_decode6_fused_matmul9_add7_cast8_cast12_add5),
    get_dict_key(fused_decode5_fused_matmul8_add6_gelu1_cast11): sch_fused_decode5_fused_matmul8_add6_gelu1_cast11(fused_decode5_fused_matmul8_add6_gelu1_cast11),
    get_dict_key(fused_decode4_fused_matmul6_add4_add5): sch_fused_decode4_fused_matmul6_add4_add5(fused_decode4_fused_matmul6_add4_add5),
    get_dict_key(fused_decode3_matmul3): sch_fused_decode3_matmul3(fused_decode3_matmul3),
    get_dict_key(fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7): sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7),
    get_dict_key(fused_decode2_fused_NT_matmul3_add6_gelu1_cast11): fused_decode2_fused_NT_matmul3_add6_gelu1_cast11_after,
    get_dict_key(fused_decode1_fused_NT_matmul1_add4): fused_decode1_fused_NT_matmul1_add4_after,
    get_dict_key(fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5): fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_after,
    get_dict_key(fused_decode1_fused_NT_matmul1_add4_add5): fused_decode1_fused_NT_matmul1_add4_add5_after,
    get_dict_key(fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7): fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7_after
}


def lookup_func(func):
    for (hash_value, func_before), f_after in tir_dispatch_dict.items():
        if tvm.ir.structural_hash(func) == hash_value and tvm.ir.structural_equal(
            func, func_before
        ):
            return f_after
    return None


@tvm.transform.module_pass(opt_level=0, name="DispatchTIROperatorAdreno")
class DispatchTIROperatorAdreno:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        for gv in mod.functions:
            scheduled_func = lookup_func(mod[gv])
            if scheduled_func is not None:
                mod[gv] = scheduled_func

        return mod
