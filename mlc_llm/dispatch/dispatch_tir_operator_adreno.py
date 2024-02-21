import tvm
from tvm import IRModule
from tvm.script import tir as T


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode4_matmul3_after(
    lv1587: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv1588: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    lv1583: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32768)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32768)), "float16", scope="local"
    )
    lv1587_local = T.alloc_buffer(
        (T.int64(512), T.int64(4096)), "uint32", scope="local"
    )
    lv1588_local = T.alloc_buffer(
        (T.int64(128), T.int64(4096)), "float16", scope="local"
    )
    lv1583_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2048)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(32768),
                            i0_i1_i2_fused_0 * T.int64(1024)
                            + i0_i1_i2_fused_1 * T.int64(32)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1583_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(2048)
                                        + ax2_1 * T.int64(64)
                                        + (ax2_y * T.int64(8) + ax2_2),
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(2048),
                                        (
                                            ax2_1 * T.int64(64)
                                            + ax2_y * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv1583[v0, v1, v2])
                                    T.writes(lv1583_shared[v0, v1, v2k])
                                    lv1583_shared[v0, v1, v2k] = lv1583[v0, v1, v2]
                for k_1 in range(T.int64(8)):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(32768),
                                    i0_i1_i2_fused_0 * T.int64(1024)
                                    + i0_i1_i2_fused_1 * T.int64(32)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1588_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(64)
                                        + (k_1 * T.int64(8) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv1588[v0, v1])
                                    T.writes(lv1588_local[v0, v1])
                                    lv1588_local[v0, v1] = lv1588[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1587_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(256)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv1587[v0, v1])
                                        T.writes(lv1587_local[v0, v1])
                                        lv1587_local[v0, v1] = lv1587[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(32768),
                                            i0_i1_i2_fused_0 * T.int64(1024)
                                            + i0_i1_i2_fused_1 * T.int64(32)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(2048)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(2048),
                                            (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv1583_shared[v_i0, v_i1, v_ki],
                                            lv1587_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv1583_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv1587_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(32768),
                                        i0_i1_i2_fused_0 * T.int64(1024)
                                        + i0_i1_i2_fused_1 * T.int64(32)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(64)
                                        + (k_1 * T.int64(8) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv1588_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv1588_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(32768),
                                i0_i1_i2_fused_0 * T.int64(1024)
                                + i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1583_shared[v0, v1, v2])
                            lv1583_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction_sum"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(4))
                            T.reads(lv1583_shared[v0, v1, v2])
                            T.writes(lv1583_shared[v0, v1, v2])
                            lv1583_shared[v0, v1, v2] = (
                                lv1583_shared[v0, v1, v2]
                                + lv1583_shared[v0, v1, v2 + T.int64(16)]
                            )
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(4096),
                                i0_i1_i2_fused_0 * T.int64(128)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1583_shared[v0, v1, v_i2k])
                            T.writes(var_matmul_intermediate[v0, v1, v2])
                            var_matmul_intermediate[v0, v1, v2] = (
                                lv1583_shared[v0, v1, v_i2k]
                                + lv1583_shared[v0, v1, v_i2k + T.int64(4)]
                                + lv1583_shared[v0, v1, v_i2k + T.int64(8)]
                                + lv1583_shared[v0, v1, v_i2k + T.int64(12)]
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode6_fused_matmul7_add1_after(
    lv1623: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv1624: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    lv200: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    lv198: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(16384)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(16384)), "float16", scope="local"
    )
    lv1623_local = T.alloc_buffer(
        (T.int64(1376), T.int64(4096)), "uint32", scope="local"
    )
    lv1624_local = T.alloc_buffer(
        (T.int64(344), T.int64(4096)), "float16", scope="local"
    )
    lv200_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2752)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(8), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(16384),
                            i0_i1_i2_fused_0 * T.int64(2048)
                            + i0_i1_i2_fused_1 * T.int64(16)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(4)):
                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(3)):
                    for ax2_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                        for ax2_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            for ax2_2 in T.vectorized(T.int64(2)):
                                with T.block("lv200_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(11008),
                                        k_0 * T.int64(2752)
                                        + (
                                            ax2_0 * T.int64(1024)
                                            + ax2_1 * T.int64(8)
                                            + (ax2_y * T.int64(2) + ax2_2)
                                        ),
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(2752),
                                        (
                                            ax2_0 * T.int64(1024)
                                            + ax2_1 * T.int64(8)
                                            + (ax2_y * T.int64(2) + ax2_2)
                                        ),
                                    )
                                    T.where(
                                        (ax2_0 * T.int64(128) + ax2_1) < T.int64(344)
                                    )
                                    T.reads(lv200[v0, v1, v2])
                                    T.writes(lv200_shared[v0, v1, v2k])
                                    lv200_shared[v0, v1, v2k] = lv200[v0, v1, v2]
                for k_1 in range(T.int64(22)):
                    for ax2_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        with T.block("lv1624_check"):
                            T.where((k_1 * T.int64(4) + ax2_y) < T.int64(86))
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_init_local"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2k = T.axis.spatial(
                                            T.int64(16384),
                                            i0_i1_i2_fused_0 * T.int64(2048)
                                            + i0_i1_i2_fused_1 * T.int64(16)
                                            + ax2_y * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads()
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = T.float16(0)
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1624_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(344),
                                            k_0 * T.int64(86)
                                            + (k_1 * T.int64(4) + ax2_y)
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(512)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv1624[v0, v1])
                                        T.writes(lv1624_local[v0, v1])
                                        lv1624_local[v0, v1] = lv1624[v0, v1]
                            for k_2 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(4)):
                                        with T.block("lv1623_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(1376),
                                                k_0 * T.int64(344)
                                                + (k_1 * T.int64(4) + ax2_y)
                                                * T.int64(4)
                                                + k_2
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(512)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                            )
                                            T.reads(lv1623[v0, v1])
                                            T.writes(lv1623_local[v0, v1])
                                            lv1623_local[v0, v1] = lv1623[v0, v1]
                                for k_3 in range(T.int64(8)):
                                    for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(
                                                T.int64(1), T.int64(0)
                                            )
                                            v_i1 = T.axis.spatial(
                                                T.int64(1), T.int64(0)
                                            )
                                            v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(512)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_i2k = T.axis.spatial(
                                                T.int64(16384),
                                                i0_i1_i2_fused_0 * T.int64(2048)
                                                + i0_i1_i2_fused_1 * T.int64(16)
                                                + ax2_y * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(11008),
                                                k_0 * T.int64(2752)
                                                + (k_1 * T.int64(4) + ax2_y)
                                                * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            v_ki = T.axis.reduce(
                                                T.int64(2752),
                                                (k_1 * T.int64(4) + ax2_y) * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            T.reads(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2k
                                                ],
                                                lv200_shared[v_i0, v_i1, v_ki],
                                                lv1623_local[v_k // T.int64(8), v_i2],
                                            )
                                            T.writes(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2k
                                                ]
                                            )
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ] = var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ] + lv200_shared[
                                                v_i0, v_i1, v_ki
                                            ] * (
                                                (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv1623_local[
                                                                    v_k // T.int64(8),
                                                                    v_i2,
                                                                ],
                                                                T.Cast(
                                                                    "uint32",
                                                                    v_k % T.int64(8),
                                                                )
                                                                * T.uint32(4),
                                                            ),
                                                            T.uint32(15),
                                                        ),
                                                    )
                                                    - T.float16(7)
                                                )
                                            )
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("multiple_scale"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2k = T.axis.spatial(
                                            T.int64(16384),
                                            i0_i1_i2_fused_0 * T.int64(2048)
                                            + i0_i1_i2_fused_1 * T.int64(16)
                                            + ax2_y * T.int64(4)
                                            + ax1,
                                        )
                                        v0 = T.axis.spatial(
                                            T.int64(344),
                                            k_0 * T.int64(86)
                                            + (k_1 * T.int64(4) + ax2_y)
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(512)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(
                                            lv1624_local[v0, v1],
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local[
                                            v_i0, v_i1, v_i2k
                                        ] = (
                                            var_matmul_intermediate_local[
                                                v_i0, v_i1, v_i2k
                                            ]
                                            + var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                            * lv1624_local[v0, v1]
                                        )
            for ax2_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(16)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(16384),
                                i0_i1_i2_fused_0 * T.int64(2048)
                                + i0_i1_i2_fused_1 * T.int64(16)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv200_shared[v0, v1, v2])
                            lv200_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(4096),
                                i0_i1_i2_fused_0 * T.int64(512)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(16)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv200_shared[v0, v1, v_i2k])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = (
                                lv198[v0, v1, v2]
                                + lv200_shared[v0, v1, v_i2k]
                                + lv200_shared[v0, v1, v_i2k + T.int64(4)]
                                + lv200_shared[v0, v1, v_i2k + T.int64(8)]
                                + lv200_shared[v0, v1, v_i2k + T.int64(12)]
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode5_fused_matmul6_multiply1_after(
    lv1617: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv1618: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    lv1622: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16", scope="local"
    )
    lv1617_local = T.alloc_buffer(
        (T.int64(512), T.int64(11008)), "uint32", scope="local"
    )
    lv1618_local = T.alloc_buffer(
        (T.int64(128), T.int64(11008)), "float16", scope="local"
    )
    lv1622_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(1024)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(43), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(22016),
                            i0_i1_i2_fused_0 * T.int64(512)
                            + i0_i1_i2_fused_1 * T.int64(8)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1622_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(1024)
                                        + ax2_y * T.int64(512)
                                        + ax2_1 * T.int64(8)
                                        + ax2_2,
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(1024),
                                        (
                                            ax2_y * T.int64(512)
                                            + ax2_1 * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv1622[v0, v1, v2])
                                    T.writes(lv1622_shared[v0, v1, v2k])
                                    lv1622_shared[v0, v1, v2k] = lv1622[v0, v1, v2]
                for k_1 in range(T.int64(16)):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(22016),
                                    i0_i1_i2_fused_0 * T.int64(512)
                                    + i0_i1_i2_fused_1 * T.int64(8)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1618_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(11008),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv1618[v0, v1])
                                    T.writes(lv1618_local[v0, v1])
                                    lv1618_local[v0, v1] = lv1618[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1617_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(128)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(11008),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv1617[v0, v1])
                                        T.writes(lv1617_local[v0, v1])
                                        lv1617_local[v0, v1] = lv1617[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(11008),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(512)
                                            + i0_i1_i2_fused_1 * T.int64(8)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(1024)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(1024),
                                            (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv1622_shared[v_i0, v_i1, v_ki],
                                            lv1617_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv1622_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv1617_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(22016),
                                        i0_i1_i2_fused_0 * T.int64(512)
                                        + i0_i1_i2_fused_1 * T.int64(8)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(11008),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv1618_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv1618_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(22016),
                                i0_i1_i2_fused_0 * T.int64(512)
                                + i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1622_shared[v0, v1, v2])
                            lv1622_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(11008),
                                i0_i1_i2_fused_0 * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1622_shared[v0, v1, v_i2k], lv4[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] * (
                                lv1622_shared[v0, v1, v_i2k]
                                + lv1622_shared[v0, v1, v_i2k + T.int64(4)]
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


@T.prim_func(private=True)
def fused_fused_decode9_matmul7(
    lv19: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    lv20: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv19[v_i // T.int64(8), v_j], lv20[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv19[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv20[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]
            )


@T.prim_func(private=True)
def fused_fused_decode9_matmul7_after(
    lv19: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    lv20: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(352256)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(352256)), "float16", scope="local"
    )
    lv19_local = T.alloc_buffer((T.int64(512), T.int64(22016)), "uint32", scope="local")
    lv20_local = T.alloc_buffer(
        (T.int64(128), T.int64(22016)), "float16", scope="local"
    )
    lv1654_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(172), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(352256),
                            i0_i1_i2_fused_0 * T.int64(2048)
                            + i0_i1_i2_fused_1 * T.int64(64)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1654_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(4096)
                                        + ax2_y * T.int64(256)
                                        + ax2_1 * T.int64(8)
                                        + ax2_2,
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(4096),
                                        (
                                            ax2_y * T.int64(256)
                                            + ax2_1 * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv1654[v0, v1, v2])
                                    T.writes(lv1654_shared[v0, v1, v2k])
                                    lv1654_shared[v0, v1, v2k] = lv1654[v0, v1, v2]
                for k_1 in range(T.int64(8)):
                    for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(352256),
                                    i0_i1_i2_fused_0 * T.int64(2048)
                                    + i0_i1_i2_fused_1 * T.int64(64)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv20_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(128)
                                        + (k_1 * T.int64(16) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(22016),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv20[v0, v1])
                                    T.writes(lv20_local[v0, v1])
                                    lv20_local[v0, v1] = lv20[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv19_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(512)
                                            + (k_1 * T.int64(16) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv19[v0, v1])
                                        T.writes(lv19_local[v0, v1])
                                        lv19_local[v0, v1] = lv19[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(352256),
                                            i0_i1_i2_fused_0 * T.int64(2048)
                                            + i0_i1_i2_fused_1 * T.int64(64)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(4096)
                                            + (k_1 * T.int64(16) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(4096),
                                            (k_1 * T.int64(16) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv1654_shared[v_i0, v_i1, v_ki],
                                            lv19_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv1654_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv19_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(352256),
                                        i0_i1_i2_fused_0 * T.int64(2048)
                                        + i0_i1_i2_fused_1 * T.int64(64)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(128)
                                        + (k_1 * T.int64(16) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(22016),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv20_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv20_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(64)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(352256),
                                i0_i1_i2_fused_0 * T.int64(2048)
                                + i0_i1_i2_fused_1 * T.int64(64)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1654_shared[v0, v1, v2])
                            lv1654_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction_1"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(64)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(8))
                            T.reads(lv1654_shared[v0, v1, v_i2k])
                            T.writes(lv1654_shared[v0, v1, v_i2k])
                            lv1654_shared[v0, v1, v_i2k] = (
                                lv1654_shared[v0, v1, v_i2k]
                                + lv1654_shared[v0, v1, v_i2k + T.int64(32)]
                            )
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction_2"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(64)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(4))
                            T.reads(lv1654_shared[v0, v1, v_i2k])
                            T.writes(lv1654_shared[v0, v1, v_i2k])
                            lv1654_shared[v0, v1, v_i2k] = (
                                lv1654_shared[v0, v1, v_i2k]
                                + lv1654_shared[v0, v1, v_i2k + T.int64(16)]
                            )
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(22016),
                                i0_i1_i2_fused_0 * T.int64(128)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                i0_i1_i2_fused_1 * T.int64(64)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1654_shared[v0, v1, v_i2k])
                            T.writes(var_matmul_intermediate[v0, v1, v2])
                            var_matmul_intermediate[v0, v1, v2] = (
                                lv1654_shared[v0, v1, v_i2k]
                                + lv1654_shared[v0, v1, v_i2k + T.int64(4)]
                                + lv1654_shared[v0, v1, v_i2k + T.int64(8)]
                                + lv1654_shared[v0, v1, v_i2k + T.int64(12)]
                            )


@T.prim_func(private=True)
def fused_fused_decode7_matmul4(
    lv3: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    lv4: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    lv1615: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(12288)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3[v_i // T.int64(8), v_j], lv4[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv3[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv4[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1615[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1615[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]
            )


@T.prim_func(private=True)
def fused_fused_decode7_matmul4_after(
    lv3: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    lv4: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    lv1615: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(12288)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(24576)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(24576)), "float16", scope="local"
    )
    lv3_local = T.alloc_buffer((T.int64(512), T.int64(12288)), "uint32", scope="local")
    lv4_local = T.alloc_buffer((T.int64(128), T.int64(12288)), "float16", scope="local")
    lv1615_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(1024)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(24576),
                            i0_i1_i2_fused_0 * T.int64(512)
                            + i0_i1_i2_fused_1 * T.int64(8)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1615_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(1024)
                                        + ax2_y * T.int64(512)
                                        + ax2_1 * T.int64(8)
                                        + ax2_2,
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(1024),
                                        (
                                            ax2_y * T.int64(512)
                                            + ax2_1 * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv1615[v0, v1, v2])
                                    T.writes(lv1615_shared[v0, v1, v2k])
                                    lv1615_shared[v0, v1, v2k] = lv1615[v0, v1, v2]
                for k_1 in range(T.int64(16)):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(24576),
                                    i0_i1_i2_fused_0 * T.int64(512)
                                    + i0_i1_i2_fused_1 * T.int64(8)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv4_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(12288),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv4[v0, v1])
                                    T.writes(lv4_local[v0, v1])
                                    lv4_local[v0, v1] = lv4[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv3_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(128)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(12288),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv3[v0, v1])
                                        T.writes(lv3_local[v0, v1])
                                        lv3_local[v0, v1] = lv3[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(12288),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(24576),
                                            i0_i1_i2_fused_0 * T.int64(512)
                                            + i0_i1_i2_fused_1 * T.int64(8)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(1024)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(1024),
                                            (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv1615_shared[v_i0, v_i1, v_ki],
                                            lv3_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv1615_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv3_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(24576),
                                        i0_i1_i2_fused_0 * T.int64(512)
                                        + i0_i1_i2_fused_1 * T.int64(8)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(12288),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv4_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv4_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(24576),
                                i0_i1_i2_fused_0 * T.int64(512)
                                + i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1615_shared[v0, v1, v2])
                            lv1615_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(12288),
                                i0_i1_i2_fused_0 * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1615_shared[v0, v1, v_i2k])
                            T.writes(var_matmul_intermediate[v0, v1, v2])
                            var_matmul_intermediate[v0, v1, v2] = (
                                lv1615_shared[v0, v1, v_i2k]
                                + lv1615_shared[v0, v1, v_i2k + T.int64(4)]
                            )


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode5_fused_matmul6_silu1_after(
    lv1611: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv1612: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    lv1622: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16", scope="local"
    )
    lv1611_local = T.alloc_buffer(
        (T.int64(512), T.int64(11008)), "uint32", scope="local"
    )
    lv1612_local = T.alloc_buffer(
        (T.int64(128), T.int64(11008)), "float16", scope="local"
    )
    lv1622_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(1024)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(43), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(22016),
                            i0_i1_i2_fused_0 * T.int64(512)
                            + i0_i1_i2_fused_1 * T.int64(8)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(4)):
                for ax2_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1622_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(1024)
                                        + ax2_y * T.int64(512)
                                        + ax2_1 * T.int64(8)
                                        + ax2_2,
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(1024),
                                        (
                                            ax2_y * T.int64(512)
                                            + ax2_1 * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv1622[v0, v1, v2])
                                    T.writes(lv1622_shared[v0, v1, v2k])
                                    lv1622_shared[v0, v1, v2k] = lv1622[v0, v1, v2]
                for k_1 in range(T.int64(16)):
                    for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(22016),
                                    i0_i1_i2_fused_0 * T.int64(512)
                                    + i0_i1_i2_fused_1 * T.int64(8)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1612_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(11008),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv1612[v0, v1])
                                    T.writes(lv1612_local[v0, v1])
                                    lv1612_local[v0, v1] = lv1612[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1611_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(128)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(11008),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv1611[v0, v1])
                                        T.writes(lv1611_local[v0, v1])
                                        lv1611_local[v0, v1] = lv1611[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(11008),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(512)
                                            + i0_i1_i2_fused_1 * T.int64(8)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(1024)
                                            + (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(1024),
                                            (k_1 * T.int64(2) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv1622_shared[v_i0, v_i1, v_ki],
                                            lv1611_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv1622_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv1611_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(22016),
                                        i0_i1_i2_fused_0 * T.int64(512)
                                        + i0_i1_i2_fused_1 * T.int64(8)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(32)
                                        + (k_1 * T.int64(2) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(11008),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv1612_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv1612_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(22016),
                                i0_i1_i2_fused_0 * T.int64(512)
                                + i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1622_shared[v0, v1, v2])
                            lv1622_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1622_shared[v0, v1, v2])
                            T.writes(lv1622_shared[v0, v1, v2])
                            lv1622_shared[v0, v1, v2] = (
                                lv1622_shared[v0, v1, v2]
                                + lv1622_shared[v0, v1, v2 + T.int64(4)]
                            )
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(11008),
                                i0_i1_i2_fused_0 * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(512),
                                i0_i1_i2_fused_1 * T.int64(8)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1622_shared[v0, v1, v_i2k])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = lv1622_shared[
                                v0, v1, v_i2k
                            ] * T.sigmoid(lv1622_shared[v0, v1, v_i2k])


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

@T.prim_func(private=True)
def fused_decode81_fused_matmul1_cast2(
    lv1576: T.Buffer((T.int64(512), T.int64(64000)), "uint32"),
    lv1577: T.Buffer((T.int64(128), T.int64(64000)), "float16"),
    lv1575: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(64000)), "float32"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(64000)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(64000)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(64000)):
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
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(64000), T.int64(4096)):
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
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(64000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_matmul_intermediate[v_i0, v_i1, v_i2]
            )

def sch_fused_decode81_fused_matmul1_cast2(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[160, 100, 4]
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




@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode4_fused_matmul4_add1_after(
    lv1605: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv1606: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    lv197: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv1581: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32768)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32768)), "float16", scope="local"
    )
    lv1605_local = T.alloc_buffer(
        (T.int64(512), T.int64(4096)), "uint32", scope="local"
    )
    lv1606_local = T.alloc_buffer(
        (T.int64(128), T.int64(4096)), "float16", scope="local"
    )
    lv197_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2048)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(32768),
                            i0_i1_i2_fused_0 * T.int64(1024)
                            + i0_i1_i2_fused_1 * T.int64(32)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(2)):
                for ax2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv197_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(2048)
                                        + ax2_1 * T.int64(64)
                                        + (ax2_y * T.int64(8) + ax2_2),
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(2048),
                                        (
                                            ax2_1 * T.int64(64)
                                            + ax2_y * T.int64(8)
                                            + ax2_2
                                        ),
                                    )
                                    T.reads(lv197[v0, v1, v2])
                                    T.writes(lv197_shared[v0, v1, v2k])
                                    lv197_shared[v0, v1, v2k] = lv197[v0, v1, v2]
                for k_1 in range(T.int64(8)):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(32768),
                                    i0_i1_i2_fused_0 * T.int64(1024)
                                    + i0_i1_i2_fused_1 * T.int64(32)
                                    + ax2_y * T.int64(4)
                                    + ax1,
                                )
                                T.reads()
                                T.writes(
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ]
                                )
                                var_matmul_intermediate_local_batch[
                                    v_i0, v_i1, v_i2k
                                ] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1606_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(64)
                                        + (k_1 * T.int64(8) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(lv1606[v0, v1])
                                    T.writes(lv1606_local[v0, v1])
                                    lv1606_local[v0, v1] = lv1606[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1605_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(256)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(4)
                                            + k_2
                                            + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv1605[v0, v1])
                                        T.writes(lv1605_local[v0, v1])
                                        lv1605_local[v0, v1] = lv1605[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(32768),
                                            i0_i1_i2_fused_0 * T.int64(1024)
                                            + i0_i1_i2_fused_1 * T.int64(32)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2,
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(2048)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(2048),
                                            (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8)
                                            + k_3,
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ],
                                            lv197_shared[v_i0, v_i1, v_ki],
                                            lv1605_local[v_k // T.int64(8), v_i2],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2k
                                            ]
                                        )
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] = var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ] + lv197_shared[
                                            v_i0, v_i1, v_ki
                                        ] * (
                                            (
                                                T.Cast(
                                                    "float16",
                                                    T.bitwise_and(
                                                        T.shift_right(
                                                            lv1605_local[
                                                                v_k // T.int64(8), v_i2
                                                            ],
                                                            T.Cast(
                                                                "uint32",
                                                                v_k % T.int64(8),
                                                            )
                                                            * T.uint32(4),
                                                        ),
                                                        T.uint32(15),
                                                    ),
                                                )
                                                - T.float16(7)
                                            )
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(32768),
                                        i0_i1_i2_fused_0 * T.int64(1024)
                                        + i0_i1_i2_fused_1 * T.int64(32)
                                        + ax2_y * T.int64(4)
                                        + ax1,
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(64)
                                        + (k_1 * T.int64(8) + ax2_y)
                                        + ax0,
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads(
                                        lv1606_local[v0, v1],
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ],
                                    )
                                    T.writes(
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                    )
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        + var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                        * lv1606_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(32768),
                                i0_i1_i2_fused_0 * T.int64(1024)
                                + i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv197_shared[v0, v1, v2])
                            lv197_shared[v0, v1, v2] = var_matmul_intermediate_local[
                                v0, v1, v_i2k
                            ]
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction_sum"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(4))
                            T.reads(lv197_shared[v0, v1, v2])
                            T.writes(lv197_shared[v0, v1, v2])
                            lv197_shared[v0, v1, v2] = (
                                lv197_shared[v0, v1, v2]
                                + lv197_shared[v0, v1, v2 + T.int64(16)]
                            )
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(4096),
                                i0_i1_i2_fused_0 * T.int64(128)
                                + i0_i1_i2_fused_1 * T.int64(4)
                                + ax2,
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(1024),
                                i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4)
                                + ax2,
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv197_shared[v0, v1, v_i2k], lv1581[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = (
                                lv1581[v0, v1, v2]
                                + lv197_shared[v0, v1, v_i2k]
                                + lv197_shared[v0, v1, v_i2k + T.int64(4)]
                                + lv197_shared[v0, v1, v_i2k + T.int64(8)]
                                + lv197_shared[v0, v1, v_i2k + T.int64(12)]
                            )

@T.prim_func(private=True)
def fused_decode82_fused_matmul1_cast2(
    lv1576: T.Buffer((T.int64(512), T.int64(64000)), "uint32"),
    lv1577: T.Buffer((T.int64(128), T.int64(64000)), "float16"),
    lv1575: T.Buffer((T.int64(1), T.int64(1), T.int64(2048)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(64000)), "float32"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2048), T.int64(64000)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(64000)), "float16"
    )
    for i, j in T.grid(T.int64(2048), T.int64(64000)):
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
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(64000), T.int64(4096)):
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
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(64000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_matmul_intermediate[v_i0, v_i1, v_i2]
            )

def sch_fused_decode82_fused_matmul1_cast2(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[160, 100, 4]
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
def fused_decode3_fused_matmul1_cast2_after(
    lv1576: T.Buffer((T.int64(512), T.int64(32000)), "uint32"),
    lv1577: T.Buffer((T.int64(128), T.int64(32000)), "float16"),
    lv1575: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(512000)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(512000)), "float16", scope="local"
    )
    lv1576_local = T.alloc_buffer(
        (T.int64(512), T.int64(32000)), "uint32", scope="local"
    )
    lv1577_local = T.alloc_buffer(
        (T.int64(128), T.int64(32000)), "float16", scope="local"
    )
    lv1575_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    )
    for i0_i1_i2_fused_0 in T.thread_binding(T.int64(125), thread="blockIdx.x"):
        for i0_i1_i2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(
                            T.int64(512000),
                            i0_i1_i2_fused_0 * T.int64(2048)
                            + i0_i1_i2_fused_1 * T.int64(32)
                            + ax2_y * T.int64(4)
                            + i0_i1_i2_fused_2_init
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
            for k_0 in range(T.int64(1)):
                for ax2_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2_2 in T.vectorized(T.int64(8)):
                                with T.block("lv1575_shared"):
                                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                    v2 = T.axis.spatial(
                                        T.int64(4096),
                                        k_0 * T.int64(4096)
                                        + ax2_y * T.int64(512)
                                        + ax2_1 * T.int64(8) + ax2_2
                                    )
                                    v2k = T.axis.spatial(
                                        T.int64(4096),
                                        (ax2_y * T.int64(512)
                                        + ax2_1 * T.int64(8) + ax2_2)
                                    )
                                    T.reads(lv1575[v0, v1, v2])
                                    T.writes(lv1575_shared[v0, v1, v2k])
                                    lv1575_shared[v0, v1, v2k] = lv1575[v0, v1, v2]
                for k_1 in range(T.int64(16)):
                    for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1 in T.vectorized(T.int64(4)):
                            with T.block("matmul_init_local"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2k = T.axis.spatial(
                                    T.int64(512000),
                                    i0_i1_i2_fused_0 * T.int64(2048)
                                    + i0_i1_i2_fused_1 * T.int64(32)
                                    + ax2_y * T.int64(4) + ax1
                                )
                                T.reads()
                                T.writes(var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k])
                                var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k] = T.float16(0)
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("lv1577_local"):
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(128)
                                        + (k_1 * T.int64(8) + ax2_y) + ax0
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(32000),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4) + ax1
                                    )
                                    T.reads(lv1577[v0, v1])
                                    T.writes(lv1577_local[v0, v1])
                                    lv1577_local[v0, v1] = lv1577[v0, v1]
                        for k_2 in range(T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv1576_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(512),
                                            k_0 * T.int64(512)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(4)
                                            + k_2 + ax0
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(32000),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1
                                        )
                                        T.reads(lv1576[v0, v1])
                                        T.writes(lv1576_local[v0, v1])
                                        lv1576_local[v0, v1] = lv1576[v0, v1]
                            for k_3 in range(T.int64(8)):
                                for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("matmul_update"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i2 = T.axis.spatial(
                                            T.int64(32000),
                                            i0_i1_i2_fused_0 * T.int64(256)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + i0_i1_i2_fused_2
                                        )
                                        v_i2k = T.axis.spatial(
                                            T.int64(512000),
                                            i0_i1_i2_fused_0 * T.int64(2048)
                                            + i0_i1_i2_fused_1 * T.int64(32)
                                            + ax2_y * T.int64(4)
                                            + i0_i1_i2_fused_2
                                        )
                                        v_k = T.axis.reduce(
                                            T.int64(4096),
                                            k_0 * T.int64(4096)
                                            + (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8) + k_3
                                        )
                                        v_ki = T.axis.reduce(
                                            T.int64(4096),
                                            (k_1 * T.int64(8) + ax2_y) * T.int64(32)
                                            + k_2 * T.int64(8) + k_3
                                        )
                                        T.reads(
                                            var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k],
                                            lv1575_shared[v_i0, v_i1, v_ki], lv1576_local[v_k // T.int64(8), v_i2]
                                        )
                                        T.writes(var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k])
                                        var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k] = (
                                            var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k]
                                            + lv1575_shared[v_i0, v_i1, v_ki]
                                            * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv1576_local[v_k // T.int64(8), v_i2],
                                            T.Cast("uint32", v_k % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)))
                                        )
                        for ax0 in range(T.int64(1)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("multiple_scale"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2k = T.axis.spatial(
                                        T.int64(512000),
                                        i0_i1_i2_fused_0 * T.int64(2048)
                                        + i0_i1_i2_fused_1 * T.int64(32)
                                        + ax2_y * T.int64(4) + ax1
                                    )
                                    v0 = T.axis.spatial(
                                        T.int64(128),
                                        k_0 * T.int64(128)
                                        + (k_1 * T.int64(8) + ax2_y) + ax0
                                    )
                                    v1 = T.axis.spatial(
                                        T.int64(32000),
                                        i0_i1_i2_fused_0 * T.int64(256)
                                        + i0_i1_i2_fused_1 * T.int64(4) + ax1
                                    )
                                    T.reads(
                                        lv1577_local[v0, v1],
                                        var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k]
                                    )
                                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2k])
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2k] = (
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2k]
                                        +  var_matmul_intermediate_local_batch[v_i0, v_i1, v_i2k] * lv1577_local[v0, v1]
                                    )
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_update"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(2048),
                                ax2_y * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4) + ax2
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(512000),
                                i0_i1_i2_fused_0 * T.int64(2048)
                                + i0_i1_i2_fused_1 * T.int64(32)
                                + ax2_y * T.int64(4) + ax2
                            )
                            T.reads(var_matmul_intermediate_local[v0, v1, v_i2k])
                            T.writes(lv1575_shared[v0, v1, v2])
                            lv1575_shared[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v_i2k]
            for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("reduction_2"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                ax2_y * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4) + ax2
                            )
                            T.where(ax2_y < T.int64(4))
                            T.reads(lv1575_shared[v0, v1, v_i2k])
                            T.writes(lv1575_shared[v0, v1, v_i2k])
                            lv1575_shared[v0, v1, v_i2k] = (
                                lv1575_shared[v0, v1, v_i2k] + lv1575_shared[v0, v1, v_i2k + T.int64(1024)]
                            )
            for ax2_y in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2 in T.vectorized(T.int64(4)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(
                                T.int64(32000),
                                i0_i1_i2_fused_0 * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4) + ax2
                            )
                            v_i2k = T.axis.spatial(
                                T.int64(2048),
                                ax2_y * T.int64(256)
                                + i0_i1_i2_fused_1 * T.int64(4) + ax2
                            )
                            T.where(ax2_y < T.int64(1))
                            T.reads(lv1575_shared[v0, v1, v_i2k])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = T.Cast(
                                "float32", lv1575_shared[v0, v1, v_i2k]
                                + lv1575_shared[v0, v1, v_i2k + T.int64(256)]
                                + lv1575_shared[v0, v1, v_i2k + T.int64(512)]
                                + lv1575_shared[v0, v1, v_i2k + T.int64(768)]
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode2_fused_NT_matmul3_add_after(
    lv8: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv9: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    p_lv5: T.handle,
    p_lv3: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv6 = T.match_buffer(p_lv5, (1, n, 11008), "float16")
    lv2 = T.match_buffer(p_lv3, (1, n, 4096), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (1, n, 4096), "float16")

    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(4096)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(4096)), "float16", scope="local"
    )
    lv8_local = T.alloc_buffer((T.int64(512), T.int64(4096)), "uint32", scope="local")
    lv9_local = T.alloc_buffer(
        (T.int64(128), T.int64(4096)), "float16", scope="local"
    )
    #lv6_shared = T.alloc_buffer(
    #    (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    #)
    for i0_i1_i2_fused_n in T.thread_binding(((n+7)//8), thread="blockIdx.y"):
        for i0_i1_i2_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    with T.block("n_check"):
                        T.where((i0_i1_i2_fused_n * T.int64(8) + ax2_y) < n)
                        for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                v_i2 = T.axis.spatial(
                                    T.int64(4096),
                                    i0_i1_i2_fused_0 * T.int64(128)
                                    + i0_i1_i2_fused_1 * T.int64(4)
                                    + i0_i1_i2_fused_2_init
                                )
                                T.reads()
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_1 in range(T.int64(344)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("matmul_init_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2k = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                    )
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ] = T.float16(0)
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(344), k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_2 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(4)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(1376),
                                                k_1 * T.int64(4)
                                                + k_2
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                            )
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_3 in range(T.int64(8)):
                                    for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                            v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(11008),
                                                k_1 * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            T.reads(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv6[v_i0, v_i1, v_k],
                                                lv8_local[v_k // T.int64(8), v_i2],
                                            )
                                            T.writes(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] = var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] + lv6[
                                                v_i0, v_i1, v_k
                                            ] * (
                                                (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv8_local[
                                                                    v_k // T.int64(8), v_i2
                                                                ],
                                                                T.Cast(
                                                                    "uint32",
                                                                    v_k % T.int64(8),
                                                                )
                                                                * T.uint32(4),
                                                            ),
                                                            T.uint32(15),
                                                        ),
                                                    )
                                                    - T.float16(7)
                                                )
                                            )
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("multiple_scale"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                        v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                        )
                                        v0 = T.axis.spatial(
                                            T.int64(344),
                                            k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(
                                            lv9_local[v0, v1],
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                        )
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = (
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                            + var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ]
                                            * lv9_local[v0, v1]
                                        )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2 in T.vectorized(T.int64(4)):
                                with T.block("var_matmul_intermediate_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax2,
                                    )
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2[v_i0, v_i1, v_i2])
                                    T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2[v_i0, v_i1, v_i2]


@T.prim_func(private=True)
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode1_fused_NT_matmul2_multiply_after(
    lv43: T.Buffer((512, 11008), "uint32"),
    lv44: T.Buffer((128, 11008), "float16"),
    p_lv45: T.handle,
    p_lv132: T.handle,
    p_output0: T.handle,
):
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


@T.prim_func(private=True)
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


@T.prim_func(private=True)
def fused_decode_fused_NT_matmul_add_after(
    lv8: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv9: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    p_lv41: T.handle,
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv6 = T.match_buffer(p_lv41, (1, n, 4096), "float16")
    lv2 = T.match_buffer(p_lv2, (1, n, 4096), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (1, n, 4096), "float16")

    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(4096)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(4096)), "float16", scope="local"
    )
    lv8_local = T.alloc_buffer((T.int64(512), T.int64(4096)), "uint32", scope="local")
    lv9_local = T.alloc_buffer(
        (T.int64(128), T.int64(4096)), "float16", scope="local"
    )
    #lv6_shared = T.alloc_buffer(
    #    (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    #)
    for i0_i1_i2_fused_n in T.thread_binding(((n+7)//8), thread="blockIdx.y"):
        for i0_i1_i2_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    with T.block("n_check"):
                        T.where((i0_i1_i2_fused_n * T.int64(8) + ax2_y) < n)
                        for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                v_i2 = T.axis.spatial(
                                    T.int64(4096),
                                    i0_i1_i2_fused_0 * T.int64(128)
                                    + i0_i1_i2_fused_1 * T.int64(4)
                                    + i0_i1_i2_fused_2_init
                                )
                                T.reads()
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_1 in range(T.int64(128)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("matmul_init_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2k = T.axis.spatial(
                                        T.int64(4096),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                    )
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ] = T.float16(0)
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(128), k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_2 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(4)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(512),
                                                k_1 * T.int64(4)
                                                + k_2
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                            )
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_3 in range(T.int64(8)):
                                    for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                            v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(4096),
                                                k_1 * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            T.reads(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv6[v_i0, v_i1, v_k],
                                                lv8_local[v_k // T.int64(8), v_i2],
                                            )
                                            T.writes(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] = var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] + lv6[
                                                v_i0, v_i1, v_k
                                            ] * (
                                                (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv8_local[
                                                                    v_k // T.int64(8), v_i2
                                                                ],
                                                                T.Cast(
                                                                    "uint32",
                                                                    v_k % T.int64(8),
                                                                )
                                                                * T.uint32(4),
                                                            ),
                                                            T.uint32(15),
                                                        ),
                                                    )
                                                    - T.float16(7)
                                                )
                                            )
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("multiple_scale"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                        v_i2 = T.axis.spatial(
                                                T.int64(4096),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                        )
                                        v0 = T.axis.spatial(
                                            T.int64(128),
                                            k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(
                                            lv9_local[v0, v1],
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                        )
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = (
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                            + var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ]
                                            * lv9_local[v0, v1]
                                        )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2 in T.vectorized(T.int64(4)):
                                with T.block("var_matmul_intermediate_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2 = T.axis.spatial(
                                            T.int64(4096),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax2,
                                    )
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2[v_i0, v_i1, v_i2])
                                    T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2[v_i0, v_i1, v_i2]


@T.prim_func(private=True)
def fused_decode4_fused_matmul6_add4(
    lv1363: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv1364: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    lv2067: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    linear_bias192: T.Buffer((T.int64(2560),), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1363[v_i // T.int64(8), v_j], lv1364[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1363[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1364[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2067[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv2067[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias192[v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias192[v_ax2]
            )


def sch_fused_decode4_fused_matmul6_add4(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[160, 8, 2]
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


@T.prim_func(private=True)
def fused_decode6_fused_matmul9_add7_cast8_cast12_add5(
    lv1393: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv1394: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    lv2121: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"),
    linear_bias197: T.Buffer((T.int64(2560),), "float32"),
    lv329: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    var_compute_intermediate_1 = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1393[v_i // T.int64(8), v_j], lv1394[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1393[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1394[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2121[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv2121[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias197[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias197[v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_add_intermediate[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[
                v_i0, v_i1, v_i2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2],
                lv329[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2]
                + lv329[v_ax0, v_ax1, v_ax2]
            )


def sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[640, 2, 8]
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
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 256, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func(private=True)
def fused_decode5_fused_matmul8_add6_gelu1_cast11(
    lv1387: T.Buffer((T.int64(320), T.int64(10240)), "uint32"),
    lv1388: T.Buffer((T.int64(80), T.int64(10240)), "float16"),
    lv2115: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    linear_bias196: T.Buffer((T.int64(10240),), "float32"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(10240)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_multiply = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
    var_T_multiply_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(10240))
    )
    for i, j in T.grid(T.int64(2560), T.int64(10240)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1387[v_i // T.int64(8), v_j], lv1388[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1387[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1388[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(10240), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2115[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv2115[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias196[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias196[v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[
                v_ax0, v_ax1, v_ax2
            ] * T.float32(0.70710678118654757)
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
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[
                v_ax0, v_ax1, v_ax2
            ] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = (
                T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2]
            )
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(10240)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_multiply_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_multiply_intermediate[v_i0, v_i1, v_i2]
            )


def sch_fused_decode5_fused_matmul8_add6_gelu1_cast11(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 4]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[80, 4, 8]
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
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 256, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func(private=True)
def fused_decode4_fused_matmul6_add4_add5(
    lv1381: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv1382: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    lv328: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    linear_bias195: T.Buffer((T.int64(2560),), "float16"),
    lv2062: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    var_T_add_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1381[v_i // T.int64(8), v_j], lv1382[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1381[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1382[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv328[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv328[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias195[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias195[v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2], lv2062[v_ax0, v_ax1, v_ax2]
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2]
                + lv2062[v_ax0, v_ax1, v_ax2]
            )


def sch_fused_decode4_fused_matmul6_add4_add5(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[10, 256, 1]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[160, 8, 2]
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
    bb4 = sch.get_block(name="T_add_1", func_name="main")
    sch.compute_inline(block=b28)
    sch.reverse_compute_inline(block=bb4)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 256, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func(private=True)
def fused_decode3_matmul3(
    lv2515: T.Buffer((T.int64(320), T.int64(50432)), "uint32"),
    lv2516: T.Buffer((T.int64(80), T.int64(50432)), "float32"),
    lv705: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(50432)), "float32"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(50432)))
    for i, j in T.grid(T.int64(2560), T.int64(50432)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2515[v_i // T.int64(8), v_j], lv2516[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float32",
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv2515[v_i // T.int64(8), v_j],
                                T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7),
                )
                * lv2516[v_i // T.int64(32), v_j]
            )
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50432), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv705[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv705[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )


def sch_fused_decode3_matmul3(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[197, 128, 2]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[80, 4, 8]
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
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch")
    l29, l30, l31, l32, l33 = sch.get_loops(block=b20)
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 128, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func(private=True)
def fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(
    lv2509: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv2510: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    lv4105: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"),
    linear_bias383: T.Buffer((T.int64(2560),), "float32"),
    lv701: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float32"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    var_compute_intermediate_1 = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    var_T_add_intermediate_1 = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2509[v_i // T.int64(8), v_j], lv2510[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv2509[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv2510[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv4105[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv4105[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_decode_intermediate[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias383[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias383[v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_add_intermediate[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[
                v_i0, v_i1, v_i2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2],
                lv701[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = (
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2]
                + lv701[v_ax0, v_ax1, v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
        with T.block("compute_2"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2]
            )


def sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[5, 256, 2]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[320, 4, 8]
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
    l34, l35, l36 = sch.split(
        loop=l33, factors=[None, 256, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l36)
    sch.bind(loop=l35, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func(private=True)
def fused_decode2_fused_NT_matmul3_add6_gelu1_cast11(
    lv36: T.Buffer((T.int64(320), T.int64(10240)), "uint32"),
    lv37: T.Buffer((T.int64(80), T.int64(10240)), "float16"),
    p_lv57: T.handle,
    linear_bias4: T.Buffer((T.int64(10240),), "float32"),
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv57 = T.match_buffer(p_lv57, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(10240)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(10240), T.int64(2560)), "float16"
    )
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
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv57[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias4[v_ax2]
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias4[v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[
                v_ax0, v_ax1, v_ax2
            ] * T.float32(0.70710678118654757)
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
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[
                v_ax0, v_ax1, v_ax2
            ] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = (
                T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2]
            )
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(10240)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_multiply_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_multiply_intermediate[v_i0, v_i1, v_i2]
            )


@T.prim_func(private=True)
def fused_decode2_fused_NT_matmul3_add6_gelu1_cast11_after(
    lv36: T.Buffer((T.int64(320), T.int64(10240)), "uint32"),
    lv37: T.Buffer((T.int64(80), T.int64(10240)), "float16"),
    p_lv57: T.handle,
    linear_bias4: T.Buffer((T.int64(10240),), "float32"),
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.noalias": T.bool(True)})
    n = T.int64()
    lv57 = T.match_buffer(p_lv57, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(10240)), "float16"
    )
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer(
            (T.int64(2560), T.int64(10240)), "float16", scope="local"
        )
        lv36_local = T.alloc_buffer(
            (T.int64(320), T.int64(10240)), "uint32", scope="local"
        )
        lv37_local = T.alloc_buffer(
            (T.int64(80), T.int64(10240)), "float16", scope="local"
        )
        lv57_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            "float16",
            scope="local",
        )
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
            (
                T.int64(1),
                (n + T.int64(31)) // T.int64(32) * T.int64(32),
                T.int64(10240),
            ),
            scope="local",
        )
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
            (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
        ):
            for i2_0 in T.thread_binding(T.int64(80), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(
                    T.int64(8), thread="threadIdx.y"
                ):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + i0_i1_fused_1_2_init,
                                    )
                                    v_i2 = T.axis.spatial(
                                        T.int64(10240),
                                        i2_0 * T.int64(128)
                                        + i2_1 * T.int64(8)
                                        + i2_2_init,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_NT_matmul_intermediate_pad_local[
                                            v_i0, v_i1, v_i2
                                        ]
                                    )
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv37_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(80),
                                            k_0_0 * T.int64(4) + k_0_1 + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(10240),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv37[v0, v1])
                                        T.writes(lv37_local[v0, v1])
                                        lv37_local[v0, v1] = lv37[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv36_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(320),
                                                k_0_0 * T.int64(16)
                                                + k_0_1 * T.int64(4)
                                                + k_1
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(10240),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(lv36[v0, v1])
                                            T.writes(lv36_local[v0, v1])
                                            lv36_local[v0, v1] = lv36[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax0,
                                                )
                                                v_j = T.axis.spatial(
                                                    T.int64(10240),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + ax1,
                                                )
                                                T.reads(
                                                    lv36_local[v_i // T.int64(8), v_j],
                                                    lv37_local[v_i // T.int64(32), v_j],
                                                )
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv36_local[
                                                                    v_i // T.int64(8),
                                                                    v_j,
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
                                                ) * lv37_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv57_pad_local"):
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
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax2,
                                                )
                                                T.reads(lv57[v0, v1, v2])
                                                T.writes(lv57_pad_local[v0, v1, v2])
                                                lv57_pad_local[
                                                    v0, v1, v2
                                                ] = T.if_then_else(
                                                    v1 < n,
                                                    lv57[v0, v1, v2],
                                                    T.float16(0),
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
                                                    T.int64(10240),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + i2_2,
                                                )
                                                v_k = T.axis.reduce(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2,
                                                )
                                                T.reads(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ],
                                                    lv57_pad_local[v_i0, v_i1, v_k],
                                                    decode_local[v_k, v_i2],
                                                )
                                                T.writes(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ]
                                                )
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] = var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] + T.Cast(
                                                    "float32",
                                                    lv57_pad_local[v_i0, v_i1, v_k],
                                                ) * T.Cast(
                                                    "float32", decode_local[v_k, v_i2]
                                                )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + ax1,
                                    )
                                    v2 = T.axis.spatial(
                                        T.int64(10240),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                    )
                                    T.reads(
                                        var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ],
                                        linear_bias4[v2],
                                    )
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = T.Cast(
                                            "float16",
                                            (
                                                var_NT_matmul_intermediate_pad_local[
                                                    v0, v1, v2
                                                ]
                                                + linear_bias4[v2]
                                            )
                                            * (
                                                T.float32(0.5)
                                                + T.erf(
                                                    (
                                                        var_NT_matmul_intermediate_pad_local[
                                                            v0, v1, v2
                                                        ]
                                                        + linear_bias4[v2]
                                                    )
                                                    * T.float32(0.70710678118654757)
                                                )
                                                * T.float32(0.5)
                                            ),
                                        )


@T.prim_func(private=True)
def fused_decode1_fused_NT_matmul1_add4(
    lv8: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv9: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    p_lv9: T.handle,
    linear_bias: T.Buffer((T.int64(2560),), "float16"),
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv9_1 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(2560), T.int64(2560)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
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
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv9_1[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias[v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias[v_ax2]
            )


@T.prim_func(private=True)
def fused_decode1_fused_NT_matmul1_add4_after(
    lv8: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv9: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    p_lv9: T.handle,
    linear_bias: T.Buffer((T.int64(2560),), "float16"),
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.noalias": T.bool(True)})
    n = T.int64()
    lv9_1 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer(
            (T.int64(2560), T.int64(2560)), "float16", scope="local"
        )
        lv8_local = T.alloc_buffer(
            (T.int64(320), T.int64(2560)), "uint32", scope="local"
        )
        lv9_local = T.alloc_buffer(
            (T.int64(80), T.int64(2560)), "float16", scope="local"
        )
        lv9_1_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            "float16",
            scope="local",
        )
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            "float16",
            scope="local",
        )
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
            (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
        ):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(
                    T.int64(8), thread="threadIdx.y"
                ):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + i0_i1_fused_1_2_init,
                                    )
                                    v_i2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128)
                                        + i2_1 * T.int64(8)
                                        + i2_2_init,
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
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(80),
                                            k_0_0 * T.int64(4) + k_0_1 + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(2560),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(320),
                                                k_0_0 * T.int64(16)
                                                + k_0_1 * T.int64(4)
                                                + k_1
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(2560),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax0,
                                                )
                                                v_j = T.axis.spatial(
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + ax1,
                                                )
                                                T.reads(
                                                    lv8_local[v_i // T.int64(8), v_j],
                                                    lv9_local[v_i // T.int64(32), v_j],
                                                )
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv8_local[
                                                                    v_i // T.int64(8),
                                                                    v_j,
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
                                                ) * lv9_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv9_1_pad_local"):
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
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax2,
                                                )
                                                T.reads(lv9_1[v0, v1, v2])
                                                T.writes(lv9_1_pad_local[v0, v1, v2])
                                                lv9_1_pad_local[
                                                    v0, v1, v2
                                                ] = T.if_then_else(
                                                    v1 < n,
                                                    lv9_1[v0, v1, v2],
                                                    T.float16(0),
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
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + i2_2,
                                                )
                                                v_k = T.axis.reduce(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2,
                                                )
                                                T.reads(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ],
                                                    lv9_1_pad_local[v_i0, v_i1, v_k],
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
                                                    + lv9_1_pad_local[v_i0, v_i1, v_k]
                                                    * decode_local[v_k, v_i2]
                                                )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + ax1,
                                    )
                                    v2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                    )
                                    T.reads(
                                        var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ],
                                        linear_bias[v2],
                                    )
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = (
                                            var_NT_matmul_intermediate_pad_local[
                                                v0, v1, v2
                                            ]
                                            + linear_bias[v2]
                                        )


@T.prim_func(private=True)
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5(
    lv43: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv44: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    p_lv63: T.handle,
    linear_bias5: T.Buffer((T.int64(2560),), "float32"),
    p_lv7: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv63 = T.match_buffer(p_lv63, (T.int64(1), n, T.int64(10240)), "float16")
    lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(2560), T.int64(10240)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer(
        (T.int64(1), n, T.int64(2560)), "float16"
    )
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
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
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv63[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias5[v_ax2]
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias5[v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_add_intermediate[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[
                v_i0, v_i1, v_i2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2],
                lv7[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2]
                + lv7[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func(private=True)
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_after(
    lv43: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv44: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    p_lv63: T.handle,
    linear_bias5: T.Buffer((T.int64(2560),), "float32"),
    p_lv7: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.noalias": T.bool(True)})
    n = T.int64()
    lv63 = T.match_buffer(p_lv63, (T.int64(1), n, T.int64(10240)), "float16")
    lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer(
            (T.int64(10240), T.int64(2560)), "float16", scope="local"
        )
        lv43_local = T.alloc_buffer(
            (T.int64(1280), T.int64(2560)), "uint32", scope="local"
        )
        lv44_local = T.alloc_buffer(
            (T.int64(320), T.int64(2560)), "float16", scope="local"
        )
        lv63_pad_local = T.alloc_buffer(
            (
                T.int64(1),
                (n + T.int64(31)) // T.int64(32) * T.int64(32),
                T.int64(10240),
            ),
            "float16",
            scope="local",
        )
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            scope="local",
        )
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
            (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
        ):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(
                    T.int64(8), thread="threadIdx.y"
                ):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + i0_i1_fused_1_2_init,
                                    )
                                    v_i2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128)
                                        + i2_1 * T.int64(8)
                                        + i2_2_init,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_NT_matmul_intermediate_pad_local[
                                            v_i0, v_i1, v_i2
                                        ]
                                    )
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(80), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv44_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(320),
                                            k_0_0 * T.int64(4) + k_0_1 + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(2560),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv44[v0, v1])
                                        T.writes(lv44_local[v0, v1])
                                        lv44_local[v0, v1] = lv44[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv43_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(1280),
                                                k_0_0 * T.int64(16)
                                                + k_0_1 * T.int64(4)
                                                + k_1
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(2560),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(lv43[v0, v1])
                                            T.writes(lv43_local[v0, v1])
                                            lv43_local[v0, v1] = lv43[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax0,
                                                )
                                                v_j = T.axis.spatial(
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + ax1,
                                                )
                                                T.reads(
                                                    lv43_local[v_i // T.int64(8), v_j],
                                                    lv44_local[v_i // T.int64(32), v_j],
                                                )
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv43_local[
                                                                    v_i // T.int64(8),
                                                                    v_j,
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
                                                ) * lv44_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv63_pad_local"):
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
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax2,
                                                )
                                                T.reads(lv63[v0, v1, v2])
                                                T.writes(lv63_pad_local[v0, v1, v2])
                                                lv63_pad_local[
                                                    v0, v1, v2
                                                ] = T.if_then_else(
                                                    v1 < n,
                                                    lv63[v0, v1, v2],
                                                    T.float16(0),
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
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + i2_2,
                                                )
                                                v_k = T.axis.reduce(
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2,
                                                )
                                                T.reads(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ],
                                                    lv63_pad_local[v_i0, v_i1, v_k],
                                                    decode_local[v_k, v_i2],
                                                )
                                                T.writes(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ]
                                                )
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] = var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] + T.Cast(
                                                    "float32",
                                                    lv63_pad_local[v_i0, v_i1, v_k],
                                                ) * T.Cast(
                                                    "float32", decode_local[v_k, v_i2]
                                                )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + ax1,
                                    )
                                    v2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                    )
                                    T.reads(
                                        var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ],
                                        linear_bias5[v2],
                                        lv7[v0, v1, v2],
                                    )
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = (
                                            T.Cast(
                                                "float16",
                                                var_NT_matmul_intermediate_pad_local[
                                                    v0, v1, v2
                                                ]
                                                + linear_bias5[v2],
                                            )
                                            + lv7[v0, v1, v2]
                                        )


@T.prim_func(private=True)
def fused_decode1_fused_NT_matmul1_add4_add5(
    lv29: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv30: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    p_lv49: T.handle,
    linear_bias3: T.Buffer((T.int64(2560),), "float16"),
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(2560), T.int64(2560)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(2560)), "float16"
    )
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(2560), T.int64(2560)):
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
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv49[v_i0, v_i1, v_k] * var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias3[v_ax2]
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias3[v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2], lv2[v_ax0, v_ax1, v_ax2]
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] + lv2[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func(private=True)
def fused_decode1_fused_NT_matmul1_add4_add5_after(
    lv29: T.Buffer((T.int64(320), T.int64(2560)), "uint32"),
    lv30: T.Buffer((T.int64(80), T.int64(2560)), "float16"),
    p_lv49: T.handle,
    linear_bias3: T.Buffer((T.int64(2560),), "float16"),
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(2560)), "float16"
    )
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer(
            (T.int64(2560), T.int64(2560)), "float16", scope="local"
        )
        lv29_local = T.alloc_buffer(
            (T.int64(320), T.int64(2560)), "uint32", scope="local"
        )
        lv30_local = T.alloc_buffer(
            (T.int64(80), T.int64(2560)), "float16", scope="local"
        )
        lv49_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            "float16",
            scope="local",
        )
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            "float16",
            scope="local",
        )
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
            (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
        ):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(
                    T.int64(8), thread="threadIdx.y"
                ):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + i0_i1_fused_1_2_init,
                                    )
                                    v_i2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128)
                                        + i2_1 * T.int64(8)
                                        + i2_2_init,
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
                        for k_0_0, k_0_1 in T.grid(T.int64(20), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv30_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(80),
                                            k_0_0 * T.int64(4) + k_0_1 + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(2560),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv30[v0, v1])
                                        T.writes(lv30_local[v0, v1])
                                        lv30_local[v0, v1] = lv30[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv29_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(320),
                                                k_0_0 * T.int64(16)
                                                + k_0_1 * T.int64(4)
                                                + k_1
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(2560),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(lv29[v0, v1])
                                            T.writes(lv29_local[v0, v1])
                                            lv29_local[v0, v1] = lv29[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax0,
                                                )
                                                v_j = T.axis.spatial(
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + ax1,
                                                )
                                                T.reads(
                                                    lv29_local[v_i // T.int64(8), v_j],
                                                    lv30_local[v_i // T.int64(32), v_j],
                                                )
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv29_local[
                                                                    v_i // T.int64(8),
                                                                    v_j,
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
                                                ) * lv30_local[v_i // T.int64(32), v_j]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv49_pad_local"):
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
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax2,
                                                )
                                                T.reads(lv49[v0, v1, v2])
                                                T.writes(lv49_pad_local[v0, v1, v2])
                                                lv49_pad_local[
                                                    v0, v1, v2
                                                ] = T.if_then_else(
                                                    v1 < n,
                                                    lv49[v0, v1, v2],
                                                    T.float16(0),
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
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + i2_2,
                                                )
                                                v_k = T.axis.reduce(
                                                    T.int64(2560),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2,
                                                )
                                                T.reads(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ],
                                                    lv49_pad_local[v_i0, v_i1, v_k],
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
                                                    + lv49_pad_local[v_i0, v_i1, v_k]
                                                    * decode_local[v_k, v_i2]
                                                )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + ax1,
                                    )
                                    v2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                    )
                                    T.reads(
                                        var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ],
                                        linear_bias3[v2],
                                        lv2[v0, v1, v2],
                                    )
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = (
                                            var_NT_matmul_intermediate_pad_local[
                                                v0, v1, v2
                                            ]
                                            + linear_bias3[v2]
                                            + lv2[v0, v1, v2]
                                        )


@T.prim_func(private=True)
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7(
    lv1345: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv1346: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    p_lv2047: T.handle,
    linear_bias191: T.Buffer((T.int64(2560),), "float32"),
    p_lv317: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv2047 = T.match_buffer(p_lv2047, (T.int64(1), n, T.int64(10240)), "float16")
    lv317 = T.match_buffer(p_lv317, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
    var_T_transpose_intermediate = T.alloc_buffer(
        (T.int64(2560), T.int64(10240)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    var_compute_intermediate_1 = T.alloc_buffer(
        (T.int64(1), n, T.int64(2560)), "float16"
    )
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
    for i, j in T.grid(T.int64(10240), T.int64(2560)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1345[v_i // T.int64(8), v_j], lv1346[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1345[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1346[v_i // T.int64(32), v_j]
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
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[
                v_i0, v_i1, v_i2
            ] + T.Cast("float32", lv2047[v_i0, v_i1, v_k]) * T.Cast(
                "float32", var_T_transpose_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias191[v_ax2]
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias191[v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float16", var_T_add_intermediate[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
            var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[
                v_i0, v_i1, v_i2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2],
                lv317[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = (
                var_compute_intermediate_1[v_ax0, v_ax1, v_ax2]
                + lv317[v_ax0, v_ax1, v_ax2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
        with T.block("compute_2"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2]
            )


@T.prim_func(private=True)
def fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7_after(
    lv1345: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"),
    lv1346: T.Buffer((T.int64(320), T.int64(2560)), "float16"),
    p_lv2047: T.handle,
    linear_bias191: T.Buffer((T.int64(2560),), "float32"),
    p_lv317: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.noalias": T.bool(True)})
    n = T.int64()
    lv2047 = T.match_buffer(p_lv2047, (T.int64(1), n, T.int64(10240)), "float16")
    lv317 = T.match_buffer(p_lv317, (T.int64(1), n, T.int64(2560)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.thread_extent_low_inclusive": 32})
        decode_local = T.alloc_buffer(
            (T.int64(10240), T.int64(2560)), "float16", scope="local"
        )
        lv1345_local = T.alloc_buffer(
            (T.int64(1280), T.int64(2560)), "uint32", scope="local"
        )
        lv1346_local = T.alloc_buffer(
            (T.int64(320), T.int64(2560)), "float16", scope="local"
        )
        lv2047_pad_local = T.alloc_buffer(
            (
                T.int64(1),
                (n + T.int64(31)) // T.int64(32) * T.int64(32),
                T.int64(10240),
            ),
            "float16",
            scope="local",
        )
        var_NT_matmul_intermediate_pad_local = T.alloc_buffer(
            (T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2560)),
            scope="local",
        )
        for i0_i1_fused_0_i0_i1_fused_1_0_fused in T.thread_binding(
            (n + T.int64(31)) // T.int64(32), thread="blockIdx.y"
        ):
            for i2_0 in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for i0_i1_fused_1_1 in T.thread_binding(
                    T.int64(8), thread="threadIdx.y"
                ):
                    for i2_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for i0_i1_fused_1_2_init in range(T.int64(4)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("NT_matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + i0_i1_fused_1_2_init,
                                    )
                                    v_i2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128)
                                        + i2_1 * T.int64(8)
                                        + i2_2_init,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_NT_matmul_intermediate_pad_local[
                                            v_i0, v_i1, v_i2
                                        ]
                                    )
                                    var_NT_matmul_intermediate_pad_local[
                                        v_i0, v_i1, v_i2
                                    ] = T.float32(0)
                        for k_0_0, k_0_1 in T.grid(T.int64(80), T.int64(4)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv1346_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(320),
                                            k_0_0 * T.int64(4) + k_0_1 + ax0,
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(2560),
                                            i2_0 * T.int64(128)
                                            + i2_1 * T.int64(8)
                                            + ax1,
                                        )
                                        T.reads(lv1346[v0, v1])
                                        T.writes(lv1346_local[v0, v1])
                                        lv1346_local[v0, v1] = lv1346[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv1345_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(1280),
                                                k_0_0 * T.int64(16)
                                                + k_0_1 * T.int64(4)
                                                + k_1
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(2560),
                                                i2_0 * T.int64(128)
                                                + i2_1 * T.int64(8)
                                                + ax1,
                                            )
                                            T.reads(lv1345[v0, v1])
                                            T.writes(lv1345_local[v0, v1])
                                            lv1345_local[v0, v1] = lv1345[v0, v1]
                                for k_2 in range(T.int64(8)):
                                    for ax0 in range(T.int64(1)):
                                        for ax1 in T.vectorized(T.int64(8)):
                                            with T.block("decode"):
                                                v_i = T.axis.spatial(
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax0,
                                                )
                                                v_j = T.axis.spatial(
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + ax1,
                                                )
                                                T.reads(
                                                    lv1345_local[
                                                        v_i // T.int64(8), v_j
                                                    ],
                                                    lv1346_local[
                                                        v_i // T.int64(32), v_j
                                                    ],
                                                )
                                                T.writes(decode_local[v_i, v_j])
                                                decode_local[v_i, v_j] = (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv1345_local[
                                                                    v_i // T.int64(8),
                                                                    v_j,
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
                                                ) * lv1346_local[
                                                    v_i // T.int64(32), v_j
                                                ]
                                    for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                                        for ax2 in T.vectorized(T.int64(1)):
                                            with T.block("lv2047_pad_local"):
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
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2
                                                    + ax2,
                                                )
                                                T.reads(lv2047[v0, v1, v2])
                                                T.writes(lv2047_pad_local[v0, v1, v2])
                                                lv2047_pad_local[
                                                    v0, v1, v2
                                                ] = T.if_then_else(
                                                    v1 < n,
                                                    lv2047[v0, v1, v2],
                                                    T.float16(0),
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
                                                    T.int64(2560),
                                                    i2_0 * T.int64(128)
                                                    + i2_1 * T.int64(8)
                                                    + i2_2,
                                                )
                                                v_k = T.axis.reduce(
                                                    T.int64(10240),
                                                    k_0_0 * T.int64(128)
                                                    + k_0_1 * T.int64(32)
                                                    + k_1 * T.int64(8)
                                                    + k_2,
                                                )
                                                T.reads(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ],
                                                    lv2047_pad_local[v_i0, v_i1, v_k],
                                                    decode_local[v_k, v_i2],
                                                )
                                                T.writes(
                                                    var_NT_matmul_intermediate_pad_local[
                                                        v_i0, v_i1, v_i2
                                                    ]
                                                )
                                                var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] = var_NT_matmul_intermediate_pad_local[
                                                    v_i0, v_i1, v_i2
                                                ] + T.Cast(
                                                    "float32",
                                                    lv2047_pad_local[v_i0, v_i1, v_k],
                                                ) * T.Cast(
                                                    "float32", decode_local[v_k, v_i2]
                                                )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(4)):
                            for ax2 in T.vectorized(T.int64(8)):
                                with T.block("var_NT_matmul_intermediate_pad_local"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(
                                        (n + T.int64(31)) // T.int64(32) * T.int64(32),
                                        i0_i1_fused_0_i0_i1_fused_1_0_fused
                                        * T.int64(32)
                                        + i0_i1_fused_1_1 * T.int64(4)
                                        + ax1,
                                    )
                                    v2 = T.axis.spatial(
                                        T.int64(2560),
                                        i2_0 * T.int64(128) + i2_1 * T.int64(8) + ax2,
                                    )
                                    T.reads(
                                        var_NT_matmul_intermediate_pad_local[
                                            v0, v1, v2
                                        ],
                                        linear_bias191[v2],
                                        lv317[v0, v1, v2],
                                    )
                                    T.writes(p_output0_intermediate[v0, v1, v2])
                                    if v1 < n:
                                        p_output0_intermediate[v0, v1, v2] = T.Cast(
                                            "float32",
                                            T.Cast(
                                                "float16",
                                                var_NT_matmul_intermediate_pad_local[
                                                    v0, v1, v2
                                                ]
                                                + linear_bias191[v2],
                                            )
                                            + lv317[v0, v1, v2],
                                        )


@T.prim_func(private=True)
def fused_decode2_NT_matmul(
    lv4: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    lv5: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    p_lv6: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(12288)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4[v_i // T.int64(8), v_j], lv5[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv4[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv5[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate[v_ax0, v_ax1])
            p_output0_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv6[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv6[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func(private=True)
def fused_decode2_NT_matmul_after(
    lv8: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    lv9: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    p_lv6: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv6 = T.match_buffer(p_lv6, (1, n, 4096), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (1, n, 12288), "float16")

    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(12288)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(12288)), "float16", scope="local"
    )
    lv8_local = T.alloc_buffer((T.int64(512), T.int64(12288)), "uint32", scope="local")
    lv9_local = T.alloc_buffer(
        (T.int64(128), T.int64(12288)), "float16", scope="local"
    )
    #lv6_shared = T.alloc_buffer(
    #    (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    #)
    for i0_i1_i2_fused_n in T.thread_binding(((n+7)//8), thread="blockIdx.y"):
        for i0_i1_i2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    with T.block("n_check"):
                        T.where((i0_i1_i2_fused_n * T.int64(8) + ax2_y) < n)
                        for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                v_i2 = T.axis.spatial(
                                    T.int64(12288),
                                    i0_i1_i2_fused_0 * T.int64(128)
                                    + i0_i1_i2_fused_1 * T.int64(4)
                                    + i0_i1_i2_fused_2_init
                                )
                                T.reads()
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_1 in range(T.int64(128)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("matmul_init_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2k = T.axis.spatial(
                                        T.int64(12288),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                    )
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ] = T.float16(0)
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(128), k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(12288),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_2 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(4)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(512),
                                                k_1 * T.int64(4)
                                                + k_2
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(12288),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                            )
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_3 in range(T.int64(8)):
                                    for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                            v_i2 = T.axis.spatial(
                                                T.int64(12288),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(4096),
                                                k_1 * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            T.reads(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv6[v_i0, v_i1, v_k],
                                                lv8_local[v_k // T.int64(8), v_i2],
                                            )
                                            T.writes(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] = var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] + lv6[
                                                v_i0, v_i1, v_k
                                            ] * (
                                                (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv8_local[
                                                                    v_k // T.int64(8), v_i2
                                                                ],
                                                                T.Cast(
                                                                    "uint32",
                                                                    v_k % T.int64(8),
                                                                )
                                                                * T.uint32(4),
                                                            ),
                                                            T.uint32(15),
                                                        ),
                                                    )
                                                    - T.float16(7)
                                                )
                                            )
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("multiple_scale"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                        v_i2 = T.axis.spatial(
                                                T.int64(12288),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                        )
                                        v0 = T.axis.spatial(
                                            T.int64(128),
                                            k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(12288),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(
                                            lv9_local[v0, v1],
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                        )
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = (
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                            + var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ]
                                            * lv9_local[v0, v1]
                                        )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2 in T.vectorized(T.int64(4)):
                                with T.block("var_matmul_intermediate_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2 = T.axis.spatial(
                                            T.int64(12288),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax2,
                                    )
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                    T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2]


@T.prim_func(private=True)
def fused_decode4_NT_matmul3(
    lv13: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    lv14: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    p_lv45: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv45 = T.match_buffer(p_lv45, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(22016)), "float16"
    )
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv13[v_i // T.int64(8), v_j], lv14[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv13[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv14[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate[v_ax0, v_ax1])
            p_output0_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv45[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv45[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func(private=True)
def fused_decode4_NT_matmul3_after(
    lv8: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    lv9: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    p_lv6: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv6 = T.match_buffer(p_lv6, (1, n, 4096), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (1, n, 22016), "float16")

    var_matmul_intermediate_local = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(22016)), "float16", scope="local"
    )
    var_matmul_intermediate_local_batch = T.alloc_buffer(
        (T.int64(1), ((n+7)//8) * 8, T.int64(22016)), "float16", scope="local"
    )
    lv8_local = T.alloc_buffer((T.int64(512), T.int64(22016)), "uint32", scope="local")
    lv9_local = T.alloc_buffer(
        (T.int64(128), T.int64(22016)), "float16", scope="local"
    )
    #lv6_shared = T.alloc_buffer(
    #    (T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared"
    #)
    for i0_i1_i2_fused_n in T.thread_binding(((n+7)//8), thread="blockIdx.y"):
        for i0_i1_i2_fused_0 in T.thread_binding(T.int64(172), thread="blockIdx.x"):
            for i0_i1_i2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax2_y in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    with T.block("n_check"):
                        T.where((i0_i1_i2_fused_n * T.int64(8) + ax2_y) < n)
                        for i0_i1_i2_fused_2_init in T.vectorized(T.int64(4)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                v_i2 = T.axis.spatial(
                                    T.int64(22016),
                                    i0_i1_i2_fused_0 * T.int64(128)
                                    + i0_i1_i2_fused_1 * T.int64(4)
                                    + i0_i1_i2_fused_2_init
                                )
                                T.reads()
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_1 in range(T.int64(128)):
                            for ax1 in T.vectorized(T.int64(4)):
                                with T.block("matmul_init_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2k = T.axis.spatial(
                                        T.int64(22016),
                                        i0_i1_i2_fused_0 * T.int64(128)
                                        + i0_i1_i2_fused_1 * T.int64(4)
                                        + ax1,
                                    )
                                    T.reads()
                                    T.writes(
                                        var_matmul_intermediate_local_batch[
                                            v_i0, v_i1, v_i2k
                                        ]
                                    )
                                    var_matmul_intermediate_local_batch[
                                        v_i0, v_i1, v_i2k
                                    ] = T.float16(0)
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("lv9_local"):
                                        v0 = T.axis.spatial(
                                            T.int64(128), k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(lv9[v0, v1])
                                        T.writes(lv9_local[v0, v1])
                                        lv9_local[v0, v1] = lv9[v0, v1]
                            for k_2 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(4)):
                                        with T.block("lv8_local"):
                                            v0 = T.axis.spatial(
                                                T.int64(512),
                                                k_1 * T.int64(4)
                                                + k_2
                                                + ax0,
                                            )
                                            v1 = T.axis.spatial(
                                                T.int64(22016),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                            )
                                            T.reads(lv8[v0, v1])
                                            T.writes(lv8_local[v0, v1])
                                            lv8_local[v0, v1] = lv8[v0, v1]
                                for k_3 in range(T.int64(8)):
                                    for i0_i1_i2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                            v_i2 = T.axis.spatial(
                                                T.int64(22016),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + i0_i1_i2_fused_2,
                                            )
                                            v_k = T.axis.reduce(
                                                T.int64(4096),
                                                k_1 * T.int64(32)
                                                + k_2 * T.int64(8)
                                                + k_3,
                                            )
                                            T.reads(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ],
                                                lv6[v_i0, v_i1, v_k],
                                                lv8_local[v_k // T.int64(8), v_i2],
                                            )
                                            T.writes(
                                                var_matmul_intermediate_local_batch[
                                                    v_i0, v_i1, v_i2
                                                ]
                                            )
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] = var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ] + lv6[
                                                v_i0, v_i1, v_k
                                            ] * (
                                                (
                                                    T.Cast(
                                                        "float16",
                                                        T.bitwise_and(
                                                            T.shift_right(
                                                                lv8_local[
                                                                    v_k // T.int64(8), v_i2
                                                                ],
                                                                T.Cast(
                                                                    "uint32",
                                                                    v_k % T.int64(8),
                                                                )
                                                                * T.uint32(4),
                                                            ),
                                                            T.uint32(15),
                                                        ),
                                                    )
                                                    - T.float16(7)
                                                )
                                            )
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(4)):
                                    with T.block("multiple_scale"):
                                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                        v_i2 = T.axis.spatial(
                                                T.int64(22016),
                                                i0_i1_i2_fused_0 * T.int64(128)
                                                + i0_i1_i2_fused_1 * T.int64(4)
                                                + ax1,
                                        )
                                        v0 = T.axis.spatial(
                                            T.int64(128),
                                            k_1
                                        )
                                        v1 = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax1,
                                        )
                                        T.reads(
                                            lv9_local[v0, v1],
                                            var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ],
                                        )
                                        T.writes(
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                        )
                                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = (
                                            var_matmul_intermediate_local[v_i0, v_i1, v_i2]
                                            + var_matmul_intermediate_local_batch[
                                                v_i0, v_i1, v_i2
                                            ]
                                            * lv9_local[v0, v1]
                                        )
                        for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax2 in T.vectorized(T.int64(4)):
                                with T.block("var_matmul_intermediate_local"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(((n+7)//8) * 8, i0_i1_i2_fused_n * T.int64(8) + ax2_y)
                                    v_i2 = T.axis.spatial(
                                            T.int64(22016),
                                            i0_i1_i2_fused_0 * T.int64(128)
                                            + i0_i1_i2_fused_1 * T.int64(4)
                                            + ax2,
                                    )
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                    T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2]



@T.prim_func(private=True)
def fused_NT_matmul1_divide2_maximum1_minimum1_cast3(lv1593: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), p_lv1603: T.handle, p_lv1582: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv1603 = T.match_buffer(p_lv1603, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    lv1582 = T.match_buffer(p_lv1582, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv1593[v_i0, v_i2, v_i1, v_k], lv1603[v_i0, v_i3, v_i1, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1593[v_i0, v_i2, v_i1, v_k] * lv1603[v_i0, v_i3, v_i1, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_minimum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1582[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1582[v_ax0, T.int64(0), v_ax2, v_ax3])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

@T.prim_func(private=True)
def fused_NT_matmul1_divide2_maximum1_minimum1_cast3_after(
    lv1593: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"),
    p_lv1603: T.handle,
    p_lv1582: T.handle,
    p_output0: T.handle
):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    n = T.int64()
    lv1603 = T.match_buffer(p_lv1603, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    lv1582 = T.match_buffer(p_lv1582, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    var_matmul_intermediate_local = T.alloc_buffer(
        (1, ((n + 7) // 8) * 8, 4096), "float16", scope="local"
    )
    lv1593_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(1024)), "float16", scope="shared"
    )
    for i_by in T.thread_binding(T.int64((n + 7) // 8), thread="blockIdx.y"):
        for i_bx in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(n), i_by * T.int64(8) + i_ty)
                            v_i2 = T.axis.spatial(
                                T.int64(4096),
                                i_bx * T.int64(128)
                                + i_tx * T.int64(4)
                                + i_v8,
                            )
                            T.reads()
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                        with T.block("lv1593_shared"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32), i_bx)
                            v_i3 = T.axis.spatial(T.int64(128), i_tx * T.int64(4) + i_v8)
                            T.reads(lv1593[v_i0, v_i1, v_i2, v_i3])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i3])
                            lv1593_shared[v_i0, v_i1, v_i3] = lv1593[v_i0, v_i1, v_i2, v_i3]
                        with T.block("matmul_compute"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1_1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(n), i_by * T.int64(8) + i_ty)
                            v_i2 = T.axis.spatial(T.int64(32), i_bx)
                            v_i3 = T.axis.spatial(T.int64(128), i_tx * T.int64(4) + i_v8)
                            v_ik = T.axis.spatial(T.int64(4096), i_bx * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_by * T.int64(8) + i_ty < n)
                            T.reads(lv1593_shared[v_i0, v_i1_1, v_i3], lv1603[v_i0, v_i1, v_i2, v_i3])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_ik])
                            var_matmul_intermediate_local[v_i0, v_i1, v_ik] = var_matmul_intermediate_local[v_i0, v_i1, v_ik] + lv1603[v_i0, v_i1, v_i2, v_i3] * lv1593_shared[v_i0, v_i1_1, v_i3]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1_1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(n), i_by * T.int64(8) + i_ty)
                            v_ik = T.axis.spatial(T.int64(4096), i_bx * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_ik])
                            T.writes(lv1593_shared[v_i0, v_i1_1, v_i2])
                            lv1593_shared[v_i0, v_i1_1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_ik]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("reduction_1"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_tx < T.int64(16))
                            T.reads(lv1593_shared[v_i0, v_i1, v_i2])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i2])
                            lv1593_shared[v_i0, v_i1, v_i2] = lv1593_shared[v_i0, v_i1, v_i2] + lv1593_shared[v_i0, v_i1, v_i2 + T.int64(64)]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("reduction_2"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_tx < T.int64(8))
                            T.reads(lv1593_shared[v_i0, v_i1, v_i2])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i2])
                            lv1593_shared[v_i0, v_i1, v_i2] = lv1593_shared[v_i0, v_i1, v_i2] + lv1593_shared[v_i0, v_i1, v_i2 + T.int64(32)]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("reduction_3"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_tx < T.int64(4))
                            T.reads(lv1593_shared[v_i0, v_i1, v_i2])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i2])
                            lv1593_shared[v_i0, v_i1, v_i2] = lv1593_shared[v_i0, v_i1, v_i2] + lv1593_shared[v_i0, v_i1, v_i2 + T.int64(16)]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("reduction_4"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_tx < T.int64(2))
                            T.reads(lv1593_shared[v_i0, v_i1, v_i2])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i2])
                            lv1593_shared[v_i0, v_i1, v_i2] = lv1593_shared[v_i0, v_i1, v_i2] + lv1593_shared[v_i0, v_i1, v_i2 + T.int64(8)]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for i_v8 in T.vectorized(T.int64(4)):
                        with T.block("reduction_4"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                            T.where(i_tx < T.int64(1))
                            T.reads(lv1593_shared[v_i0, v_i1, v_i2])
                            T.writes(lv1593_shared[v_i0, v_i1, v_i2])
                            lv1593_shared[v_i0, v_i1, v_i2] = lv1593_shared[v_i0, v_i1, v_i2] + lv1593_shared[v_i0, v_i1, v_i2 + T.int64(4)]
            for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for ax0 in range(T.int64(1)):
                        with T.block("Output_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(32), i_bx)
                            v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i3 = T.axis.spatial(T.int64(n), i_by * T.int64(8) + i_ty)
                            v_ik = T.axis.spatial(T.int64(1024), i_ty * T.int64(128))
                            T.where(i_by * T.int64(8) + i_ty < n)
                            T.reads(lv1593_shared[v_i0, v_i2, v_ik])
                            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", T.min(T.max((lv1593_shared[v_i0, v_i2, v_ik] + lv1593_shared[v_i0, v_i2, v_ik + T.int64(1)]
                                                + lv1593_shared[v_i0, v_i2, v_ik + T.int64(2)] + lv1593_shared[v_i0, v_i2, v_ik + T.int64(3)])
                                                * T.float16(0.088397790055248615), T.float16(-65504)), lv1582[v_i0, T.int64(0), v_i2, v_i3]))



# [gx,gy, gz] [lx, ly, lz]

@T.prim_func(private=True)
def NT_matmul3(var_A: T.handle, var_B: T.handle, NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128), n):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_k, v_i2, v_i3], B[v_i0, v_i2, v_i1, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_k, v_i2, v_i3] * B[v_i0, v_i2, v_i1, v_k]

@T.prim_func(private=True)
def NT_matmul3_after(
    var_A: T.handle,
    var_B: T.handle,
    NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")
):

    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_matmul_intermediate_local = T.alloc_buffer(
        (1, 8, 4096), "float16", scope="local"
    )
    B_shared = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(1024)), "float16", scope="shared"
    )
    for i_bx in T.thread_binding(T.int64(32), thread="blockIdx.x"):
        for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i_v8 in T.vectorized(T.int64(4)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(8),  i_ty)
                        v_i2 = T.axis.spatial(
                            T.int64(4096),
                            i_bx * T.int64(128) + i_tx * T.int64(4)
                            + i_v8,
                        )
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float16(0)
                for ax0 in range((n+255)//256):
                    with T.block("B_shared"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(32), i_bx)
                        v_i2 = T.axis.spatial(((n+255)//256) * 256, ax0 * T.int64(256) + i_ty * T.int64(32) + i_tx)
                        v_i2k = T.axis.spatial(T.int64(256), i_ty * T.int64(32) + i_tx)
                        #T.where(ax0 * T.int64(256) + i_ty * T.int64(32) + i_tx < n)
                        T.reads(B[v_i0, v_i1, T.int64(0), v_i2])
                        T.writes(B_shared[v_i0, v_i1, v_i2k])
                        B_shared[v_i0, T.int64(0), v_i2k] = T.if_then_else(v_i2 < n, B[v_i0, v_i1, T.int64(0), v_i2], T.float16(0))
                    for ax1 in range(32):
                        #with T.block("n_check"):
                        #    T.where(ax0 * T.int64(256)  + ax1 * T.int64(8) + i_ty < n)
                        for i_v8 in T.vectorized(T.int64(4)):
                            with T.block("matmul_compute"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(((n+255)//256) * 256, ax0 * T.int64(256)  + ax1 * T.int64(8) + i_ty)
                                v_i1_1 = T.axis.spatial(T.int64(8), i_ty)
                                v_i2 = T.axis.spatial(T.int64(32), i_bx)
                                v_i3 = T.axis.spatial(T.int64(128), i_tx * T.int64(4) + i_v8)
                                v_ik = T.axis.spatial(T.int64(256), ax1 * T.int64(8) + i_ty)
                                v_ik1 = T.axis.spatial(T.int64(4096), i_bx * T.int64(128) + i_tx * T.int64(4) + i_v8)
                                T.reads(B_shared[v_i0, T.int64(0), v_ik], A[v_i0, v_i1, v_i2, v_i3])
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1_1, v_ik1])
                                var_matmul_intermediate_local[v_i0, v_i1_1, v_ik1] = var_matmul_intermediate_local[v_i0, v_i1_1, v_ik1] + T.if_then_else(v_i1 < n, A[v_i0, v_i1, v_i2, v_i3], T.float16(0))  * B_shared[v_i0, T.int64(0), v_ik]

        for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i_v8 in T.vectorized(T.int64(4)):
                    with T.block("matmul_update"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(8), i_ty)
                        v_i2 = T.axis.spatial(T.int64(4096), i_bx * T.int64(128) + i_tx * T.int64(4) + i_v8)
                        v_ik = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                        T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        T.writes(B_shared[v_i0, T.int64(0), v_ik])
                        B_shared[v_i0, T.int64(0), v_ik] = var_matmul_intermediate_local[v_i0, v_i1, v_i2]
        for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i_v8 in T.vectorized(T.int64(4)):
                    with T.block("reduction_1"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                        T.where(i_ty < T.int64(4))
                        T.reads(B_shared[v_i0, v_i1, v_i2])
                        T.writes(B_shared[v_i0, v_i1, v_i2])
                        B_shared[v_i0, v_i1, v_i2] = B_shared[v_i0, v_i1, v_i2] + B_shared[v_i0, v_i1, v_i2 + T.int64(512)]
        for i_tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i_ty in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                for i_v8 in T.vectorized(T.int64(4)):
                    with T.block("Output_update"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(T.int64(32), i_bx)
                        v_i3 = T.axis.spatial(T.int64(128), i_tx * T.int64(4) + i_v8)
                        v_ik = T.axis.spatial(T.int64(1024), i_ty * T.int64(128) + i_tx * T.int64(4) + i_v8)
                        T.where(i_ty < 1)
                        T.reads(B_shared[v_i0, v_i1, v_ik])
                        T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                        NT_matmul[v_i0, v_i1, v_i2, v_i3] = B_shared[v_i0, v_i1, v_ik] + B_shared[v_i0, v_i1, v_ik + T.int64(128)] + B_shared[v_i0, v_i1, v_ik + T.int64(256)] + B_shared[v_i0, v_i1, v_ik + T.int64(384)]

@T.prim_func(private=True)
def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

@T.prim_func(private=True)
def rms_norm_after(var_A: T.handle, B: T.Buffer((4096,), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    n = T.int32()
    A = T.match_buffer(var_A, (1, n, 4096), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (1, n, 4096), "float16")
    # with T.block("root"):
    Ared_temp_shared = T.alloc_buffer((1, n), scope="shared")
    Ared_temp_rf_local = T.alloc_buffer((64, 1, n), scope="local")
    for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
        for ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
            with T.block("Ared_temp_rf_init"):
                vax1_fused_1, v0 = T.axis.remap("SS", [ax1_fused_1, ax0_fused])
                T.reads()
                T.writes(Ared_temp_rf_local[vax1_fused_1, 0, v0])
                Ared_temp_rf_local[vax1_fused_1, 0, v0] = T.float32(0)
            for ax1_fused_0, u in T.grid(64, 1):
                with T.block("Ared_temp_rf_update"):
                    vax1_fused_1, v0, vax1_fused_0 = T.axis.remap("SSR", [ax1_fused_1, ax0_fused, ax1_fused_0])
                    T.reads(Ared_temp_rf_local[vax1_fused_1, 0, v0], A[0, v0, vax1_fused_0 * 64 + vax1_fused_1])
                    T.writes(Ared_temp_rf_local[vax1_fused_1, 0, v0])
                    Ared_temp_rf_local[vax1_fused_1, 0, v0] = Ared_temp_rf_local[vax1_fused_1, 0, v0] + T.Cast("float32", A[0, v0, vax1_fused_0 * 64 + vax1_fused_1]) * T.Cast("float32", A[0, v0, vax1_fused_0 * 64 + vax1_fused_1])
        for ax1_fused in range(1):
            for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("Ared_temp"):
                    vax1_fused_1, v0 = T.axis.remap("RS", [ax0, ax0_fused])
                    T.reads(Ared_temp_rf_local[vax1_fused_1, 0, v0])
                    T.writes(Ared_temp_shared[0, v0])
                    with T.init():
                        Ared_temp_shared[0, v0] = T.float32(0)
                    Ared_temp_shared[0, v0] = Ared_temp_shared[0, v0] + Ared_temp_rf_local[vax1_fused_1, 0, v0]
        for ax0_fused_0 in range(64):
            for ax0_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("rms_norm"):
                    v0 = T.axis.spatial(n, ax0_fused)
                    v1 = T.axis.spatial(4096, ax0_fused_0 * 64 + ax0_fused_1)
                    T.reads(B[v1], A[0, v0, v1], Ared_temp_shared[0, v0])
                    T.writes(rms_norm_1[0, v0, v1])
                    rms_norm_1[0, v0, v1] = T.Cast("float16", T.Cast("float32", B[v1]) * (T.Cast("float32", A[0, v0, v1]) / T.sqrt(Ared_temp_shared[0, v0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

@T.prim_func(private=True)
def slice(var_A: T.handle, slice_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, n - T.int64(1), v_k])
            T.writes(slice_1[v_i, v_j, v_k])
            slice_1[v_i, v_j, v_k] = A[v_i, n - T.int64(1), v_k]

@T.prim_func(private=True)
def slice_after(var_A: T.handle, slice_1: T.Buffer((1, 1, 4096), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    n = T.int32()
    A = T.match_buffer(var_A, (1, n, 4096), "float16")
    # with T.block("root"):
    for ax0_fused_0 in T.thread_binding(16, thread="blockIdx.x"):
        for ax0_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("slice"):
                v0 = T.axis.spatial(4096, ax0_fused_0 * 256 + ax0_fused_1)
                T.reads(A[0, n - 1, v0])
                T.writes(slice_1[0, 0, v0])
                slice_1[0, 0, v0] = A[0, n - 1, v0]

@T.prim_func(private=True)
def NT_matmul2(var_A: T.handle, var_B: T.handle, var_NT_matmul: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), m, T.int64(32), T.int64(128)), "float16")
    n = T.int64()
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, m), "float16")
    NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), n, T.int64(32), T.int64(128), m):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_k, v_i2, v_i3], B[v_i0, v_i2, v_i1, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_k, v_i2, v_i3] * B[v_i0, v_i2, v_i1, v_k]

@T.prim_func(private=True)
def NT_matmul2_after(var_A: T.handle, var_B: T.handle, var_NT_matmul: T.handle):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    m = T.int32()
    A = T.match_buffer(var_A, (1, m, 32, 128), "float16")
    n = T.int32()
    B = T.match_buffer(var_B, (1, 32, n, m), "float16")
    NT_matmul = T.match_buffer(var_NT_matmul, (1, n, 32, 128), "float16")
    # with T.block("root"):
    NT_matmul_reindex_pad_local = T.alloc_buffer((32, 128, (n + 63) // 64 * 64), "float16", scope="local")
    A_reindex_pad_shared = T.alloc_buffer((32, 128, (m + 15) // 16 * 16), "float16", scope="shared")
    B_reindex_pad_shared = T.alloc_buffer((32, (n + 63) // 64 * 64, (m + 15) // 16 * 16), "float16", scope="shared")
    for ax0_ax2_0_fused in T.thread_binding((n + 63) // 64 * 32, thread="blockIdx.y"):
        for ax1_0 in T.thread_binding(4, thread="blockIdx.x"):
            for ax2_1 in T.thread_binding(1, thread="vthread.y"):
                for ax1_1 in T.thread_binding(1, thread="vthread.x"):
                    for ax2_2 in T.thread_binding(16, thread="threadIdx.y"):
                        for ax1_2 in T.thread_binding(8, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax2_3_init, ax1_3_init in T.grid(4, 4):
                                with T.block("NT_matmul_init"):
                                    v0 = T.axis.spatial(32, ax0_ax2_0_fused // ((n + 63) // 64))
                                    v1 = T.axis.spatial(128, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                    v2 = T.axis.spatial((n + 63) // 64 * 64, ax0_ax2_0_fused % ((n + 63) // 64) * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_init)
                                    T.reads()
                                    T.writes(NT_matmul_reindex_pad_local[v0, v1, v2])
                                    NT_matmul_reindex_pad_local[v0, v1, v2] = T.float16(0)
                            for ax3_0 in range((m + 15) // 16):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_fused_2 in range(2):
                                            for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                with T.block("A_reindex_pad_shared"):
                                                    v0 = T.axis.spatial(32, ax0_ax2_0_fused // ((n + 63) // 64))
                                                    v1 = T.axis.spatial(128, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                    v2 = T.axis.spatial((m + 15) // 16 * 16, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                    T.reads(A[0, v2, v0, v1])
                                                    T.writes(A_reindex_pad_shared[v0, v1, v2])
                                                    T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                    A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < m, A[0, v2, v0, v1], T.float16(0))
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_fused_2 in range(4):
                                            for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                with T.block("B_reindex_pad_shared"):
                                                    v0 = T.axis.spatial(32, ax0_ax2_0_fused // ((n + 63) // 64))
                                                    v1 = T.axis.spatial((n + 63) // 64 * 64, ax0_ax2_0_fused % ((n + 63) // 64) * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                    v2 = T.axis.spatial((m + 15) // 16 * 16, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                    T.reads(B[0, v0, v1, v2])
                                                    T.writes(B_reindex_pad_shared[v0, v1, v2])
                                                    T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                    B_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n and v2 < m, B[0, v0, v1, v2], T.float16(0))
                                for ax3_1, ax2_3, ax1_3 in T.grid(16, 4, 4):
                                    with T.block("NT_matmul_update"):
                                        v0 = T.axis.spatial(32, ax0_ax2_0_fused // ((n + 63) // 64))
                                        v1 = T.axis.spatial(128, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                        v2 = T.axis.spatial((n + 63) // 64 * 64, ax0_ax2_0_fused % ((n + 63) // 64) * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3)
                                        v3 = T.axis.reduce((m + 15) // 16 * 16, ax3_0 * 16 + ax3_1)
                                        T.reads(NT_matmul_reindex_pad_local[v0, v1, v2], A_reindex_pad_shared[v0, v1, v3], B_reindex_pad_shared[v0, v2, v3])
                                        T.writes(NT_matmul_reindex_pad_local[v0, v1, v2])
                                        NT_matmul_reindex_pad_local[v0, v1, v2] = NT_matmul_reindex_pad_local[v0, v1, v2] + A_reindex_pad_shared[v0, v1, v3] * B_reindex_pad_shared[v0, v2, v3]
                            for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                for ax2_1_1 in T.vectorized(2):
                                    with T.block("NT_matmul_reindex_pad_local"):
                                        v0 = T.axis.spatial(32, ax0_ax2_0_fused // ((n + 63) // 64) + ax0)
                                        v1 = T.axis.spatial(128, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                        v2 = T.axis.spatial((n + 63) // 64 * 64, ax0_ax2_0_fused % ((n + 63) // 64) * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                        T.reads(NT_matmul_reindex_pad_local[v0, v1, v2])
                                        T.writes(NT_matmul[0, v2, v0, v1])
                                        if v2 < n:
                                            NT_matmul[0, v2, v0, v1] = NT_matmul_reindex_pad_local[v0, v1, v2]


def get_dict_key(func):
    return tvm.ir.structural_hash(func), func


tir_dispatch_dict = {
    get_dict_key(fused_decode4_matmul3): fused_decode4_matmul3_after,
    get_dict_key(
        fused_decode6_fused_matmul7_add1
    ): fused_decode6_fused_matmul7_add1_after,
    get_dict_key(
        fused_decode5_fused_matmul6_multiply1
    ): fused_decode5_fused_matmul6_multiply1_after,
    get_dict_key(
        fused_decode5_fused_matmul6_silu1
    ): fused_decode5_fused_matmul6_silu1_after,
    get_dict_key(
        fused_decode4_fused_matmul4_add1
    ): fused_decode4_fused_matmul4_add1_after,
    get_dict_key(
        fused_decode3_fused_matmul1_cast2
    ): fused_decode3_fused_matmul1_cast2_after,
    get_dict_key(
        fused_decode2_fused_NT_matmul3_add
    ): fused_decode2_fused_NT_matmul3_add_after,
    get_dict_key(fused_decode_NT_matmul): fused_decode_NT_matmul_after,
    get_dict_key(fused_decode2_NT_matmul): fused_decode2_NT_matmul_after,
    get_dict_key(fused_decode4_NT_matmul3): fused_decode4_NT_matmul3_after,
    get_dict_key(
        fused_decode1_fused_NT_matmul2_silu
    ): fused_decode1_fused_NT_matmul2_silu_after,
    get_dict_key(
        fused_decode1_fused_NT_matmul2_multiply
    ): fused_decode1_fused_NT_matmul2_multiply_after,
    get_dict_key(
        fused_decode_fused_NT_matmul_add
    ): fused_decode_fused_NT_matmul_add_after,
    get_dict_key(
        fused_decode4_fused_matmul6_add4
    ): sch_fused_decode4_fused_matmul6_add4(fused_decode4_fused_matmul6_add4),
    get_dict_key(
        fused_decode6_fused_matmul9_add7_cast8_cast12_add5
    ): sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5(
        fused_decode6_fused_matmul9_add7_cast8_cast12_add5
    ),
    get_dict_key(
        fused_decode5_fused_matmul8_add6_gelu1_cast11
    ): sch_fused_decode5_fused_matmul8_add6_gelu1_cast11(
        fused_decode5_fused_matmul8_add6_gelu1_cast11
    ),
    get_dict_key(fused_decode81_fused_matmul1_cast2
    ): sch_fused_decode81_fused_matmul1_cast2(fused_decode81_fused_matmul1_cast2
    ),
    get_dict_key(
        fused_decode4_fused_matmul6_add4_add5
    ): sch_fused_decode4_fused_matmul6_add4_add5(fused_decode4_fused_matmul6_add4_add5),
    get_dict_key(fused_decode3_matmul3): sch_fused_decode3_matmul3(
        fused_decode3_matmul3
    ),
    get_dict_key(
        fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7
    ): sch_fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7(
        fused_decode6_fused_matmul9_add7_cast8_cast12_add5_cast7
    ),
    get_dict_key(
        fused_decode2_fused_NT_matmul3_add6_gelu1_cast11
    ): fused_decode2_fused_NT_matmul3_add6_gelu1_cast11_after,
    get_dict_key(
        fused_decode1_fused_NT_matmul1_add4
    ): fused_decode1_fused_NT_matmul1_add4_after,
    get_dict_key(
        fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5
    ): fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_after,
    get_dict_key(
        fused_decode1_fused_NT_matmul1_add4_add5
    ): fused_decode1_fused_NT_matmul1_add4_add5_after,
    get_dict_key(
        fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7
    ): fused_decode3_fused_NT_matmul4_add7_cast8_cast12_add5_cast7_after,
    get_dict_key(fused_fused_decode9_matmul7): fused_fused_decode9_matmul7_after,
    get_dict_key(fused_fused_decode7_matmul4): fused_fused_decode7_matmul4_after,
    get_dict_key(fused_NT_matmul1_divide2_maximum1_minimum1_cast3): fused_NT_matmul1_divide2_maximum1_minimum1_cast3_after,
    get_dict_key(NT_matmul3): NT_matmul3_after,
    get_dict_key(slice): slice_after,
    get_dict_key(rms_norm): rms_norm_after,
    get_dict_key(NT_matmul2): NT_matmul2_after,
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
