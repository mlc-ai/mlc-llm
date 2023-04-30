import tvm
from tvm import IRModule
from tvm.script import tir as T


# fmt: off
@T.prim_func
def fused_min_max_triu_te_broadcast_to(p_output0: T.handle, n: T.int64):
    T.func_attr({"tir.noalias": T.bool(True)})
    var_T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n), "float16")
    # with T.block("root"):
    var_make_diag_mask_te_intermediate = T.alloc_buffer((n, n), "float16")
    for i, j in T.grid(n, n):
        with T.block("make_diag_mask_te"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(var_make_diag_mask_te_intermediate[v_i, v_j])
            var_make_diag_mask_te_intermediate[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(65504))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
        with T.block("T_broadcast_to"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_make_diag_mask_te_intermediate[v_ax2, v_ax3])
            T.writes(var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_make_diag_mask_te_intermediate[v_ax2, v_ax3]


def fused_min_max_triu_te_broadcast_to_sch_func():
    sch = tvm.tir.Schedule(fused_min_max_triu_te_broadcast_to)
    b0 = sch.get_block("T_broadcast_to")
    sch.reverse_compute_inline(b0)
    return sch.mod["main"]


@T.prim_func
def rms_norm_before(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(4096),), "float32"), var_rms_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)))
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    rxplaceholderred_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rxplaceholderred_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(rxplaceholder_1[v_bsz, v_i, v_k])
            T.writes(rxplaceholderred_temp[v_bsz, v_i])
            with T.init():
                rxplaceholderred_temp[v_bsz, v_i] = T.float32(0)
            rxplaceholderred_temp[v_bsz, v_i] = rxplaceholderred_temp[v_bsz, v_i] + rxplaceholder_1[v_bsz, v_i, v_k] * rxplaceholder_1[v_bsz, v_i, v_k]
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(rxplaceholder[v_k], rxplaceholder_1[v_bsz, v_i, v_k], rxplaceholderred_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = rxplaceholder[v_k] * (rxplaceholder_1[v_bsz, v_i, v_k] / T.sqrt(rxplaceholderred_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07)))


@T.prim_func
def rms_norm_after(var_A: T.handle, var_weight: T.Buffer((T.int64(4096),), "float32"), var_rms_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)))
    rms_norm = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("compute_o"):
            v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
            v_i_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i_0)
            T.reads(A[v_bsz, v_i_o * T.int64(32):v_i_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            T.writes(rms_norm[v_bsz, T.int64(0) : T.int64(n), T.int64(0):T.int64(4096)])
            sq_sum_pad_local = T.alloc_buffer((T.int64(32),), scope="shared")
            for bsz, i_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(16)):
                for k_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("compute"):
                        v_i_i = T.axis.spatial(T.int64(32), i_1)
                        v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(256) + k_1)
                        T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i])
                        T.writes(sq_sum_pad_local[v_i_i])
                        with T.init():
                            sq_sum_pad_local[v_i_i] = T.float32(0)
                        sq_sum_pad_local[v_i_i] = sq_sum_pad_local[v_i_i] + T.if_then_else(v_i_o * T.int64(32) + v_i_i < n, A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i], T.float32(0)) * T.if_then_else(v_i_o * T.int64(32) + v_i_i < n, A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i], T.float32(0))
            for bsz_i_fused_1, k_0 in T.grid(T.int64(32), T.int64(16)):
                for k_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("compute_cache_write"):
                        v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i_i = T.axis.spatial(n, bsz_i_fused_1)
                        v_k = T.axis.spatial(T.int64(4096), k_0 * T.int64(256) + k_1)
                        T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k], var_weight[v_k], sq_sum_pad_local[v_i_i])
                        T.writes(rms_norm[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k])
                        if v_i_i < n:
                            rms_norm[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k] = var_weight[v_k] * (A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k] / T.sqrt(sq_sum_pad_local[v_i_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07)))


@T.prim_func
def rms_norm_fp16_before(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    rxplaceholderred_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rxplaceholderred_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(rxplaceholder_1[v_bsz, v_i, v_k])
            T.writes(rxplaceholderred_temp[v_bsz, v_i])
            with T.init():
                rxplaceholderred_temp[v_bsz, v_i] = T.float32(0)
            rxplaceholderred_temp[v_bsz, v_i] = rxplaceholderred_temp[v_bsz, v_i] + T.Cast("float32", rxplaceholder_1[v_bsz, v_i, v_k]) * T.Cast("float32", rxplaceholder_1[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(rxplaceholder[v_k], rxplaceholder_1[v_bsz, v_i, v_k], rxplaceholderred_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", rxplaceholder[v_k]) * (T.Cast("float32", rxplaceholder_1[v_bsz, v_i, v_k]) / T.sqrt(rxplaceholderred_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))


@T.prim_func
def rms_norm_fp16_after(var_A: T.handle, var_weight: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), dtype="float16")
    rms_norm = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), dtype="float16")
    # with T.block("root"):
    for i_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("compute_o"):
            v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
            v_i_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i_0)
            T.reads(A[v_bsz, v_i_o * T.int64(32):v_i_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            T.writes(rms_norm[v_bsz, T.int64(0) : T.int64(n), T.int64(0):T.int64(4096)])
            sq_sum_pad_local = T.alloc_buffer((T.int64(32),), scope="shared")
            for bsz, i_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(16)):
                for k_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("compute"):
                        v_i_i = T.axis.spatial(T.int64(32), i_1)
                        v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(256) + k_1)
                        T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i])
                        T.writes(sq_sum_pad_local[v_i_i])
                        with T.init():
                            sq_sum_pad_local[v_i_i] = T.float32(0)
                        sq_sum_pad_local[v_i_i] = sq_sum_pad_local[v_i_i] + T.if_then_else(v_i_o * T.int64(32) + v_i_i < n, T.Cast("float32", A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i]) * T.Cast("float32", A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i]), T.float32(0))
            for bsz_i_fused_1, k_0 in T.grid(T.int64(32), T.int64(16)):
                for k_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("compute_cache_write"):
                        v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i_i = T.axis.spatial(n, bsz_i_fused_1)
                        v_k = T.axis.spatial(T.int64(4096), k_0 * T.int64(256) + k_1)
                        T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k], var_weight[v_k], sq_sum_pad_local[v_i_i])
                        T.writes(rms_norm[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k])
                        if v_i_i < n:
                            rms_norm[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k] = T.Cast("float16", T.Cast("float32", var_weight[v_k]) * (T.Cast("float32", A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k]) / T.sqrt(sq_sum_pad_local[v_i_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))


@T.prim_func
def softmax_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, n))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, n))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], rxplaceholder[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(rxplaceholder[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_after(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, n))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, n))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_maxelem_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(n + T.int64(127)) // T.int64(128) * T.int64(128)])
            T.writes(T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_maxelem_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (n + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((n + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i])
                        T.writes(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.max(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i], T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < n, A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T.float32(-3.4028234663852886e+38)))
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
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(n + T.int64(127)) // T.int64(128) * T.int64(128)], T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T.writes(T_softmax_expsum[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_expsum_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (n + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((n + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T.writes(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float32(0)
                        T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] + T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < n, T.exp(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i] - T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i]), T.float32(0))
            for i0_i1_i2_1_fused_0 in range(T.int64(8)):
                for i0_i1_i2_1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum_cache_write"):
                        v_i1_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) // T.int64(32))
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32))
                        T.where(v_i2_o * T.int64(32) + (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32) < n)
                        T.reads(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i]
    for i0_i1_i2_fused_i3_fused_0 in T.thread_binding((n * T.int64(32) * n + T.int64(255)) // T.int64(256), thread="blockIdx.x"):
        for i0_i1_i2_fused_i3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("T_softmax_norm"):
                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_i1 = T.axis.spatial(T.int64(32), (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // n // n)
                v_i2 = T.axis.spatial(n, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // n % n)
                v_i3 = T.axis.spatial(n, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) % n)
                T.where(i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1 < n * T.int64(32) * n)
                T.reads(T_softmax_expsum[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]) / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_mxn_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
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
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
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

@T.prim_func
def softmax_cast_mxn_before(p_lv37: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    lv37 = T.match_buffer(p_lv37, (T.int64(1), T.int64(32), n, m))
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
    var_T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv37[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv37[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv37[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv37[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
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
            T.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])


@T.prim_func
def softmax_cast_mxn_after(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m), dtype="float16")
    # with T.block("root"):
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_maxelem_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(m + T.int64(127)) // T.int64(128) * T.int64(128)])
            T.writes(T_softmax_norm[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):m])
            T_softmax_maxelem_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
            T_softmax_expsum_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared")
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
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (m + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((m + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float32(0)
                        T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] + T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < m, T.exp(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i] - T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i]), T.float32(0))
            for i0_i1_i2_1_i3_fused_0 in range((T.int64(32) * T.int64(32) * m) // T.int64(128)):
                for i0_i1_i2_1_i3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(32), (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) // T.int64(32) // m)
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) // m % T.int64(32))
                        v_i3 = T.axis.spatial(m, (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) % m)
                        T.where(i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1 < T.int64(32) * T.int64(32) * m)
                        T.reads(T_softmax_expsum_pad_0_local[v_i0, v_i1, v_i2_i], A[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3], T_softmax_maxelem_pad_0_local[v_i0, v_i1, v_i2_i])
                        T.writes(T_softmax_norm[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3])
                        if v_i2_o * T.int64(32) + v_i2_i < n:
                            T_softmax_norm[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3] = T.Cast("float16", T.exp(A[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3] - T_softmax_maxelem_pad_0_local[v_i0, v_i1, v_i2_i]) / T_softmax_expsum_pad_0_local[v_i0, v_i1, v_i2_i])


@T.prim_func
def softmax_mxn_fp16_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    m = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, m), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
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
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax_mxn_fp16_after(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), dtype="float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m), dtype="float16")
    # with T.block("root"):
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_maxelem_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(m + T.int64(127)) // T.int64(128) * T.int64(128)])
            T.writes(T_softmax_norm[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):m])
            T_softmax_maxelem_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared", dtype="float16")
            T_softmax_expsum_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared", dtype="float16")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (m + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((m + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i])
                        T.writes(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float16(-65504)
                        T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.max(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i], T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < m, A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T.float16(-65504)))
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (m + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((m + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float16(0)
                        T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] + T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < m, T.exp(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i] - T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i]), T.float16(0))
            for i0_i1_i2_1_i3_fused_0 in range((T.int64(32) * T.int64(32) * m) // T.int64(128)):
                for i0_i1_i2_1_i3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(32), (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) // T.int64(32) // m)
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) // m % T.int64(32))
                        v_i3 = T.axis.spatial(m, (i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1) % m)
                        T.where(i0_i1_i2_1_i3_fused_0 * T.int64(128) + i0_i1_i2_1_i3_fused_1 < T.int64(32) * T.int64(32) * m)
                        T.reads(T_softmax_expsum_pad_0_local[v_i0, v_i1, v_i2_i], A[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3], T_softmax_maxelem_pad_0_local[v_i0, v_i1, v_i2_i])
                        T.writes(T_softmax_norm[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3])
                        if v_i2_o * T.int64(32) + v_i2_i < n:
                            T_softmax_norm[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3] = T.exp(A[v_i0, v_i1, v_i2_o * T.int64(32) + v_i2_i, v_i3] - T_softmax_maxelem_pad_0_local[v_i0, v_i1, v_i2_i]) / T_softmax_expsum_pad_0_local[v_i0, v_i1, v_i2_i]


@T.prim_func
def softmax_fp16_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, n), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, n), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, n), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], rxplaceholder[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(rxplaceholder[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_fp16_after(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, n), dtype="float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, n), dtype="float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n), dtype="float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n), dtype="float16")
    for i2_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
        with T.block("T_softmax_maxelem_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0)
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(n + T.int64(127)) // T.int64(128) * T.int64(128)])
            T.writes(T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_maxelem_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared", dtype="float16")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (n + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((n + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i])
                        T.writes(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float16(-65504)
                        T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.max(T_softmax_maxelem_pad_0_local[v_i0, v_i1_i, v_i2_i], T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < n, A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T.float16(-65504)))
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
            T.reads(A[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):(n + T.int64(127)) // T.int64(128) * T.int64(128)], T_softmax_maxelem[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T.writes(T_softmax_expsum[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32)])
            T_softmax_expsum_pad_0_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32)), scope="shared", dtype="float16")
            for i0, i1, i2_1, k_0 in T.grid(T.int64(1), T.int64(32), T.int64(32), (n + T.int64(127)) // T.int64(128)):
                for k_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        v_i1_i, v_i2_i = T.axis.remap("SS", [i1, i2_1])
                        v_k_i = T.axis.reduce(T.int64(32) * ((n + T.int64(127)) // T.int64(128)), k_0 * T.int64(128) + k_1)
                        T.reads(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i], T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T.writes(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        with T.init():
                            T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T.float16(0)
                        T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i] + T.if_then_else(v_i2_o * T.int64(32) + v_i2_i < n and v_k_i < n, T.exp(A[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i, v_k_i] - T_softmax_maxelem[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i]), T.float16(0))
            for i0_i1_i2_1_fused_0 in range(T.int64(8)):
                for i0_i1_i2_1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    with T.block("T_softmax_expsum_cache_write"):
                        v_i1_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) // T.int64(32))
                        v_i2_i = T.axis.spatial(T.int64(32), (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32))
                        T.where(v_i2_o * T.int64(32) + (i0_i1_i2_1_fused_0 * T.int64(128) + i0_i1_i2_1_fused_1) % T.int64(32) < n)
                        T.reads(T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i])
                        T.writes(T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i])
                        T_softmax_expsum[v_i0, v_i1_i, v_i2_o * T.int64(32) + v_i2_i] = T_softmax_expsum_pad_0_local[v_i0, v_i1_i, v_i2_i]
    for i0_i1_i2_fused_i3_fused_0 in T.thread_binding((n * T.int64(32) * n + T.int64(255)) // T.int64(256), thread="blockIdx.x"):
        for i0_i1_i2_fused_i3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("T_softmax_norm"):
                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_i1 = T.axis.spatial(T.int64(32), (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // n // n)
                v_i2 = T.axis.spatial(n, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) // n % n)
                v_i3 = T.axis.spatial(n, (i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1) % n)
                T.where(i0_i1_i2_fused_i3_fused_0 * T.int64(256) + i0_i1_i2_fused_i3_fused_1 < n * T.int64(32) * n)
                T.reads(T_softmax_expsum[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]) / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_1xn_before(var_inp0: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    inp0 = T.match_buffer(var_inp0, (T.int64(1), T.int64(32), T.int64(1), n))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), T.int64(1), n))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(inp0[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], inp0[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(inp0[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(inp0[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def softmax_cast_1xn_before(p_lv1614: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv1614 = T.match_buffer(p_lv1614, (T.int64(1), T.int64(32), T.int64(1), n))
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    var_T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1614[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv1614[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv1614[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv1614[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])


@T.prim_func
def softmax_1xn_fp16_before(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], rxplaceholder[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(rxplaceholder[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


def softmax_1xn_sch_func(f_softmax, cast_to_fp16: bool = False):
    sch = tvm.tir.Schedule(f_softmax)
    if cast_to_fp16:
        b_cast = sch.get_block("compute")
        sch.reverse_compute_inline(b_cast)

    b0 = sch.get_block("T_softmax_exp")
    sch.compute_inline(b0)
    b1 = sch.get_block("T_softmax_norm")
    l2, l3, l4, l5 = sch.get_loops(b1)
    l6, l7 = sch.split(l5, [None, 128])
    sch.bind(l7, "threadIdx.x")
    b8 = sch.get_block("T_softmax_expsum")
    sch.compute_at(b8, l4)
    sch.set_scope(b8, 0, "local")
    l9, l10, l11, l12 = sch.get_loops(b8)
    l13, l14 = sch.split(l12, [None, 128])
    sch.bind(l14, "threadIdx.x")
    b15 = sch.get_block("T_softmax_maxelem")
    sch.compute_at(b15, l4)
    sch.set_scope(b15, 0, "local")
    l16, l17, l18, l19 = sch.get_loops(b15)
    l20, l21 = sch.split(l19, [None, 128])
    sch.bind(l21, "threadIdx.x")
    l22 = sch.fuse(l2, l3, l4)
    sch.bind(l22, "blockIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def matmul1_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), T.int64(1), n))
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(rxplaceholder[T.int64(0), v_i1, v_i2, v_k], rxplaceholder_1[T.int64(0), v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + rxplaceholder[T.int64(0), v_i1, v_i2, v_k] * rxplaceholder_1[T.int64(0), v_i1, v_k, v_i3]


@T.prim_func
def matmul1_after(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), T.int64(1), n))
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    matmul_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), scope="local")
    rxplaceholder_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128)), scope="shared")
    rxplaceholder_1_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), scope="shared")
    for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
        for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                        v_i1 = T.axis.spatial(T.int64(32), i0_2_i1_2_i2_2_i3_2_fused // T.int64(8) * T.int64(2) + i1_3_init * T.int64(2) + i1_4_init)
                        v_i2 = T.axis.spatial(T.int64(1), i2_3_init + i2_4_init)
                        v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused * T.int64(8) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(8) + i3_3_init + i3_4_init)
                        T.reads()
                        T.writes(matmul_local[v_i0, v_i1, v_i2, v_i3])
                        T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                        matmul_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                for k_0, k_1_0 in T.grid((n + T.int64(127)) // T.int64(128), T.int64(8)):
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                            for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                with T.block("rxplaceholder_pad_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(16))
                                    v2 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v3 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(16))
                                    T.reads(rxplaceholder[v0, v1, v2, v3])
                                    T.writes(rxplaceholder_pad_shared[v0, v1, v2, v3])
                                    rxplaceholder_pad_shared[v0, v1, v2, v3] = T.if_then_else(v3 < n, rxplaceholder[v0, v1, v2, v3], T.float32(0))
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(8)):
                        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                            for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                with T.block("rxplaceholder_1_pad_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(128))
                                    v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(128) // T.int64(8))
                                    v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                    T.reads(rxplaceholder_1[v0, v1, v2, v3])
                                    T.writes(rxplaceholder_1_pad_shared[v0, v1, v2, v3])
                                    rxplaceholder_1_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < n, rxplaceholder_1[v0, v1, v2, v3], T.float32(0))
                    for k_1_1, i0_3, i1_3, i2_3, i3_3, k_1_2, i0_4, i1_4, i2_4, i3_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(8), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                            v_i1 = T.axis.spatial(T.int64(32), i0_2_i1_2_i2_2_i3_2_fused // T.int64(8) * T.int64(2) + i1_3 * T.int64(2) + i1_4)
                            v_i2 = T.axis.spatial(T.int64(1), i2_3 + i2_4)
                            v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused * T.int64(8) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(8) + i3_3 + i3_4)
                            v_k = T.axis.reduce((n + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(16) + k_1_1 * T.int64(8) + k_1_2)
                            T.reads(matmul_local[v_i0, v_i1, v_i2, v_i3], rxplaceholder_pad_shared[v_i0, v_i1, v_i2, v_k], rxplaceholder_1_pad_shared[v_i0, v_i1, v_k, v_i3])
                            T.writes(matmul_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            matmul_local[v_i0, v_i1, v_i2, v_i3] = matmul_local[v_i0, v_i1, v_i2, v_i3] + rxplaceholder_pad_shared[v_i0, v_i1, v_i2, v_k] * rxplaceholder_1_pad_shared[v_i0, v_i1, v_k, v_i3]
                for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                    with T.block("matmul_local"):
                        v0 = T.axis.spatial(T.int64(1), ax0)
                        v1 = T.axis.spatial(T.int64(32), i0_2_i1_2_i2_2_i3_2_fused // T.int64(8) * T.int64(2) + ax1)
                        v2 = T.axis.spatial(T.int64(1), ax2)
                        v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused * T.int64(8) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(8) + ax3)
                        T.reads(matmul_local[v0, v1, v2, v3])
                        T.writes(matmul[v0, v1, v2, v3])
                        matmul[v0, v1, v2, v3] = matmul_local[v0, v1, v2, v3]


@T.prim_func
def matmul2_before(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    inp0 = T.match_buffer(var_inp0, (T.int64(1), n, T.int64(4096)))
    matmul = T.match_buffer(var_matmul, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(inp0[v_i0, v_i1, v_k], inp1[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

def matmul2_sch_func():
    sch = tvm.tir.Schedule(matmul2_before)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l2, l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l11, l12, l13, l14, l15 = sch.split(loop=l2, factors=[v6, v7, v8, v9, v10], preserve_unit_iters=True)
    v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[2, 2, 2, 4, 1])
    l21, l22, l23, l24, l25 = sch.split(loop=l3, factors=[v16, v17, v18, v19, v20], preserve_unit_iters=True)
    v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[128, 2, 16, 1, 1])
    l31, l32, l33, l34, l35 = sch.split(loop=l4, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
    v36, v37, v38 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64, decision=[512, 4, 2])
    l39, l40, l41 = sch.split(loop=l5, factors=[v36, v37, v38], preserve_unit_iters=True)
    sch.reorder(l11, l21, l31, l12, l22, l32, l13, l23, l33, l39, l40, l14, l24, l34, l41, l15, l25, l35)
    l42 = sch.fuse(l11, l21, l31, preserve_unit_iters=True)
    sch.bind(loop=l42, thread_axis="blockIdx.x")
    l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="vthread.x")
    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b45 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b45, loop=l44, preserve_unit_loops=True, index=-1)
    b46 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b46, loop=l39, preserve_unit_loops=True, index=-1)
    _, l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
    l54 = sch.fuse(l51, l52, l53, preserve_unit_iters=True)
    v55 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
    b56 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b56, loop=l39, preserve_unit_loops=True, index=-1)
    _, l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b56)
    l63 = sch.fuse(l61, l62, preserve_unit_iters=True)
    v64 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch", ann_val=v64)
    v65 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v65)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
    _, l66, l67, l68, l69, l70 = sch.get_loops(block=b46)
    l71, l72, l73 = sch.split(loop=l70, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l73)
    sch.bind(loop=l72, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch")
    _, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)
    l79, l80, l81 = sch.split(loop=l78, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l81)
    sch.bind(loop=l80, thread_axis="threadIdx.x")
    b82 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")
    _, b83, b84, b85, b86, _  = sch.get_child_blocks(b82)
    _, l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b83)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l94, l95, l96, l97, l98, l99, l100 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l94, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l94, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l101, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l101, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l101, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    b119 = sch.get_block(name="matmul", func_name="main")
    _, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b119)
    b132 = sch.decompose_reduction(block=b119, loop=l123)
    b1 = sch.get_block("inp0_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("matmul_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def matmul5_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, n))
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)))
    matmul_1 = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(rxplaceholder[T.int64(0), v_i1, v_i2, v_k], rxplaceholder_1[T.int64(0), v_i1, v_k, v_i3])
            T.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + rxplaceholder[T.int64(0), v_i1, v_i2, v_k] * rxplaceholder_1[T.int64(0), v_i1, v_k, v_i3]


@T.prim_func
def matmul5_after(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, n))
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)))
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    C_pad = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), T.int64(128)))
    C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), scope="local")
    A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (n + T.int64(127)) // T.int64(128) * T.int64(128)), scope="shared")
    B_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), scope="shared")
    for i2_0 in range((n + T.int64(127)) // T.int64(128)):
        for i0_0_i1_0_i2_1_0_i3_0_fused in T.thread_binding(T.int64(256), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1}):
            for i0_1_i1_1_i2_1_1_i3_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                for i0_2_i1_2_i2_1_2_i3_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_1_3_init, i3_3_init, i0_4_init, i1_4_init, i2_1_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(4), T.int64(1)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial((n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + i2_1_3_init * T.int64(4) + i2_1_4_init)
                            v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + i3_3_init + i3_4_init)
                            T.reads()
                            T.writes(C_pad_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                            C_pad_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0, k_1_0 in T.grid((n + T.int64(127)) // T.int64(128), T.int64(16)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("A_pad_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8))
                                        v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                        v3 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                        T.reads(rxplaceholder[v0, v1, v2, v3])
                                        T.writes(A_pad_shared[v0, v1, v2, v3])
                                        A_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < n and v3 < n, rxplaceholder[v0, v1, v2, v3], T.float32(0))
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("B_pad_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8))
                                        v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(64))
                                        v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                                        T.reads(rxplaceholder_1[v0, v1, v2, v3])
                                        T.writes(B_pad_shared[v0, v1, v2, v3])
                                        B_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < n, rxplaceholder_1[v0, v1, v2, v3], T.float32(0))
                        for k_1_1, i0_3, i1_3, i2_1_3, i3_3, k_1_2, i0_4, i1_4, i2_1_4, i3_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(4), T.int64(1)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial((n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + i2_1_3 * T.int64(4) + i2_1_4)
                                v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + i3_3 + i3_4)
                                v_k = T.axis.reduce((n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + k_1_1 * T.int64(4) + k_1_2)
                                T.reads(C_pad_local[v_i0, v_i1, v_i2, v_i3], A_pad_shared[T.int64(0), v_i1, v_i2, v_k], B_pad_shared[T.int64(0), v_i1, v_k, v_i3])
                                T.writes(C_pad_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[v_i0, v_i1, v_i2, v_i3] = C_pad_local[v_i0, v_i1, v_i2, v_i3] + A_pad_shared[T.int64(0), v_i1, v_i2, v_k] * B_pad_shared[T.int64(0), v_i1, v_k, v_i3]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(2)):
                        with T.block("C_pad_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + ax1)
                            v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + ax2)
                            v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + ax3)
                            T.reads(C_pad_local[v0, v1, v2, v3])
                            T.writes(C_pad[v0, v1, v2, v3])
                            C_pad[v0, v1, v2, v3] = C_pad_local[v0, v1, v2, v3]
    for i0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
        for i1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i2, i3 in T.grid(n, T.int64(128)):
                with T.block("C_pad"):
                    vi0, vi1, vi2, vi3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(C_pad[vi0, vi1, vi2, vi3])
                    T.writes(matmul[vi0, vi1, vi2, vi3])
                    matmul[vi0, vi1, vi2, vi3] = C_pad[vi0, vi1, vi2, vi3]

@T.prim_func
def matmul5_with_m_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    A = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, m))
    B = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), m, T.int64(128)))
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), m):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]


@T.prim_func
def matmul5_with_m_after(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, m))
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), m, T.int64(128)))
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)))
    # with T.block("root"):
    C_pad = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), T.int64(128)))
    C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), scope="local")
    A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(127)) // T.int64(128) * T.int64(128), (m + T.int64(127)) // T.int64(128) * T.int64(128)), scope="shared")
    B_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (m + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(128)), scope="shared")
    for i2_0 in range((n + T.int64(127)) // T.int64(128)):
        for i0_0_i1_0_i2_1_0_i3_0_fused in T.thread_binding(T.int64(256), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1}):
            for i0_1_i1_1_i2_1_1_i3_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                for i0_2_i1_2_i2_1_2_i3_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_1_3_init, i3_3_init, i0_4_init, i1_4_init, i2_1_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(4), T.int64(1)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial((n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + i2_1_3_init * T.int64(4) + i2_1_4_init)
                            v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + i3_3_init + i3_4_init)
                            T.reads()
                            T.writes(C_pad_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                            C_pad_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0, k_1_0 in T.grid((m + T.int64(127)) // T.int64(128), T.int64(16)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("A_pad_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8))
                                        v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                        v3 = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                        T.reads(rxplaceholder[v0, v1, v2, v3])
                                        T.writes(A_pad_shared[v0, v1, v2, v3])
                                        A_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < n and v3 < m, rxplaceholder[v0, v1, v2, v3], T.float32(0))
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("B_pad_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8))
                                        v2 = T.axis.spatial((m + T.int64(127)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(64))
                                        v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                                        T.reads(rxplaceholder_1[v0, v1, v2, v3])
                                        T.writes(B_pad_shared[v0, v1, v2, v3])
                                        B_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < m, rxplaceholder_1[v0, v1, v2, v3], T.float32(0))
                        for k_1_1, i0_3, i1_3, i2_1_3, i3_3, k_1_2, i0_4, i1_4, i2_1_4, i3_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(4), T.int64(1)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial((n + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + i2_1_3 * T.int64(4) + i2_1_4)
                                v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + i3_3 + i3_4)
                                v_k = T.axis.reduce((m + T.int64(128) - T.int64(1)) // T.int64(128) * T.int64(128), k_0 * T.int64(128) + k_1_0 * T.int64(8) + k_1_1 * T.int64(4) + k_1_2)
                                T.reads(C_pad_local[v_i0, v_i1, v_i2, v_i3], A_pad_shared[T.int64(0), v_i1, v_i2, v_k], B_pad_shared[T.int64(0), v_i1, v_k, v_i3])
                                T.writes(C_pad_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[v_i0, v_i1, v_i2, v_i3] = C_pad_local[v_i0, v_i1, v_i2, v_i3] + A_pad_shared[T.int64(0), v_i1, v_i2, v_k] * B_pad_shared[T.int64(0), v_i1, v_k, v_i3]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(2)):
                        with T.block("C_pad_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_0_fused // T.int64(8) + ax1)
                            v2 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), i2_0 * T.int64(128) + i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(8) // T.int64(2) * T.int64(32) + i0_1_i1_1_i2_1_1_i3_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_2_i2_1_2_i3_2_fused // T.int64(16) * T.int64(4) + ax2)
                            v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_1_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_1_i1_1_i2_1_1_i3_1_fused % T.int64(2) * T.int64(32) + i0_2_i1_2_i2_1_2_i3_2_fused % T.int64(16) * T.int64(2) + ax3)
                            T.reads(C_pad_local[v0, v1, v2, v3])
                            T.writes(C_pad[v0, v1, v2, v3])
                            C_pad[v0, v1, v2, v3] = C_pad_local[v0, v1, v2, v3]
    for i0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
        for i1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            for i2, i3 in T.grid(n, T.int64(128)):
                with T.block("C_pad"):
                    vi0, vi1, vi2, vi3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(C_pad[vi0, vi1, vi2, vi3])
                    T.writes(matmul[vi0, vi1, vi2, vi3])
                    matmul[vi0, vi1, vi2, vi3] = C_pad[vi0, vi1, vi2, vi3]


@T.prim_func
def NT_matmul_before(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_NT_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)))
    NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder_1[v_i0, v_i1, v_k], rxplaceholder[v_i2, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
            NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + rxplaceholder_1[v_i0, v_i1, v_k] * rxplaceholder[v_i2, v_k]


@T.prim_func
def NT_matmul_after(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_NT_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)))
    NT_matmul_1 = T.match_buffer(var_NT_matmul, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i1_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i1_0)
            T.reads(rxplaceholder_1[T.Add(v_i0, T.int64(0)), v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)], rxplaceholder[T.int64(0):T.int64(4096), T.int64(0):T.int64(4096)])
            T.writes(NT_matmul_1[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
            A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
            rxplaceholder_shared = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="shared")
            for i0_0_i1_1_0_i2_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_1_i2_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                    for i0_2_i1_1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_1_3_init, i2_3_init, i1_1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(2)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3_init * T.int64(4) + i1_1_4_init)
                                v_i2_i = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3_init * T.int64(2) + i2_4_init)
                                T.reads()
                                T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[T.int64(0), v_i1_i, v_i2_i] = T.float32(0)
                        for k_0 in range(T.int64(128)):
                            for ax0_ax1_ax2_fused_0 in range(T.int64(8)):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("A_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) % T.int64(32))
                                            T.reads(rxplaceholder_1[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                            T.writes(A_pad_shared[v0, v1, v2])
                                            A_pad_shared[v0, v1, v2] = T.if_then_else(v_i1_o * T.int64(32) + v1 < n, rxplaceholder_1[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], T.float32(0))
                            for ax0_ax1_fused_0 in range(T.int64(8)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("rxplaceholder_shared"):
                                            v0 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(32))
                                            v1 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(32))
                                            T.reads(rxplaceholder[v0, v1])
                                            T.writes(rxplaceholder_shared[v0, v1])
                                            rxplaceholder_shared[v0, v1] = rxplaceholder[v0, v1]
                            for k_1, i0_3, i1_1_3, i2_3, k_2, i0_4, i1_1_4, i2_4 in T.grid(T.int64(8), T.int64(1), T.int64(1), T.int64(2), T.int64(4), T.int64(1), T.int64(4), T.int64(2)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3 * T.int64(4) + i1_1_4)
                                    v_i2_i = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3 * T.int64(2) + i2_4)
                                    v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(4) + k_2)
                                    T.reads(C_pad_local[T.int64(0), v_i1_i, v_i2_i], A_pad_shared[T.int64(0), v_i1_i, v_k_i], rxplaceholder_shared[v_i2_i, v_k_i])
                                    T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    C_pad_local[T.int64(0), v_i1_i, v_i2_i] = C_pad_local[T.int64(0), v_i1_i, v_i2_i] + A_pad_shared[T.int64(0), v_i1_i, v_k_i] * rxplaceholder_shared[v_i2_i, v_k_i]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                            with T.block("C_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + ax2)
                                T.reads(C_pad_local[v0, v1, v2])
                                T.writes(NT_matmul_1[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i1_o * T.int64(32) + v1 and v_i1_o * T.int64(32) + v1 < n:
                                if v_i1_o * T.int64(32) + v1 < n:
                                    NT_matmul_1[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] = C_pad_local[v0, v1, v2]


@T.prim_func
def NT_matmul4_before(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(32000), T.int64(4096)), "float32"), var_NT_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)))
    NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), n, T.int64(32000)))
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder_1[v_i0, v_i1, v_k], rxplaceholder[v_i2, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
            NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + rxplaceholder_1[v_i0, v_i1, v_k] * rxplaceholder[v_i2, v_k]


def NT_matmul4_sch_func():
    sch = tvm.tir.Schedule(NT_matmul4_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 256, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l2, l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l11, l12, l13, l14, l15 = sch.split(loop=l2, factors=[v6, v7, v8, v9, v10], preserve_unit_iters=True)
    v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 8, 4, 1])
    l21, l22, l23, l24, l25 = sch.split(loop=l3, factors=[v16, v17, v18, v19, v20], preserve_unit_iters=True)
    v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[668, 1, 8, 1, 6])
    l31, l32, l33, l34, l35 = sch.split(loop=l4, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
    v36, v37, v38 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64, decision=[128, 4, 8])
    l39, l40, l41 = sch.split(loop=l5, factors=[v36, v37, v38], preserve_unit_iters=True)
    sch.reorder(l11, l21, l31, l12, l22, l32, l13, l23, l33, l39, l40, l14, l24, l34, l41, l15, l25, l35)
    l42 = sch.fuse(l11, l21, l31, preserve_unit_iters=True)
    sch.bind(loop=l42, thread_axis="blockIdx.x")
    l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="vthread.x")
    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b45 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b45, loop=l44, preserve_unit_loops=True, index=-1)
    b46 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b46, loop=l39, preserve_unit_loops=True, index=-1)
    _, l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
    l54 = sch.fuse(l51, l52, l53, preserve_unit_iters=True)
    v55 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
    b56 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b56, loop=l39, preserve_unit_loops=True, index=-1)
    _, l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b56)
    l63 = sch.fuse(l61, l62, preserve_unit_iters=True)
    v64 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch", ann_val=v64)
    v65 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v65)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
    _, l66, l67, l68, l69, l70 = sch.get_loops(block=b46)
    l71, l72, l73 = sch.split(loop=l70, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l73)
    sch.bind(loop=l72, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch")
    _, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)
    l79, l80, l81 = sch.split(loop=l78, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l81)
    sch.bind(loop=l80, thread_axis="threadIdx.x")
    b82 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")
    _, b83, b84, b85, b86, _ = sch.get_child_blocks(b82)
    _, l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b83)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l94, l95, l96, l97, l98, l99, l100 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l94, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l94, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l101, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l101, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l101, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    b119 = sch.get_block(name="NT_matmul", func_name="main")
    _, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b119)
    b132 = sch.decompose_reduction(block=b119, loop=l123)
    b1 = sch.get_block("rxplaceholder_1_pad")
    sch.compute_inline(b1)
    b3 = sch.get_block("NT_matmul_pad")
    sch.reverse_compute_inline(b3)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def NT_matmul9_before(rxplaceholder: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), rxplaceholder_1: T.Buffer((T.int64(32000), T.int64(4096)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_k], rxplaceholder_1[v_i2, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
            NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + rxplaceholder[v_i0, v_i1, v_k] * rxplaceholder_1[v_i2, v_k]


def NT_matmul9_sch_func():
    sch = tvm.tir.Schedule(NT_matmul9_before)
    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l2, l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l11, l12, l13, l14, l15 = sch.split(loop=l2, factors=[v6, v7, v8, v9, v10], preserve_unit_iters=True)
    v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l21, l22, l23, l24, l25 = sch.split(loop=l3, factors=[v16, v17, v18, v19, v20], preserve_unit_iters=True)
    v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[668, 1, 48, 1, 1])
    l31, l32, l33, l34, l35 = sch.split(loop=l4, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
    v36, v37, v38 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64, decision=[64, 64, 1])
    l39, l40, l41 = sch.split(loop=l5, factors=[v36, v37, v38], preserve_unit_iters=True)
    sch.reorder(l11, l21, l31, l12, l22, l32, l13, l23, l33, l39, l40, l14, l24, l34, l41, l15, l25, l35)
    l42 = sch.fuse(l11, l21, l31, preserve_unit_iters=True)
    sch.bind(loop=l42, thread_axis="blockIdx.x")
    l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="vthread.x")
    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b45 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b45, loop=l44, preserve_unit_loops=True, index=-1)
    b46 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b46, loop=l39, preserve_unit_loops=True, index=-1)
    l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
    l54 = sch.fuse(l51, l52, l53, preserve_unit_iters=True)
    v55 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
    b56 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b56, loop=l39, preserve_unit_loops=True, index=-1)
    l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b56)
    l63 = sch.fuse(l61, l62, preserve_unit_iters=True)
    v64 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch", ann_val=v64)
    v65 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v65)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
    l66, l67, l68, l69, l70 = sch.get_loops(block=b46)
    l71, l72, l73 = sch.split(loop=l70, factors=[None, 48, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l73)
    sch.bind(loop=l72, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch")
    l74, l75, l76, l77, l78 = sch.get_loops(block=b56)
    l79, l80, l81 = sch.split(loop=l78, factors=[None, 48, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l81)
    sch.bind(loop=l80, thread_axis="threadIdx.x")
    b82 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")
    b83, b84, b85, b86 = sch.get_child_blocks(b82)
    l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b83)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    l94, l95, l96, l97, l98, l99, l100 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l94, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l94, ann_key="pragma_unroll_explicit", ann_val=1)
    l101, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l101, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l101, ann_key="pragma_unroll_explicit", ann_val=1)
    l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    b119 = sch.get_block(name="NT_matmul", func_name="main")
    l120, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b119)
    b132 = sch.decompose_reduction(block=b119, loop=l123)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)



@T.prim_func
def fused_matmul1_add1(p_lv39: T.handle, lv40: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv39 = T.match_buffer(p_lv39, (T.int64(1), n, T.int64(4096)))
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv39[v_i0, v_i1, v_k], lv40[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv39[v_i0, v_i1, v_k] * lv40[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_matmul1_add1_sch_func():
    sch = tvm.tir.Schedule(fused_matmul1_add1)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 8, 4, 1, 1])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[128, 2, 16, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[512, 4, 2])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="matmul", func_name="main")
    _, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    b133 = sch.decompose_reduction(block=b120, loop=l124)
    b1 = sch.get_block("lv39_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_matmul3_multiply(p_lv43: T.handle, lv46: T.Buffer((T.int64(4096), T.int64(11008)), "float32"), p_lv48: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)))
    lv48 = T.match_buffer(p_lv48, (T.int64(1), n, T.int64(11008)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], lv46[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * lv46[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv48[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = lv48[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_matmul3_multiply_sch_func():
    sch = tvm.tir.Schedule(fused_matmul3_multiply)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="T_multiply", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 4, 2, 4, 1])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[344, 2, 16, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[512, 1, 8])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="matmul", func_name="main")
    _, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    b133 = sch.decompose_reduction(block=b120, loop=l124)
    b1 = sch.get_block("lv43_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_matmul3_silu(p_lv43: T.handle, lv44: T.Buffer((T.int64(4096), T.int64(11008)), "float32"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], lv44[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * lv44[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


def fused_matmul3_silu_sch_func():
    sch = tvm.tir.Schedule(fused_matmul3_silu)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="compute", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l4, l5, l6, l7 = sch.get_loops(block=b0)
    v8, v9, v10, v11, v12 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l13, l14, l15, l16, l17 = sch.split(loop=l4, factors=[v8, v9, v10, v11, v12], preserve_unit_iters=True)
    v18, v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 2, 2, 8, 1])
    l23, l24, l25, l26, l27 = sch.split(loop=l5, factors=[v18, v19, v20, v21, v22], preserve_unit_iters=True)
    v28, v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[344, 2, 16, 1, 1])
    l33, l34, l35, l36, l37 = sch.split(loop=l6, factors=[v28, v29, v30, v31, v32], preserve_unit_iters=True)
    v38, v39, v40 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64, decision=[512, 1, 8])
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
    _, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b48)
    l56 = sch.fuse(l53, l54, l55, preserve_unit_iters=True)
    v57 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch", ann_val=v57)
    b58 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b58, loop=l41, preserve_unit_loops=True, index=-1)
    _, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b58)
    l65 = sch.fuse(l63, l64, preserve_unit_iters=True)
    v66 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v66)
    sch.compute_inline(block=b1)
    v67 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
    sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v67)
    l68, l69, l70 = sch.get_loops(block=b2)
    l71 = sch.fuse(l68, l69, l70, preserve_unit_iters=True)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 256, 256], preserve_unit_iters=True)
    sch.reorder(l73, l74, l72)
    sch.bind(loop=l73, thread_axis="blockIdx.x")
    sch.bind(loop=l74, thread_axis="threadIdx.x")
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b48)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
    _, l83, l84, l85, l86, l87 = sch.get_loops(block=b58)
    l88, l89, l90 = sch.split(loop=l87, factors=[None, 32, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l90)
    sch.bind(loop=l89, thread_axis="threadIdx.x")
    b91 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b91, ann_key="meta_schedule.unroll_explicit")
    _, b92, b93, b94, b95, _, b96 = sch.get_child_blocks(b91)
    _, l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b92)
    sch.annotate(block_or_loop=l97, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l97, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l104, l105, l106, l107, l108, l109, l110 = sch.get_loops(block=b93)
    sch.annotate(block_or_loop=l104, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l104, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l111, l112, l113, l114, l115, l116, l117, l118, l119, l120, l121, l122 = sch.get_loops(block=b94)
    sch.annotate(block_or_loop=l111, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l111, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l123, l124, l125, l126, l127, l128 = sch.get_loops(block=b95)
    sch.annotate(block_or_loop=l123, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l123, ann_key="pragma_unroll_explicit", ann_val=1)
    l129, l130, l131 = sch.get_loops(block=b96)
    sch.annotate(block_or_loop=l129, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l129, ann_key="pragma_unroll_explicit", ann_val=1)
    b132 = sch.get_block(name="matmul", func_name="main")
    _, l133, l134, l135, l136, l137, l138, l139, l140, l141, l142, l143, l144 = sch.get_loops(block=b132)
    b145 = sch.decompose_reduction(block=b132, loop=l136)
    b1 = sch.get_block("lv43_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_matmul4_add1(p_lv49: T.handle, lv50: T.Buffer((T.int64(11008), T.int64(4096)), "float32"), p_lv42: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(11008)))
    lv42 = T.match_buffer(p_lv42, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv49[v_i0, v_i1, v_k], lv50[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv49[v_i0, v_i1, v_k] * lv50[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv42[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv42[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_matmul4_add1_sch_func():
    sch = tvm.tir.Schedule(fused_matmul4_add1)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 4, 8, 1])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[128, 2, 16, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[1376, 2, 4])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="matmul", func_name="main")
    _, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    b133 = sch.decompose_reduction(block=b120, loop=l124)
    b1 = sch.get_block("lv49_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul_add1_before(p_lv39: T.handle, linear_weight3: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv39 = T.match_buffer(p_lv39, (T.int64(1), n, T.int64(4096)))
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv39[v_i0, v_i1, v_k], linear_weight3[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv39[v_i0, v_i1, v_k] * linear_weight3[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_NT_matmul_add1_after(p_lv33: T.handle, linear_weight3: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv33 = T.match_buffer(p_lv33, (T.int64(1), n, T.int64(4096)))
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i1_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i1_0)
            T.reads(lv33[T.Add(v_i0, T.int64(0)), v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)], linear_weight3[T.int64(0):T.int64(4096), T.int64(0):T.int64(4096)], lv2[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            T.writes(var_T_add_intermediate[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
            lv33_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
            linear_weight3_shared = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="shared")
            for i0_0_i1_1_0_i2_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_1_i2_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for i0_2_i1_1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_1_3_init, i2_3_init, i1_1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused * T.int64(8) + i0_2_i1_1_2_i2_2_fused // T.int64(8) + i1_1_3_init + i1_1_4_init)
                                v_i2_i = T.axis.spatial(T.int64(4096), i2_4_init + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = T.float32(0)
                        for k_0 in range(T.int64(128)):
                            for ax0_ax1_ax2_fused_0 in range(T.int64(8)):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("lv33_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) % T.int64(32))
                                            T.reads(lv33[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                            T.writes(lv33_pad_shared[v0, v1, v2])
                                            lv33_pad_shared[v0, v1, v2] = T.if_then_else(v_i1_o * T.int64(32) + v1 < n, lv33[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], T.float32(0))
                            for ax0_ax1_fused_0 in range(T.int64(8)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("linear_weight3_shared"):
                                            v0 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(32))
                                            v1 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(32))
                                            T.reads(linear_weight3[v0, v1])
                                            T.writes(linear_weight3_shared[v0, v1])
                                            linear_weight3_shared[v0, v1] = linear_weight3[v0, v1]
                            for k_1, i0_3, i1_1_3, i2_3, k_2, i0_4, i1_1_4, i2_4 in T.grid(T.int64(8), T.int64(1), T.int64(1), T.int64(4), T.int64(4), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused * T.int64(8) + i0_2_i1_1_2_i2_2_fused // T.int64(8) + i1_1_3 + i1_1_4)
                                    v_i2_i = T.axis.spatial(T.int64(4096), i2_4 + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3)
                                    v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(4) + k_2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i], lv33_pad_shared[T.int64(0), v_i1_i, v_k_i], linear_weight3_shared[v_i2_i, v_k_i])
                                    T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] + lv33_pad_shared[T.int64(0), v_i1_i, v_k_i] * linear_weight3_shared[v_i2_i, v_k_i]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused * T.int64(8) + i0_2_i1_1_2_i2_2_fused // T.int64(8) + ax1)
                                v2 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + ax2)
                                T.reads(lv2[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], var_NT_matmul_intermediate_pad_local[v0, v1, v2])
                                T.writes(var_T_add_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i1_o * T.int64(32) + v1 and v_i1_o * T.int64(32) + v1 < n:
                                if v_i1_o * T.int64(32) + v1 < n:
                                    var_T_add_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] = lv2[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] + var_NT_matmul_intermediate_pad_local[v0, v1, v2]


@T.prim_func
def fused_NT_matmul1_divide_add_maximum_before(p_lv28: T.handle, p_lv29: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv28 = T.match_buffer(p_lv28, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv29 = T.match_buffer(p_lv29, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, n))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, n))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n))
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, n, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv28[T.int64(0), v_i1, v_i2, v_k], lv29[T.int64(0), v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv28[T.int64(0), v_i1, v_i2, v_k] * lv29[T.int64(0), v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.088388349161020605)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))


@T.prim_func
def fused_NT_matmul1_divide_add_maximum_after(p_lv22: T.handle, p_lv23: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv22 = T.match_buffer(p_lv22, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv23 = T.match_buffer(p_lv23, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, n))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, n))
    # with T.block("root"):
    for i2_0_i3_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32) * ((n + T.int64(31)) // T.int64(32)), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0_i3_0_fused // ((n + T.int64(31)) // T.int64(32)))
            v_i3_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0_i3_0_fused % ((n + T.int64(31)) // T.int64(32)))
            T.reads(lv22[T.int64(0), T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv23[T.int64(0), T.int64(0):T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv5[v_i0, T.int64(0), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            T.writes(var_T_maximum_intermediate[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), scope="local")
            A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), scope="shared")
            B_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), scope="shared")
            for i0_0_i1_0_i2_1_0_i3_1_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_i2_1_1_i3_1_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for i0_2_i1_2_i2_1_2_i3_1_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_3_init, i2_1_3_init, i3_1_3_init, i1_4_init, i2_1_4_init, i3_1_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3_init)
                                v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3_init + i2_1_4_init)
                                v_i3_i = T.axis.spatial(T.int64(32), i3_1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3_init)
                                T.reads()
                                T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = T.float32(0)
                        for k_0 in range(T.int64(16)):
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("A_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3])
                                            T.writes(A_pad_shared[v0, v1, v2, v3])
                                            A_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i2_o * T.int64(32) + v2 < n, lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("B_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3])
                                            T.writes(B_pad_shared[v0, v1, v2, v3])
                                            B_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i3_o * T.int64(32) + v2 < n, lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3], T.float32(0))
                            for k_1, i0_3, i1_3, i2_1_3, i3_1_3, k_2, i0_4, i1_4, i2_1_4, i3_1_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3)
                                    v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3 + i2_1_4)
                                    v_i3_i = T.axis.spatial(T.int64(32), i3_1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3)
                                    v_k_i = T.axis.reduce(T.int64(128), k_0 * T.int64(8) + k_1 * T.int64(2) + k_2)
                                    T.reads(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i], A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i], B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i])
                                    T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] + A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i] * B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i]
                        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("C_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + ax2)
                                v3 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + ax3)
                                T.reads(C_pad_local[v0, v1, v2, v3], lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                T.writes(var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i2_o * T.int64(32) + v2 and v_i2_o * T.int64(32) + v2 < n and T.int64(0) <= v_i3_o * T.int64(32) + v3 and v_i3_o * T.int64(32) + v3 < n:
                                if v_i2_o * T.int64(32) + v2 < n and v_i3_o * T.int64(32) + v3 < n:
                                    var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3] = T.max(C_pad_local[v0, v1, v2, v3] * T.float32(0.088388349161020605) + lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3], T.float32(-3.4028234663852886e+38))

@T.prim_func
def fused_NT_matmul1_divide_add_maximum_with_m_before(p_lv30: T.handle, p_lv31: T.handle, p_lv7: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv30 = T.match_buffer(p_lv30, (T.int64(1), T.int64(32), n, T.int64(128)))
    m = T.int64()
    lv31 = T.match_buffer(p_lv31, (T.int64(1), T.int64(32), m, T.int64(128)))
    lv7 = T.match_buffer(p_lv7, (T.int64(1), T.int64(1), n, m))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv30[v_i0, v_i1, v_i2, v_k], lv31[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv30[v_i0, v_i1, v_i2, v_k] * lv31[v_i0, v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.088388349161020605)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv7[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv7[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))

@T.prim_func
def fused_NT_matmul1_divide_add_maximum_with_m_after(p_lv22: T.handle, p_lv23: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    lv22 = T.match_buffer(p_lv22, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv23 = T.match_buffer(p_lv23, (T.int64(1), T.int64(32), m, T.int64(128)))
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    for i2_0_i3_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32) * ((m + T.int64(31)) // T.int64(32)), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0_i3_0_fused // ((m + T.int64(31)) // T.int64(32)))
            v_i3_o = T.axis.spatial((m + T.int64(31)) // T.int64(32), i2_0_i3_0_fused % ((m + T.int64(31)) // T.int64(32)))
            T.reads(lv22[T.int64(0), T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv23[T.int64(0), T.int64(0):T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv5[v_i0, T.int64(0), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            T.writes(var_T_maximum_intermediate[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), scope="local")
            A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), scope="shared")
            B_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), scope="shared")
            for i0_0_i1_0_i2_1_0_i3_1_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_i2_1_1_i3_1_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for i0_2_i1_2_i2_1_2_i3_1_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_3_init, i2_1_3_init, i3_1_3_init, i1_4_init, i2_1_4_init, i3_1_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3_init)
                                v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3_init + i2_1_4_init)
                                v_i3_i = T.axis.spatial(T.int64(32), i3_1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3_init)
                                T.reads()
                                T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = T.float32(0)
                        for k_0 in range(T.int64(16)):
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("A_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3])
                                            T.writes(A_pad_shared[v0, v1, v2, v3])
                                            A_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i2_o * T.int64(32) + v2 < n, lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("B_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3])
                                            T.writes(B_pad_shared[v0, v1, v2, v3])
                                            B_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i3_o * T.int64(32) + v2 < m, lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3], T.float32(0))
                            for k_1, i0_3, i1_3, i2_1_3, i3_1_3, k_2, i0_4, i1_4, i2_1_4, i3_1_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3)
                                    v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3 + i2_1_4)
                                    v_i3_i = T.axis.spatial(T.int64(32), i3_1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3)
                                    v_k_i = T.axis.reduce(T.int64(128), k_0 * T.int64(8) + k_1 * T.int64(2) + k_2)
                                    T.reads(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i], A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i], B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i])
                                    T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] + A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i] * B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i]
                        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("C_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + ax2)
                                v3 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + ax3)
                                T.reads(C_pad_local[v0, v1, v2, v3], lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                T.writes(var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i2_o * T.int64(32) + v2 and v_i2_o * T.int64(32) + v2 < n and T.int64(0) <= v_i3_o * T.int64(32) + v3 and v_i3_o * T.int64(32) + v3 < n:
                                if v_i2_o * T.int64(32) + v2 < n and v_i3_o * T.int64(32) + v3 < m:
                                    var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3] = T.max(C_pad_local[v0, v1, v2, v3] * T.float32(0.088388349161020605) + lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3], T.float32(-3.4028234663852886e+38))


@T.prim_func
def fused_NT_matmul6_divide1_add2_maximum1_before(lv2732: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32"), p_lv2733: T.handle, p_lv2709: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv2733 = T.match_buffer(p_lv2733, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv2709 = T.match_buffer(p_lv2709, (T.int64(1), T.int64(1), T.int64(1), n))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv2732[T.int64(0), v_i1, v_i2, v_k], lv2733[T.int64(0), v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv2732[T.int64(0), v_i1, v_i2, v_k] * lv2733[T.int64(0), v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.088388349161020605)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2709[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv2709[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))


@T.prim_func
def fused_NT_matmul6_divide1_add2_maximum1_after(lv2732: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32"), p_lv2733: T.handle, p_lv2709: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv2733 = T.match_buffer(p_lv2733, (T.int64(1), T.int64(32), n, T.int64(128)))
    lv2709 = T.match_buffer(p_lv2709, (T.int64(1), T.int64(1), T.int64(1), n))
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32)), scope="local")
    lv2732_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), scope="shared")
    lv2733_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(128)), scope="shared")
    for i3_0 in range((n + T.int64(31)) // T.int64(32)):
        for i0_0_i1_0_i2_0_i3_1_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for i0_1_i1_1_i2_1_i3_1_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_1_2_fused in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_1_3_init, i0_4_init, i1_4_init, i2_4_init, i3_1_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_1_0_fused // T.int64(4) * T.int64(4) + i0_2_i1_2_i2_2_i3_1_2_fused // T.int64(8) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(1), i2_3_init + i2_4_init)
                            v_i3 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i3_0 * T.int64(32) + i0_0_i1_0_i2_0_i3_1_0_fused % T.int64(4) * T.int64(8) + i0_2_i1_2_i2_2_i3_1_2_fused % T.int64(8) + i3_1_3_init + i3_1_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0 in range(T.int64(8)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv2732_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_1_0_fused // T.int64(4) * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(16))
                                        v2 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(16))
                                        T.reads(lv2732[v0, v1, v2, v3])
                                        T.writes(lv2732_shared[v0, v1, v2, v3])
                                        lv2732_shared[v0, v1, v2, v3] = lv2732[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("lv2733_pad_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_1_0_fused // T.int64(4) * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(128))
                                        v2 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i3_0 * T.int64(32) + i0_0_i1_0_i2_0_i3_1_0_fused % T.int64(4) * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(128) // T.int64(16))
                                        v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(16))
                                        T.reads(lv2733[v0, v1, v2, v3])
                                        T.writes(lv2733_pad_shared[v0, v1, v2, v3])
                                        lv2733_pad_shared[v0, v1, v2, v3] = T.if_then_else(v2 < n, lv2733[v0, v1, v2, v3], T.float32(0))
                        for k_1, i0_3, i1_3, i2_3, i3_1_3, k_2, i0_4, i1_4, i2_4, i3_1_4 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(16), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_1_0_fused // T.int64(4) * T.int64(4) + i0_2_i1_2_i2_2_i3_1_2_fused // T.int64(8) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(1), i2_3 + i2_4)
                                v_i3 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i3_0 * T.int64(32) + i0_0_i1_0_i2_0_i3_1_0_fused % T.int64(4) * T.int64(8) + i0_2_i1_2_i2_2_i3_1_2_fused % T.int64(8) + i3_1_3 + i3_1_4)
                                v_k = T.axis.reduce(T.int64(128), k_0 * T.int64(16) + k_1 * T.int64(16) + k_2)
                                T.reads(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3], lv2732_shared[v_i0, v_i1, v_i2, v_k], lv2733_pad_shared[v_i0, v_i1, v_i3, v_k])
                                T.writes(var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate_pad_local[v_i0, v_i1, v_i2, v_i3] + lv2732_shared[v_i0, v_i1, v_i2, v_k] * lv2733_pad_shared[v_i0, v_i1, v_i3, v_k]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("var_NT_matmul_intermediate_pad_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_1_0_fused // T.int64(4) * T.int64(4) + i0_2_i1_2_i2_2_i3_1_2_fused // T.int64(8) + ax1)
                            v2 = T.axis.spatial(T.int64(1), ax2)
                            v3 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), i3_0 * T.int64(32) + i0_0_i1_0_i2_0_i3_1_0_fused % T.int64(4) * T.int64(8) + i0_2_i1_2_i2_2_i3_1_2_fused % T.int64(8) + ax3)
                            T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2, v3])
                            T.writes(var_NT_matmul_intermediate[v0, v1, v2, v3])
                            if v3 < n:
                                var_NT_matmul_intermediate[v0, v1, v2, v3] = var_NT_matmul_intermediate_pad_local[v0, v1, v2, v3]
    for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(n, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_ax1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) // n)
                v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                v_ax3 = T.axis.spatial(n, (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) % n)
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2709[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.088388349161020605) + lv2709[v_ax0, T.int64(0), v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))


@T.prim_func
def fused_NT_matmul2_multiply_before(p_lv43: T.handle, linear_weight6: T.Buffer((T.int64(11008), T.int64(4096)), "float32"), p_lv48: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)))
    lv48 = T.match_buffer(p_lv48, (T.int64(1), n, T.int64(11008)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], linear_weight6[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * linear_weight6[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv48[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = lv48[v_ax0, v_ax1, v_ax2] * var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_NT_matmul2_multiply_after(p_lv37: T.handle, linear_weight6: T.Buffer((T.int64(11008), T.int64(4096)), "float32"), p_lv42: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv37 = T.match_buffer(p_lv37, (T.int64(1), n, T.int64(4096)))
    lv42 = T.match_buffer(p_lv42, (T.int64(1), n, T.int64(11008)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    for i1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i1_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i1_0)
            T.reads(lv37[T.Add(v_i0, T.int64(0)), v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)], linear_weight6[T.int64(0):T.int64(11008), T.int64(0):T.int64(4096)], lv42[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(11008)])
            T.writes(var_T_multiply_intermediate[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(11008)])
            var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(11008)), scope="local")
            lv37_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
            linear_weight6_shared = T.alloc_buffer((T.int64(11008), T.int64(4096)), scope="shared")
            for i0_0_i1_1_0_i2_0_fused in T.thread_binding(T.int64(344), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_1_i2_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                    for i0_2_i1_1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_1_3_init, i2_3_init, i1_1_4_init, i2_4_init in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(2)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3_init * T.int64(2) + i1_1_4_init)
                                v_i2_i = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3_init * T.int64(2) + i2_4_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = T.float32(0)
                        for k_0 in range(T.int64(128)):
                            for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("lv37_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) % T.int64(32))
                                            T.reads(lv37[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                            T.writes(lv37_pad_shared[v0, v1, v2])
                                            lv37_pad_shared[v0, v1, v2] = T.if_then_else(v_i1_o * T.int64(32) + v1 < n, lv37[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], T.float32(0))
                            for ax0_ax1_fused_0 in range(T.int64(8)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("linear_weight6_shared"):
                                            v0 = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(32))
                                            v1 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(32))
                                            T.reads(linear_weight6[v0, v1])
                                            T.writes(linear_weight6_shared[v0, v1])
                                            linear_weight6_shared[v0, v1] = linear_weight6[v0, v1]
                            for k_1, i0_3, i1_1_3, i2_3, k_2, i0_4, i1_1_4, i2_4 in T.grid(T.int64(8), T.int64(1), T.int64(2), T.int64(2), T.int64(4), T.int64(1), T.int64(2), T.int64(2)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3 * T.int64(2) + i1_1_4)
                                    v_i2_i = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3 * T.int64(2) + i2_4)
                                    v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(4) + k_2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i], lv37_pad_shared[T.int64(0), v_i1_i, v_k_i], linear_weight6_shared[v_i2_i, v_k_i])
                                    T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] + lv37_pad_shared[T.int64(0), v_i1_i, v_k_i] * linear_weight6_shared[v_i2_i, v_k_i]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + ax2)
                                T.reads(lv42[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], var_NT_matmul_intermediate_pad_local[v0, v1, v2])
                                T.writes(var_T_multiply_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i1_o * T.int64(32) + v1 and v_i1_o * T.int64(32) + v1 < n:
                                if v_i1_o * T.int64(32) + v1 < n:
                                    var_T_multiply_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] = lv42[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] * var_NT_matmul_intermediate_pad_local[v0, v1, v2]


@T.prim_func
def fused_NT_matmul2_silu_before(p_lv43: T.handle, linear_weight4: T.Buffer((T.int64(11008), T.int64(4096)), "float32"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], linear_weight4[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * linear_weight4[v_i2, v_k]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_NT_matmul2_silu_after(p_lv37: T.handle, linear_weight4: T.Buffer((T.int64(11008), T.int64(4096)), "float32"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv37 = T.match_buffer(p_lv37, (T.int64(1), n, T.int64(4096)))
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)))
    for i1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i1_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i1_0)
            T.reads(lv37[T.Add(v_i0, T.int64(0)), v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)], linear_weight4[T.int64(0):T.int64(11008), T.int64(0):T.int64(4096)])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(11008)])
            var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(11008)), scope="local")
            lv37_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
            linear_weight4_shared = T.alloc_buffer((T.int64(11008), T.int64(4096)), scope="shared")
            for i0_0_i1_1_0_i2_0_fused in T.thread_binding(T.int64(344), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_1_i2_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                    for i0_2_i1_1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_1_3_init, i2_3_init, i1_1_4_init, i2_4_init in T.grid(T.int64(2), T.int64(4), T.int64(2), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3_init * T.int64(2) + i1_1_4_init)
                                v_i2_i = T.axis.spatial(T.int64(11008), i2_4_init + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = T.float32(0)
                        for k_0 in range(T.int64(128)):
                            for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("lv37_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) % T.int64(32))
                                            T.reads(lv37[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                            T.writes(lv37_pad_shared[v0, v1, v2])
                                            lv37_pad_shared[v0, v1, v2] = T.if_then_else(v_i1_o * T.int64(32) + v1 < n, lv37[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], T.float32(0))
                            for ax0_ax1_fused_0 in range(T.int64(8)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("linear_weight4_shared"):
                                            v0 = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(32))
                                            v1 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(32))
                                            T.reads(linear_weight4[v0, v1])
                                            T.writes(linear_weight4_shared[v0, v1])
                                            linear_weight4_shared[v0, v1] = linear_weight4[v0, v1]
                            for k_1, i0_3, i1_1_3, i2_3, k_2, i0_4, i1_1_4, i2_4 in T.grid(T.int64(8), T.int64(1), T.int64(2), T.int64(4), T.int64(4), T.int64(1), T.int64(2), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_1_3 * T.int64(2) + i1_1_4)
                                    v_i2_i = T.axis.spatial(T.int64(11008), i2_4 + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3)
                                    v_k_i = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(4) + k_2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i], lv37_pad_shared[T.int64(0), v_i1_i, v_k_i], linear_weight4_shared[v_i2_i, v_k_i])
                                    T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] + lv37_pad_shared[T.int64(0), v_i1_i, v_k_i] * linear_weight4_shared[v_i2_i, v_k_i]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(11008), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(4) + ax2)
                                T.reads(var_NT_matmul_intermediate_pad_local[v0, v1, v2])
                                T.writes(var_NT_matmul_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i1_o * T.int64(32) + v1 and v_i1_o * T.int64(32) + v1 < n:
                                if v_i1_o * T.int64(32) + v1 < n:
                                    var_NT_matmul_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] = var_NT_matmul_intermediate_pad_local[v0, v1, v2]
    for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            for ax0_ax1_ax2_fused_0 in range((n * T.int64(11008) + T.int64(65535)) // T.int64(65536)):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * T.int64(65536) + ax0_ax1_ax2_fused_1 * T.int64(256) + ax0_ax1_ax2_fused_2) // T.int64(11008))
                    v_ax2 = T.axis.spatial(T.int64(11008), (ax0_ax1_ax2_fused_0 * T.int64(65536) + ax0_ax1_ax2_fused_1 * T.int64(256) + ax0_ax1_ax2_fused_2) % T.int64(11008))
                    T.where((ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1) * T.int64(256) + ax0_ax1_ax2_fused_2 < n * T.int64(11008))
                    T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                    T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                    var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] * T.sigmoid(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])


@T.prim_func
def fused_NT_matmul3_add1_before(p_lv49: T.handle, linear_weight5: T.Buffer((T.int64(4096), T.int64(11008)), "float32"), p_lv42: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(11008)))
    lv42 = T.match_buffer(p_lv42, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv49[v_i0, v_i1, v_k], linear_weight5[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv49[v_i0, v_i1, v_k] * linear_weight5[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv42[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv42[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_NT_matmul3_add1_after(p_lv43: T.handle, linear_weight5: T.Buffer((T.int64(4096), T.int64(11008)), "float32"), p_lv36: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(11008)))
    lv36 = T.match_buffer(p_lv36, (T.int64(1), n, T.int64(4096)))
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)))
    # with T.block("root"):
    for i1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i1_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i1_0)
            T.reads(lv43[T.Add(v_i0, T.int64(0)), v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(11008)], linear_weight5[T.int64(0):T.int64(4096), T.int64(0):T.int64(11008)], lv36[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            T.writes(var_T_add_intermediate[v_i0, v_i1_o * T.int64(32):v_i1_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(4096)])
            var_NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
            lv43_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(11008)), scope="shared")
            linear_weight5_shared = T.alloc_buffer((T.int64(4096), T.int64(11008)), scope="shared")
            for i0_0_i1_1_0_i2_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_1_i2_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for i0_2_i1_1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_1_3_init, i2_3_init, i1_1_4_init, i2_4_init in T.grid(T.int64(2), T.int64(2), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(2) + i1_1_3_init + i1_1_4_init)
                                v_i2_i = T.axis.spatial(T.int64(4096), i2_4_init + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_1_i1_1_1_i2_1_fused % T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(2) + i2_3_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = T.float32(0)
                        for k_0 in range(T.int64(344)):
                            for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                                        with T.block("lv43_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(11008), k_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) % T.int64(32))
                                            T.reads(lv43[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                            T.writes(lv43_pad_shared[v0, v1, v2])
                                            lv43_pad_shared[v0, v1, v2] = T.if_then_else(v_i1_o * T.int64(32) + v1 < n, lv43[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], T.float32(0))
                            for ax0_ax1_fused_0 in range(T.int64(8)):
                                for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("linear_weight5_shared"):
                                            v0 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(32))
                                            v1 = T.axis.spatial(T.int64(11008), k_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(32))
                                            T.reads(linear_weight5[v0, v1])
                                            T.writes(linear_weight5_shared[v0, v1])
                                            linear_weight5_shared[v0, v1] = linear_weight5[v0, v1]
                            for k_1, i0_3, i1_1_3, i2_3, k_2, i0_4, i1_1_4, i2_4 in T.grid(T.int64(8), T.int64(1), T.int64(2), T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(2) + i1_1_3 + i1_1_4)
                                    v_i2_i = T.axis.spatial(T.int64(4096), i2_4 + i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_1_i1_1_1_i2_1_fused % T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(2) + i2_3)
                                    v_k_i = T.axis.reduce(T.int64(11008), k_0 * T.int64(32) + k_1 * T.int64(4) + k_2)
                                    T.reads(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i], lv43_pad_shared[T.int64(0), v_i1_i, v_k_i], linear_weight5_shared[v_i2_i, v_k_i])
                                    T.writes(var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] = var_NT_matmul_intermediate_pad_local[T.int64(0), v_i1_i, v_i2_i] + lv43_pad_shared[T.int64(0), v_i1_i, v_k_i] * linear_weight5_shared[v_i2_i, v_k_i]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(2)):
                            with T.block("var_NT_matmul_intermediate_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_1_i1_1_1_i2_1_fused // T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused // T.int64(8) * T.int64(2) + ax1)
                                v2 = T.axis.spatial(T.int64(4096), i0_0_i1_1_0_i2_0_fused * T.int64(32) + i0_1_i1_1_1_i2_1_fused % T.int64(2) * T.int64(16) + i0_2_i1_1_2_i2_2_fused % T.int64(8) * T.int64(2) + ax2)
                                T.reads(lv36[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2], var_NT_matmul_intermediate_pad_local[v0, v1, v2])
                                T.writes(var_T_add_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i1_o * T.int64(32) + v1 and v_i1_o * T.int64(32) + v1 < n:
                                if v_i1_o * T.int64(32) + v1 < n:
                                    var_T_add_intermediate[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] = lv36[v_i0 + v0, v_i1_o * T.int64(32) + v1, v2] + var_NT_matmul_intermediate_pad_local[v0, v1, v2]



@T.prim_func
def fused_NT_matmul_divide_maximum_minimum_cast_before(lv1605: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), p_lv1606: T.handle, p_lv1582: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv1606 = T.match_buffer(p_lv1606, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
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
            T.reads(lv1605[v_i0, v_i1, v_i2, v_k], lv1606[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1605[v_i0, v_i1, v_i2, v_k] * lv1606[v_i0, v_i1, v_i3, v_k]
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

def fused_NT_matmul_divide_maximum_minimum_cast_sch_func():
    sch = tvm.tir.Schedule(fused_NT_matmul_divide_maximum_minimum_cast_before)
    b_cast = sch.get_block("compute")
    sch.reverse_compute_inline(b_cast)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 1, 1, 32, 1])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l4, [None, 32])
    sch.reorder(l6, l1, l2, l3, l7, l5)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_divide", func_name="main")
    b2 = sch.get_block(name="T_maximum", func_name="main")
    b3 = sch.get_block(name="T_minimum", func_name="main")
    b4 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l5, l6, l7, l8, l9 = sch.get_loops(block=b0)
    v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l15, l16, l17, l18, l19 = sch.split(loop=l5, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
    v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[8, 1, 4, 1, 1])
    l25, l26, l27, l28, l29 = sch.split(loop=l6, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
    v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l35, l36, l37, l38, l39 = sch.split(loop=l7, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
    v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[2, 1, 16, 1, 1])
    l45, l46, l47, l48, l49 = sch.split(loop=l8, factors=[v40, v41, v42, v43, v44], preserve_unit_iters=True)
    v50, v51, v52 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[4, 4, 8])
    l53, l54, l55 = sch.split(loop=l9, factors=[v50, v51, v52], preserve_unit_iters=True)
    sch.reorder(l15, l25, l35, l45, l16, l26, l36, l46, l17, l27, l37, l47, l53, l54, l18, l28, l38, l48, l55, l19, l29, l39, l49)
    l56 = sch.fuse(l15, l25, l35, l45, preserve_unit_iters=True)
    sch.bind(loop=l56, thread_axis="blockIdx.x")
    l57 = sch.fuse(l16, l26, l36, l46, preserve_unit_iters=True)
    sch.bind(loop=l57, thread_axis="vthread.x")
    l58 = sch.fuse(l17, l27, l37, l47, preserve_unit_iters=True)
    sch.bind(loop=l58, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b59 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b59, loop=l58, preserve_unit_loops=True, index=-1)
    b60 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b60, loop=l53, preserve_unit_loops=True, index=-1)
    _, l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b60)
    l69 = sch.fuse(l65, l66, l67, l68, preserve_unit_iters=True)
    v70 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
    b71 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b71, loop=l53, preserve_unit_loops=True, index=-1)
    _, l72, l73, l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b71)
    l80 = sch.fuse(l76, l77, l78, l79, preserve_unit_iters=True)
    v81 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
    sch.reverse_compute_inline(block=b3)
    sch.compute_inline(block=b1)
    v82 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
    sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v82)

    # inline ewise
    sch.reverse_compute_inline(b2)
    # l83, l84, l85, l86 = sch.get_loops(block=b2)
    # l87 = sch.fuse(l83, l84, l85, l86, preserve_unit_iters=True)
    # v88 = sch.sample_categorical(candidates=[32, 64, 128, 256], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    # l89, l90 = sch.split(loop=l87, factors=[None, v88], preserve_unit_iters=True)
    # sch.bind(loop=l89, thread_axis="blockIdx.x")
    # sch.bind(loop=l90, thread_axis="threadIdx.x")

    sch.enter_postproc()
    sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
    _, l91, l92, l93, l94, l95 = sch.get_loops(block=b60)
    l96, l97 = sch.split(loop=l95, factors=[None, 64], preserve_unit_iters=True)
    sch.bind(loop=l97, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch")
    _, l98, l99, l100, l101, l102 = sch.get_loops(block=b71)
    l103, l104 = sch.split(loop=l102, factors=[None, 64], preserve_unit_iters=True)
    sch.bind(loop=l104, thread_axis="threadIdx.x")
    b105 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b105, ann_key="meta_schedule.unroll_explicit")
    _, b106, b107, b108, b109, _ = sch.get_child_blocks(b105)
    _, l111, l112, l113, l114, l115, l116 = sch.get_loops(block=b106)
    sch.annotate(block_or_loop=l111, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l111, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l117, l118, l119, l120, l121, l122 = sch.get_loops(block=b107)
    sch.annotate(block_or_loop=l117, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l117, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132, l133, l134, l135, l136 = sch.get_loops(block=b108)
    sch.annotate(block_or_loop=l123, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l123, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l137, l138, l139, l140, l141, l142, l143 = sch.get_loops(block=b109)
    sch.annotate(block_or_loop=l137, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l137, ann_key="pragma_unroll_explicit", ann_val=1)

    b146 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l147, l148, l149, l150, l151, l152, l153, l154, l155, l156, l157, l158, l159, l160 = sch.get_loops(block=b146)
    sch.bind(l0, "blockIdx.y")
    b161 = sch.decompose_reduction(block=b146, loop=l150)

    b1 = sch.get_block("lv1606_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    return sch.mod["main"].with_attr("tir.is_scheduled", 1)



@T.prim_func
def fused_NT_matmul1_add3_before(p_lv39: T.handle, lv1848: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv39 = T.match_buffer(p_lv39, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv39[v_i0, v_i1, v_k], lv1848[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv39[v_i0, v_i1, v_k] * lv1848[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_NT_matmul1_add3_sch_func():
    sch = tvm.tir.Schedule(fused_NT_matmul1_add3_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 2, 8, 1, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[256, 1, 4, 4, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[256, 1, 16])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 32, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    sch.bind(l0, "blockIdx.y")
    b133 = sch.decompose_reduction(block=b120, loop=l124)

    b1 = sch.get_block("lv39_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul2_divide1_add2_maximum1_before(p_lv28: T.handle, p_lv29: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv28 = T.match_buffer(p_lv28, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv29 = T.match_buffer(p_lv29, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, n), "float16")
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, n), "float16")
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n), "float16")
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, n), "float16")
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, n, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv28[v_i0, v_i1, v_i2, v_k], lv29[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv28[v_i0, v_i1, v_i2, v_k] * lv29[v_i0, v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, n):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))


def fused_NT_matmul2_divide1_add2_maximum1_sch_func(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 1, 32, 32, 1])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l3, [None, 32])
    l8, l9 = sch.split(l4, [None, 32])
    sch.reorder(l6, l8, l1, l2, l7, l9, l5)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_divide", func_name="main")
    b2 = sch.get_block(name="T_add", func_name="main")
    b3 = sch.get_block(name="T_maximum", func_name="main")
    b4 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, _, l5, l6, l7, l8, l9 = sch.get_loops(block=b0)
    v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l15, l16, l17, l18, l19 = sch.split(loop=l5, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
    v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[32, 1, 1, 1, 1])
    l25, l26, l27, l28, l29 = sch.split(loop=l6, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
    v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[2, 1, 8, 1, 2])
    l35, l36, l37, l38, l39 = sch.split(loop=l7, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
    v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[2, 1, 8, 1, 2])
    l45, l46, l47, l48, l49 = sch.split(loop=l8, factors=[v40, v41, v42, v43, v44], preserve_unit_iters=True)
    v50, v51, v52 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[8, 16, 1])
    l53, l54, l55 = sch.split(loop=l9, factors=[v50, v51, v52], preserve_unit_iters=True)
    sch.reorder(l15, l25, l35, l45, l16, l26, l36, l46, l17, l27, l37, l47, l53, l54, l18, l28, l38, l48, l55, l19, l29, l39, l49)
    l56 = sch.fuse(l15, l25, l35, l45, preserve_unit_iters=True)
    sch.bind(loop=l56, thread_axis="blockIdx.x")
    l57 = sch.fuse(l16, l26, l36, l46, preserve_unit_iters=True)
    sch.bind(loop=l57, thread_axis="vthread.x")
    l58 = sch.fuse(l17, l27, l37, l47, preserve_unit_iters=True)
    sch.bind(loop=l58, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b59 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b59, loop=l58, preserve_unit_loops=True, index=-1)
    b60 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b60, loop=l53, preserve_unit_loops=True, index=-1)
    _, _, l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b60)
    l69 = sch.fuse(l65, l66, l67, l68, preserve_unit_iters=True)
    v70 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
    b71 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b71, loop=l53, preserve_unit_loops=True, index=-1)
    _, _, l72, l73, l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b71)
    l80 = sch.fuse(l76, l77, l78, l79, preserve_unit_iters=True)
    v81 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
    sch.reverse_compute_inline(block=b3)
    sch.compute_inline(block=b1)
    v82 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v82)
    l83, l84, l85, l86 = sch.get_loops(block=b2)
    l87 = sch.fuse(l83, l84, l85, l86, preserve_unit_iters=True)
    v88 = sch.sample_categorical(candidates=[32, 64, 128, 256], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    l89, l90 = sch.split(loop=l87, factors=[None, v88], preserve_unit_iters=True)
    sch.bind(loop=l89, thread_axis="blockIdx.x")
    sch.bind(loop=l90, thread_axis="threadIdx.x")
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
    _, _, l91, l92, l93, l94, l95 = sch.get_loops(block=b60)
    l96, l97, l98 = sch.split(loop=l95, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l98)
    sch.bind(loop=l97, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch")
    _, _, l99, l100, l101, l102, l103 = sch.get_loops(block=b71)
    l104, l105, l106 = sch.split(loop=l103, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l106)
    sch.bind(loop=l105, thread_axis="threadIdx.x")
    b107 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b107, ann_key="meta_schedule.unroll_explicit")
    _, _, b108, b109, b110, b111, _, b112 = sch.get_child_blocks(b107)
    _, _, l113, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b108)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b109)
    sch.annotate(block_or_loop=l120, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l120, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l127, l128, l129, l130, l131, l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b110)
    sch.annotate(block_or_loop=l127, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l127, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b111)
    sch.annotate(block_or_loop=l141, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l141, ann_key="pragma_unroll_explicit", ann_val=1)
    l148, l149 = sch.get_loops(block=b112)
    sch.annotate(block_or_loop=l148, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l148, ann_key="pragma_unroll_explicit", ann_val=1)
    b150 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l1, l151, l152, l153, l154, l155, l156, l157, l158, l159, l160, l161, l162, l163, l164 = sch.get_loops(block=b150)
    l2 = sch.fuse(l0, l1)
    sch.bind(l2, "blockIdx.y")
    b165 = sch.decompose_reduction(block=b150, loop=l154)

    b1 = sch.get_block("lv28_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("lv29_pad")
    sch.compute_inline(b2)
    b3 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b3)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul2_divide1_maximum1_minimum1_cast3_before(p_lv28: T.handle, p_lv29: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv28 = T.match_buffer(p_lv28, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    m = T.int64()
    lv29 = T.match_buffer(p_lv29, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv28[v_i0, v_i1, v_i2, v_k], lv29[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv28[v_i0, v_i1, v_i2, v_k] * lv29[v_i0, v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_minimum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

@T.prim_func
def fused_NT_matmul2_divide1_maximum1_minimum1_cast3_after(p_lv22: T.handle, p_lv23: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    n = T.int64()
    m = T.int64()
    lv22 = T.match_buffer(p_lv22, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv23 = T.match_buffer(p_lv23, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
    var_T_maximum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    for i2_0_i3_0_fused in T.thread_binding((n + T.int64(31)) // T.int64(32) * ((m + T.int64(31)) // T.int64(32)), thread="blockIdx.y"):
        with T.block("NT_matmul_o"):
            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            v_i2_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i2_0_i3_0_fused // ((m + T.int64(31)) // T.int64(32)))
            v_i3_o = T.axis.spatial((m + T.int64(31)) // T.int64(32), i2_0_i3_0_fused % ((m + T.int64(31)) // T.int64(32)))
            T.reads(lv22[T.int64(0), T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv23[T.int64(0), T.int64(0):T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32), T.int64(0):T.int64(128)], lv5[v_i0, T.int64(0), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            T.writes(var_T_maximum_intermediate[v_i0, T.int64(0):T.int64(32), v_i2_o * T.int64(32):v_i2_o * T.int64(32) + T.int64(32), v_i3_o * T.int64(32):v_i3_o * T.int64(32) + T.int64(32)])
            C_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16", scope="local")
            A_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), "float16", scope="shared")
            B_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(128)), "float16", scope="shared")
            for i0_0_i1_0_i2_1_0_i3_1_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
                for i0_1_i1_1_i2_1_1_i3_1_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for i0_2_i1_2_i2_1_2_i3_1_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                        for i1_3_init, i2_1_3_init, i3_1_3_init, i1_4_init, i2_1_4_init, i3_1_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_init"):
                                v_i1_i = T.axis.spatial(T.int64(32), i1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3_init)
                                v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3_init + i2_1_4_init)
                                v_i3_i = T.axis.spatial(T.int64(32), i3_1_4_init + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3_init)
                                T.reads()
                                T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = T.float32(0)
                        for k_0 in range(T.int64(16)):
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("A_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3])
                                            T.writes(A_pad_shared[v0, v1, v2, v3])
                                            A_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i2_o * T.int64(32) + v2 < n, lv22[v0, v1, v_i2_o * T.int64(32) + v2, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                        with T.block("B_pad_shared"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4))
                                            v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(8))
                                            v3 = T.axis.spatial(T.int64(128), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                            T.reads(lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3])
                                            T.writes(B_pad_shared[v0, v1, v2, v3])
                                            B_pad_shared[v0, v1, v2, v3] = T.if_then_else(v_i3_o * T.int64(32) + v2 < m, lv23[v0, v1, v_i3_o * T.int64(32) + v2, v3], T.float32(0))
                            for k_1, i0_3, i1_3, i2_1_3, i3_1_3, k_2, i0_4, i1_4, i2_1_4, i3_1_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("NT_matmul_update"):
                                    v_i1_i = T.axis.spatial(T.int64(32), i1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + i1_3)
                                    v_i2_i = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + i2_1_3 + i2_1_4)
                                    v_i3_i = T.axis.spatial(T.int64(32), i3_1_4 + i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + i3_1_3)
                                    v_k_i = T.axis.reduce(T.int64(128), k_0 * T.int64(8) + k_1 * T.int64(2) + k_2)
                                    T.reads(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i], A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i], B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i])
                                    T.writes(C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 256, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] = C_pad_local[T.int64(0), v_i1_i, v_i2_i, v_i3_i] + A_pad_shared[T.int64(0), v_i1_i, v_i2_i, v_k_i] * B_pad_shared[T.int64(0), v_i1_i, v_i3_i, v_k_i]
                        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("C_pad_local"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused // T.int64(4) + ax1)
                                v2 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused // T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused // T.int64(8) + ax2)
                                v3 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_1_0_i3_1_0_fused % T.int64(2) * T.int64(16) + i0_1_i1_1_i2_1_1_i3_1_1_fused % T.int64(2) * T.int64(8) + i0_2_i1_2_i2_1_2_i3_1_2_fused % T.int64(8) + ax3)
                                T.reads(C_pad_local[v0, v1, v2, v3], lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                T.writes(var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3])
                                # if T.int64(0) <= v_i0 and v_i0 < T.int64(1) and T.int64(0) <= v_i2_o * T.int64(32) + v2 and v_i2_o * T.int64(32) + v2 < n and T.int64(0) <= v_i3_o * T.int64(32) + v3 and v_i3_o * T.int64(32) + v3 < n:
                                if v_i2_o * T.int64(32) + v2 < n and v_i3_o * T.int64(32) + v3 < m:
                                    var_T_maximum_intermediate[v_i0 + v0, v1, v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3] = T.Cast("float32", T.min(T.max(C_pad_local[v0, v1, v2, v3] * T.float32(0.088397790055248615), T.float16(-65504)), lv5[v_i0 + v0, T.int64(0), v_i2_o * T.int64(32) + v2, v_i3_o * T.int64(32) + v3]))

def fused_NT_matmul2_divide1_add2_maximum1_sch_func(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 1, 32, 32, 1])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l3, [None, 32])
    l8, l9 = sch.split(l4, [None, 32])
    sch.reorder(l6, l8, l1, l2, l7, l9, l5)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_divide", func_name="main")
    b2 = sch.get_block(name="T_add", func_name="main")
    b3 = sch.get_block(name="T_maximum", func_name="main")
    b4 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, _, l5, l6, l7, l8, l9 = sch.get_loops(block=b0)
    v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l15, l16, l17, l18, l19 = sch.split(loop=l5, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
    v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[32, 1, 1, 1, 1])
    l25, l26, l27, l28, l29 = sch.split(loop=l6, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
    v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[2, 1, 8, 1, 2])
    l35, l36, l37, l38, l39 = sch.split(loop=l7, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
    v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[2, 1, 8, 1, 2])
    l45, l46, l47, l48, l49 = sch.split(loop=l8, factors=[v40, v41, v42, v43, v44], preserve_unit_iters=True)
    v50, v51, v52 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[8, 16, 1])
    l53, l54, l55 = sch.split(loop=l9, factors=[v50, v51, v52], preserve_unit_iters=True)
    sch.reorder(l15, l25, l35, l45, l16, l26, l36, l46, l17, l27, l37, l47, l53, l54, l18, l28, l38, l48, l55, l19, l29, l39, l49)
    l56 = sch.fuse(l15, l25, l35, l45, preserve_unit_iters=True)
    sch.bind(loop=l56, thread_axis="blockIdx.x")
    l57 = sch.fuse(l16, l26, l36, l46, preserve_unit_iters=True)
    sch.bind(loop=l57, thread_axis="vthread.x")
    l58 = sch.fuse(l17, l27, l37, l47, preserve_unit_iters=True)
    sch.bind(loop=l58, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b59 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b59, loop=l58, preserve_unit_loops=True, index=-1)
    b60 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b60, loop=l53, preserve_unit_loops=True, index=-1)
    _, _, l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b60)
    l69 = sch.fuse(l65, l66, l67, l68, preserve_unit_iters=True)
    v70 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
    b71 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b71, loop=l53, preserve_unit_loops=True, index=-1)
    _, _, l72, l73, l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b71)
    l80 = sch.fuse(l76, l77, l78, l79, preserve_unit_iters=True)
    v81 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
    sch.reverse_compute_inline(block=b3)
    sch.compute_inline(block=b1)
    v82 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v82)
    l83, l84, l85, l86 = sch.get_loops(block=b2)
    l87 = sch.fuse(l83, l84, l85, l86, preserve_unit_iters=True)
    v88 = sch.sample_categorical(candidates=[32, 64, 128, 256], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    l89, l90 = sch.split(loop=l87, factors=[None, v88], preserve_unit_iters=True)
    sch.bind(loop=l89, thread_axis="blockIdx.x")
    sch.bind(loop=l90, thread_axis="threadIdx.x")
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
    _, _, l91, l92, l93, l94, l95 = sch.get_loops(block=b60)
    l96, l97, l98 = sch.split(loop=l95, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l98)
    sch.bind(loop=l97, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch")
    _, _, l99, l100, l101, l102, l103 = sch.get_loops(block=b71)
    l104, l105, l106 = sch.split(loop=l103, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l106)
    sch.bind(loop=l105, thread_axis="threadIdx.x")
    b107 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b107, ann_key="meta_schedule.unroll_explicit")
    _, _, b108, b109, b110, b111, _, b112 = sch.get_child_blocks(b107)
    _, _, l113, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b108)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b109)
    sch.annotate(block_or_loop=l120, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l120, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l127, l128, l129, l130, l131, l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b110)
    sch.annotate(block_or_loop=l127, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l127, ann_key="pragma_unroll_explicit", ann_val=1)
    _, _, l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b111)
    sch.annotate(block_or_loop=l141, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l141, ann_key="pragma_unroll_explicit", ann_val=1)
    l148, l149 = sch.get_loops(block=b112)
    sch.annotate(block_or_loop=l148, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l148, ann_key="pragma_unroll_explicit", ann_val=1)
    b150 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l1, l151, l152, l153, l154, l155, l156, l157, l158, l159, l160, l161, l162, l163, l164 = sch.get_loops(block=b150)
    l2 = sch.fuse(l0, l1)
    sch.bind(l2, "blockIdx.y")
    b165 = sch.decompose_reduction(block=b150, loop=l154)

    b1 = sch.get_block("lv28_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("lv29_pad")
    sch.compute_inline(b2)
    b3 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b3)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul3_multiply1_before(p_lv43: T.handle, lv1866: T.Buffer((T.int64(11008), T.int64(4096)), "float16"), p_lv48: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)), "float16")
    lv48 = T.match_buffer(p_lv48, (T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], lv1866[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * lv1866[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv48[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = lv48[v_ax0, v_ax1, v_ax2] * var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_NT_matmul3_multiply1_sch_func():
    sch = tvm.tir.Schedule(fused_NT_matmul3_multiply1_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_multiply", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 8, 2, 2])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[344, 4, 8, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[128, 16, 2])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    sch.bind(l0, "blockIdx.y")
    b133 = sch.decompose_reduction(block=b120, loop=l124)

    b1 = sch.get_block("lv43_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul3_silu1_before(p_lv43: T.handle, lv1857: T.Buffer((T.int64(11008), T.int64(4096)), "float16"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv43 = T.match_buffer(p_lv43, (T.int64(1), n, T.int64(4096)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(11008), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv43[v_i0, v_i1, v_k], lv1857[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * lv1857[v_i2, v_k]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


def fused_NT_matmul3_silu1_sch_func():
    sch = tvm.tir.Schedule(fused_NT_matmul3_silu1_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="compute", func_name="main")
    b2 = sch.get_block(name="T_multiply", func_name="main")
    b3 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l4, l5, l6, l7 = sch.get_loops(block=b0)
    v8, v9, v10, v11, v12 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l13, l14, l15, l16, l17 = sch.split(loop=l4, factors=[v8, v9, v10, v11, v12], preserve_unit_iters=True)
    v18, v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 8, 4, 1])
    l23, l24, l25, l26, l27 = sch.split(loop=l5, factors=[v18, v19, v20, v21, v22], preserve_unit_iters=True)
    v28, v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[344, 4, 8, 1, 1])
    l33, l34, l35, l36, l37 = sch.split(loop=l6, factors=[v28, v29, v30, v31, v32], preserve_unit_iters=True)
    v38, v39, v40 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64, decision=[128, 16, 2])
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
    _, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b48)
    l56 = sch.fuse(l53, l54, l55, preserve_unit_iters=True)
    v57 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch", ann_val=v57)
    b58 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b58, loop=l41, preserve_unit_loops=True, index=-1)
    _, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b58)
    l65 = sch.fuse(l63, l64, preserve_unit_iters=True)
    v66 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v66)
    sch.compute_inline(block=b1)
    v67 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
    sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v67)

    # reverse compute inline the silu part
    sch.reverse_compute_inline(b2)
    # l68, l69, l70 = sch.get_loops(block=b2)
    # l71 = sch.fuse(l68, l69, l70, preserve_unit_iters=True)
    # l72, l73, l74 = sch.split(loop=l71, factors=[None, 256, 256], preserve_unit_iters=True)
    #sch.reorder(l73, l74, l72)
    # sch.bind(loop=l73, thread_axis="blockIdx.x")
    # sch.bind(loop=l74, thread_axis="threadIdx.x")
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b48)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
    _, l83, l84, l85, l86, l87 = sch.get_loops(block=b58)
    l88, l89, l90 = sch.split(loop=l87, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l90)
    sch.bind(loop=l89, thread_axis="threadIdx.x")
    b91 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b91, ann_key="meta_schedule.unroll_explicit")
    _, b92, b93, b94, b95, _ = sch.get_child_blocks(b91)
    _, l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b92)
    sch.annotate(block_or_loop=l97, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l97, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l104, l105, l106, l107, l108, l109, l110 = sch.get_loops(block=b93)
    sch.annotate(block_or_loop=l104, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l104, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l111, l112, l113, l114, l115, l116, l117, l118, l119, l120, l121, l122 = sch.get_loops(block=b94)
    sch.annotate(block_or_loop=l111, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l111, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l123, l124, l125, l126, l127, l128 = sch.get_loops(block=b95)
    sch.annotate(block_or_loop=l123, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l123, ann_key="pragma_unroll_explicit", ann_val=1)
    # l129, l130, l131 = sch.get_loops(block=b96)
    # sch.annotate(block_or_loop=l129, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    # sch.annotate(block_or_loop=l129, ann_key="pragma_unroll_explicit", ann_val=1)
    b132 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l133, l134, l135, l136, l137, l138, l139, l140, l141, l142, l143, l144 = sch.get_loops(block=b132)
    sch.bind(l0, "blockIdx.y")
    b145 = sch.decompose_reduction(block=b132, loop=l136)

    b1 = sch.get_block("lv43_pad")
    sch.compute_inline(b1)

    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)

    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_NT_matmul4_add3_before(p_lv49: T.handle, lv1875: T.Buffer((T.int64(4096), T.int64(11008)), "float16"), p_lv42: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(11008)), "float16")
    lv42 = T.match_buffer(p_lv42, (T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv49[v_i0, v_i1, v_k], lv1875[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv49[v_i0, v_i1, v_k] * lv1875[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv42[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv42[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]


def fused_NT_matmul4_add3_sch_func():
    sch = tvm.tir.Schedule(fused_NT_matmul4_add3_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l3, l4, l5, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 8, 1, 4])
    l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[128, 2, 8, 2, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[688, 16, 1])
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
    _, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
    l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
    v56 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
    b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
    _, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
    l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
    v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
    sch.reverse_compute_inline(block=b1)
    v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
    _, l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
    l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l74)
    sch.bind(loop=l73, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
    l80, l81, l82 = sch.split(loop=l79, factors=[None, 64, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l82)
    sch.bind(loop=l81, thread_axis="threadIdx.x")
    b83 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
    _, b84, b85, b86, b87, _ = sch.get_child_blocks(b83)
    _, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
    b120 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
    sch.bind(l0, "blockIdx.y")
    b133 = sch.decompose_reduction(block=b120, loop=l124)

    b1 = sch.get_block("lv49_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("var_NT_matmul_intermediate_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def matmul1_fp16_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k], rxplaceholder_1[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + rxplaceholder[v_i0, v_i1, v_i2, v_k] * rxplaceholder_1[v_i0, v_i1, v_k, v_i3]


def matmul1_fp16_sch_func():
    sch = tvm.tir.Schedule(matmul1_fp16_before)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 1, 1, 1, 128])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    sch.split(l5, [None, 128])

    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l2, l3, l4, l5, ko, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[2, 1, 16, 1, 1])
    l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[8, 1, 16, 1, 1])
    l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
    v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[4, 16, 2])
    l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
    sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, ko, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
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
    l58, l59, l60, _, l61, l62, l63, l64, l65 = sch.get_loops(block=b57)
    l66 = sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
    v67 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
    b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
    l69, l70, l71, _, l72, l73, l74, l75, l76 = sch.get_loops(block=b68)
    l77 = sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
    v78 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
    v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    l80, l81, l82, _, l83, l84 = sch.get_loops(block=b57)
    l85, l86 = sch.split(loop=l84, factors=[None, 256], preserve_unit_iters=True)
    sch.bind(loop=l86, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
    l87, l88, l89, _, l90, l91 = sch.get_loops(block=b68)
    l92, l93, l94 = sch.split(loop=l91, factors=[None, 256, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l94)
    sch.bind(loop=l93, thread_axis="threadIdx.x")
    b95 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b95, ann_key="meta_schedule.unroll_explicit")
    _, _, b96, b97, b98, b99 = sch.get_child_blocks(b95)
    l100, l101, l102, _, l103, l104, l105 = sch.get_loops(block=b96)
    sch.annotate(block_or_loop=l100, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l100, ann_key="pragma_unroll_explicit", ann_val=1)
    l106, l107, l108, _, l109, l110, l111, l112 = sch.get_loops(block=b97)
    sch.annotate(block_or_loop=l106, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l106, ann_key="pragma_unroll_explicit", ann_val=1)
    l113, l114, l115, _, l116, l117, l118, l119, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b98)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    l127, l128, l129, l130, l131, l132, l133 = sch.get_loops(block=b99)
    sch.annotate(block_or_loop=l127, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l127, ann_key="pragma_unroll_explicit", ann_val=1)
    b134 = sch.get_block(name="matmul", func_name="main")
    l135, l136, l137, ko, l138, l139, l140, l141, l142, l143, l144, l145, l146, l147, l148 = sch.get_loops(block=b134)
    b149 = sch.decompose_reduction(block=b134, loop=ko)

    b1 = sch.get_block("rxplaceholder_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("rxplaceholder_1_pad")
    sch.compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def matmul8_fp16_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, n), "float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k], rxplaceholder_1[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + rxplaceholder[v_i0, v_i1, v_i2, v_k] * rxplaceholder_1[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul8_with_m_fp16_before(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    m = T.int64()
    rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), T.int64(32), n, m), "float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), m):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_k], rxplaceholder_1[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + rxplaceholder[v_i0, v_i1, v_i2, v_k] * rxplaceholder_1[v_i0, v_i1, v_k, v_i3]

def matmul8_fp16_sch_func(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block("matmul")
    sch.pad_einsum(b0, [1, 1, 32, 1, 128])
    l1, l2, l3, l4, l5 = sch.get_loops(b0)
    l6, l7 = sch.split(l3, [None, 32])
    l8, l9 = sch.split(l5, [None, 128])
    sch.reorder(l6, l1, l2, l7, l4, l8, l9)

    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l2, l3, l4, l5, ko, l6 = sch.get_loops(block=b0)
    v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[32, 1, 1, 1, 1])
    l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 4, 2, 4])
    l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
    v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[4, 1, 16, 2, 1])
    l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
    v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[16, 1, 8])
    l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
    sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, ko, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
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
    _, l58, l59, l60, _, l61, l62, l63, l64, l65 = sch.get_loops(block=b57)
    l66 = sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
    v67 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
    b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
    _, l69, l70, l71, _, l72, l73, l74, l75, l76 = sch.get_loops(block=b68)
    l77 = sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
    v78 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
    v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
    _, l80, l81, l82, _, l83, l84 = sch.get_loops(block=b57)
    l85, l86, l87 = sch.split(loop=l84, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l87)
    sch.bind(loop=l86, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
    _, l88, l89, l90, _, l91, l92 = sch.get_loops(block=b68)
    l93, l94 = sch.split(loop=l92, factors=[None, 64], preserve_unit_iters=True)
    sch.bind(loop=l94, thread_axis="threadIdx.x")
    b95 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b95, ann_key="meta_schedule.unroll_explicit")
    _, _, b96, b97, b98, b99, _ = sch.get_child_blocks(b95)
    _, l100, l101, l102, _, l103, l104, l105, l106 = sch.get_loops(block=b96)
    sch.annotate(block_or_loop=l100, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l100, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l107, l108, l109, _, l110, l111, l112 = sch.get_loops(block=b97)
    sch.annotate(block_or_loop=l107, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l107, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l113, l114, l115, _, l116, l117, l118, l119, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b98)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l127, l128, l129, l130, l131, l132, l133 = sch.get_loops(block=b99)
    sch.annotate(block_or_loop=l127, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l127, ann_key="pragma_unroll_explicit", ann_val=1)
    b134 = sch.get_block(name="matmul", func_name="main")
    l0, l135, l136, l137, ko, l138, l139, l140, l141, l142, l143, l144, l145, l146, l147, l148 = sch.get_loops(block=b134)
    sch.bind(l0, "blockIdx.y")
    b149 = sch.decompose_reduction(block=b134, loop=ko)

    b1 = sch.get_block("rxplaceholder_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("rxplaceholder_1_pad")
    sch.compute_inline(b2)
    b3 = sch.get_block("matmul_pad")
    sch.reverse_compute_inline(b3)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def NT_matmul1_fp16_before(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), var_NT_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (T.int64(1), n, T.int64(4096)), "float16")
    NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(rxplaceholder_1[v_i0, v_i1, v_k], rxplaceholder[v_i2, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2] = T.float16(0)
            NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + rxplaceholder_1[v_i0, v_i1, v_k] * rxplaceholder[v_i2, v_k]


def NT_matmul1_fp16_sch_func():
    sch = tvm.tir.Schedule(NT_matmul1_fp16_before)
    b0 = sch.get_block("NT_matmul")
    sch.pad_einsum(b0, [1, 32, 1, 1])
    l1, l2, l3, l4 = sch.get_loops(b0)
    l5, l6 = sch.split(l2, [None, 32])
    sch.reorder(l5, l1, l6, l3, l4)

    b0 = sch.get_block(name="NT_matmul", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    _, l2, l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l11, l12, l13, l14, l15 = sch.split(loop=l2, factors=[v6, v7, v8, v9, v10], preserve_unit_iters=True)
    v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 4, 2, 4])
    l21, l22, l23, l24, l25 = sch.split(loop=l3, factors=[v16, v17, v18, v19, v20], preserve_unit_iters=True)
    v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[128, 1, 16, 1, 2])
    l31, l32, l33, l34, l35 = sch.split(loop=l4, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
    v36, v37, v38 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64, decision=[512, 2, 4])
    l39, l40, l41 = sch.split(loop=l5, factors=[v36, v37, v38], preserve_unit_iters=True)
    sch.reorder(l11, l21, l31, l12, l22, l32, l13, l23, l33, l39, l40, l14, l24, l34, l41, l15, l25, l35)
    l42 = sch.fuse(l11, l21, l31, preserve_unit_iters=True)
    sch.bind(loop=l42, thread_axis="blockIdx.x")
    l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="vthread.x")
    l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
    sch.bind(loop=l44, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
    b45 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b45, loop=l44, preserve_unit_loops=True, index=-1)
    b46 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b46, loop=l39, preserve_unit_loops=True, index=-1)
    _, l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
    l54 = sch.fuse(l51, l52, l53, preserve_unit_iters=True)
    v55 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
    b56 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
    sch.compute_at(block=b56, loop=l39, preserve_unit_loops=True, index=-1)
    _, l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b56)
    l63 = sch.fuse(l61, l62, preserve_unit_iters=True)
    v64 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch", ann_val=v64)
    v65 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v65)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
    _, l66, l67, l68, l69, l70 = sch.get_loops(block=b46)
    l71, l72, l73 = sch.split(loop=l70, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l73)
    sch.bind(loop=l72, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch")
    _, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)
    l79, l80, l81 = sch.split(loop=l78, factors=[None, 64, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l81)
    sch.bind(loop=l80, thread_axis="threadIdx.x")
    b82 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")
    _, b83, b84, b85, b86, _ = sch.get_child_blocks(b82)
    _, l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b83)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l94, l95, l96, l97, l98, l99, l100 = sch.get_loops(block=b84)
    sch.annotate(block_or_loop=l94, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l94, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l101, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b85)
    sch.annotate(block_or_loop=l101, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l101, ann_key="pragma_unroll_explicit", ann_val=1)
    _, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b86)
    sch.annotate(block_or_loop=l113, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l113, ann_key="pragma_unroll_explicit", ann_val=1)
    b119 = sch.get_block(name="NT_matmul", func_name="main")
    l0, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b119)
    sch.bind(l0, "blockIdx.y")
    b132 = sch.decompose_reduction(block=b119, loop=l123)

    b1 = sch.get_block("rxplaceholder_1_pad")
    sch.compute_inline(b1)
    b2 = sch.get_block("NT_matmul_pad")
    sch.reverse_compute_inline(b2)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def decode6(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)))
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def decode7(rxplaceholder: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(11008)), "uint32"), T_transpose: T.Buffer((T.int64(11008), T.int64(4096)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)))
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def decode8(rxplaceholder: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(344), T.int64(4096)), "uint32"), T_transpose: T.Buffer((T.int64(4096), T.int64(11008)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)))
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def decode4_fp16(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(4096)), "float16"), rxplaceholder_2: T.Buffer((T.int64(128), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j], rxplaceholder_2[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * rxplaceholder_1[v_i // T.int64(32), v_j] + rxplaceholder_2[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

@T.prim_func
def decode5_fp16(rxplaceholder: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(11008)), "float16"), rxplaceholder_2: T.Buffer((T.int64(128), T.int64(11008)), "float16"), T_transpose: T.Buffer((T.int64(11008), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j], rxplaceholder_2[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * rxplaceholder_1[v_i // T.int64(32), v_j] + rxplaceholder_2[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

@T.prim_func
def decode6_fp16(rxplaceholder: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(344), T.int64(4096)), "float16"), rxplaceholder_2: T.Buffer((T.int64(344), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(11008)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j], rxplaceholder_2[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * rxplaceholder_1[v_i // T.int64(32), v_j] + rxplaceholder_2[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def decode_int3_fp16(A: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), B: T.Buffer((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(10), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode_1[v_i, v_j])
            decode_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode_1[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode_1[v_ax1, v_ax0]

@T.prim_func
def decode1_int3_fp16(A: T.Buffer((T.int64(412), T.int64(11008)), "uint32"), B: T.Buffer((T.int64(103), T.int64(11008)), "float16"), T_transpose: T.Buffer((T.int64(11008), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(10), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

@T.prim_func
def decode2_int3_fp16(A: T.Buffer((T.int64(1104), T.int64(4096)), "uint32"), B: T.Buffer((T.int64(276), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(11008)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(10), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def decode_int3_int16_fp16(A: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), B: T.Buffer((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(5), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode_1[v_i, v_j])
            decode_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", A[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode_1[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode_1[v_ax1, v_ax0]

@T.prim_func
def decode1_int3_int16_fp16(A: T.Buffer((T.int64(824), T.int64(11008)), "uint16"), B: T.Buffer((T.int64(103), T.int64(11008)), "float16"), T_transpose: T.Buffer((T.int64(11008), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(5), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", A[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

@T.prim_func
def decode2_int3_int16_fp16(A: T.Buffer((T.int64(2208), T.int64(4096)), "uint16"), B: T.Buffer((T.int64(276), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(11008)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A[v_i // T.int64(5), v_j], B[v_i // T.int64(40), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", A[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(T_transpose[v_ax0, v_ax1])
            T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


def decode_sch_func(orig_func):
    sch = tvm.tir.Schedule(orig_func)
    b0 = sch.get_block(name="decode", func_name="main")
    l1, l2 = sch.get_loops(block=b0)
    l3, l4 = sch.split(loop=l1, factors=[None, 8], preserve_unit_iters=True)
    v5, v6, v7 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=4, decision=[32, 8, 2])
    l8, l9, l10 = sch.split(loop=l3, factors=[v5, v6, v7], preserve_unit_iters=True)
    v11, v12 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=16, decision=[256, 16])
    l13, l14 = sch.split(loop=l2, factors=[v11, v12], preserve_unit_iters=True)
    sch.reorder(l8, l13, l9, l14, l10, l4)
    sch.bind(loop=l8, thread_axis="blockIdx.y")
    sch.bind(loop=l13, thread_axis="blockIdx.x")
    sch.bind(loop=l9, thread_axis="threadIdx.y")
    sch.bind(loop=l14, thread_axis="threadIdx.x")
    sch.unroll(loop=l4)
    b15 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="shared")
    sch.compute_inline(block=b15)
    b16 = sch.get_block(name="T_transpose", func_name="main")
    sch.reverse_compute_at(block=b16, loop=l13, preserve_unit_loops=True, index=-1)
    b17 = sch.get_block(name="T_transpose", func_name="main")
    l18, l19, l20, l21 = sch.get_loops(block=b17)
    l22 = sch.fuse(l20, l21, preserve_unit_iters=True)
    l23, l24, l25 = sch.split(loop=l22, factors=[None, v12, 4], preserve_unit_iters=True)
    sch.bind(loop=l24, thread_axis="threadIdx.x")
    sch.vectorize(loop=l25)
    sch.storage_align(block=b0, buffer_index=0, axis=0, factor=32, offset=1)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


@T.prim_func
def fused_decode3_matmul1_before(lv2931: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv2932: T.Buffer((T.int64(128), T.int64(32000)), "uint32"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)))
        for i, j in T.grid(T.int64(4096), T.int64(32000)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv2931[v_i // T.int64(8), v_j], lv2932[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv2931[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv2932[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv2932[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1511[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1511[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode3_matmul1_after(lv1123: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv1124: T.Buffer((T.int64(128), T.int64(32000)), "uint32"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_pad_local = T.alloc_buffer((T.int64(4096), T.int64(32000)), scope="local")
    var_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), scope="local")
    lv1511_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(125), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv1511_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv1511[v0, v1, v2])
                                T.writes(lv1511_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv1511_shared[v0, v1, v2] = lv1511[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("var_decode_intermediate_pad"):
                                    v0 = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v1 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1123[v0 // T.int64(8), v1], lv1124[v0 // T.int64(32), v1])
                                    T.writes(var_decode_intermediate_pad_local[v0, v1])
                                    var_decode_intermediate_pad_local[v0, v1] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1123[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1124[v0 // T.int64(32), v1], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1124[v0 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv1511_shared[v_i0, v_i1, v_k], var_decode_intermediate_pad_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv1511_shared[v_i0, v_i1, v_k] * var_decode_intermediate_pad_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_pad_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_pad_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_pad_local[v0, v1, v2]


@T.prim_func
def fused_decode4_fused_matmul5_add3_before(lv3184: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv3185: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), lv452: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)))
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)))
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3184[v_i // T.int64(8), v_j], lv3185[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv3184[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv3185[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv3185[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv452[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv452[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2710[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv2710[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode4_fused_matmul5_add3_after(lv1143: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv1144: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="local")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local")
    lv3_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv3_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv3[v0, v1, v2])
                                T.writes(lv3_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv3_shared[v0, v1, v2] = lv3[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1143[v_j // T.int64(8), v_i], lv1144[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1143[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1144[v_j // T.int64(32), v_i], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1144[v_j // T.int64(32), v_i], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv3_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv3_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv2710[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv2710[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode4_matmul5_before(lv3166: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv3167: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)))
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3166[v_i // T.int64(8), v_j], lv3167[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv3166[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv3167[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv3167[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2712[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2712[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode4_matmul5_after(lv1128: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv1129: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="local")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local")
    lv2712_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2712_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2712[v0, v1, v2])
                                T.writes(lv2712_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2712_shared[v0, v1, v2] = lv2712[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1128[v_j // T.int64(8), v_i], lv1129[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1128[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1129[v_j // T.int64(32), v_i], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1129[v_j // T.int64(32), v_i], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2712_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2712_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_before(lv1617: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv1618: T.Buffer((T.int64(128), T.int64(11008)), "uint32"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)))
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)))
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1617[v_i // T.int64(8), v_j], lv1618[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1617[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1618[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1618[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2749[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2749[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv4[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv4[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_after(lv1153: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv1154: T.Buffer((T.int64(128), T.int64(11008)), "uint32"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(11008)), scope="local")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2749_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2749[v0, v1, v2])
                                T.writes(lv2749_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2749_shared[v0, v1, v2] = lv2749[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1153[v_j // T.int64(8), v_i], lv1154[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1153[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1154[v_j // T.int64(32), v_i], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1154[v_j // T.int64(32), v_i], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv5[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv5[v0, v1, v2] * var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_before(lv1611: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv1612: T.Buffer((T.int64(128), T.int64(11008)), "uint32"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)))
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)))
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)))
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1611[v_i // T.int64(8), v_j], lv1612[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1611[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1612[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1612[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2749[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2749[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_after(lv1148: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv1149: T.Buffer((T.int64(128), T.int64(11008)), "uint32"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(11008)), scope="local")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2749_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2749[v0, v1, v2])
                                T.writes(lv2749_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2749_shared[v0, v1, v2] = lv2749[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1148[v_j // T.int64(8), v_i], lv1149[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1148[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1149[v_j // T.int64(32), v_i], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1149[v_j // T.int64(32), v_i], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2] * T.sigmoid(var_matmul_intermediate_local[v0, v1, v2])


@T.prim_func
def fused_decode6_fused_matmul9_add3_before(lv1623: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv1624: T.Buffer((T.int64(344), T.int64(4096)), "uint32"), lv230: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32"), lv228: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)))
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)))
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1623[v_i // T.int64(8), v_j], lv1624[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1623[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1624[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1624[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv230[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv230[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv228[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv228[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_after(lv1158: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv1159: T.Buffer((T.int64(344), T.int64(4096)), "uint32"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float32"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate_local = T.alloc_buffer((T.int64(11008), T.int64(4096)), scope="local")
        var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local")
        lv6_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="shared")
        for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0_0 in range(T.int64(2)):
                        for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(22)):
                            for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                                with T.block("lv6_shared"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2 = T.axis.spatial(T.int64(11008), k_0_0 * T.int64(5504) + (ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1))
                                    T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(5504))
                                    T.reads(lv6[v0, v1, v2])
                                    T.writes(lv6_shared[v0, v1, v2])
                                    T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                    lv6_shared[v0, v1, v2] = lv6[v0, v1, v2]
                        for k_0_1 in range(T.int64(86)):
                            for ax0_0 in range(T.int64(8)):
                                for ax0_1 in T.unroll(T.int64(8)):
                                    for ax1 in range(T.int64(1)):
                                        with T.block("decode"):
                                            v_j = T.axis.spatial(T.int64(11008), k_0_0 * T.int64(5504) + k_0_1 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                            v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                            T.reads(lv1158[v_j // T.int64(8), v_i], lv1159[v_j // T.int64(32), v_i])
                                            T.writes(var_decode_intermediate_local[v_j, v_i])
                                            var_decode_intermediate_local[v_j, v_i] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1158[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1159[v_j // T.int64(32), v_i], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1159[v_j // T.int64(32), v_i], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                            for k_0_2_k_1_fused in range(T.int64(64)):
                                with T.block("matmul_update"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                                    v_k = T.axis.reduce(T.int64(11008), k_0_0 * T.int64(5504) + k_0_1 * T.int64(64) + k_0_2_k_1_fused)
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv6_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv6_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                            T.reads(lv4[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode3_matmul1_fp16_before(lv5865: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv5866: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv5867: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv2705: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv5865[v_i // T.int64(8), v_j], lv5866[v_i // T.int64(32), v_j], lv5867[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv5865[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv5866[v_i // T.int64(32), v_j] + lv5867[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2705[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2705[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode3_matmul1_fp16_after(lv1123: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv5866: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv5867: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_pad_local = T.alloc_buffer((T.int64(4096), T.int64(32000)), scope="local", dtype="float16")
    var_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), scope="local", dtype="float16")
    lv1511_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(125), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv1511_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv1511[v0, v1, v2])
                                T.writes(lv1511_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv1511_shared[v0, v1, v2] = lv1511[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("var_decode_intermediate_pad"):
                                    v0 = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v1 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1123[v0 // T.int64(8), v1], lv5866[v0 // T.int64(32), v1], lv5867[v0 // T.int64(32), v1])
                                    T.writes(var_decode_intermediate_pad_local[v0, v1])
                                    var_decode_intermediate_pad_local[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1123[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8) * T.int64(4))), T.uint32(15))) * lv5866[v0 // T.int64(32), v1] + lv5867[v0 // T.int64(32), v1]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv1511_shared[v_i0, v_i1, v_k], var_decode_intermediate_pad_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv1511_shared[v_i0, v_i1, v_k] * var_decode_intermediate_pad_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_pad_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_pad_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_pad_local[v0, v1, v2]


@T.prim_func
def fused_decode3_matmul1_cast_fp16_before(lv1803: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv1804: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv1805: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv3025: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1803[v_i // T.int64(8), v_j], lv1804[v_i // T.int64(32), v_j], lv1805[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1803[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv1804[v_i // T.int64(32), v_j] + lv1805[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv3025[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv3025[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_decode3_matmul1_cast_fp16_after(lv1123: T.Buffer((T.int64(512), T.int64(32000)), "uint32"), lv5866: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv5867: T.Buffer((T.int64(128), T.int64(32000)), "float16"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_pad_local = T.alloc_buffer((T.int64(4096), T.int64(32000)), scope="local", dtype="float16")
    var_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), scope="local", dtype="float16")
    lv1511_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(125), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv1511_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv1511[v0, v1, v2])
                                T.writes(lv1511_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv1511_shared[v0, v1, v2] = lv1511[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("var_decode_intermediate_pad"):
                                    v0 = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v1 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1123[v0 // T.int64(8), v1], lv5866[v0 // T.int64(32), v1], lv5867[v0 // T.int64(32), v1])
                                    T.writes(var_decode_intermediate_pad_local[v0, v1])
                                    var_decode_intermediate_pad_local[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1123[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8) * T.int64(4))), T.uint32(15))) * lv5866[v0 // T.int64(32), v1] + lv5867[v0 // T.int64(32), v1]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv1511_shared[v_i0, v_i1, v_k], var_decode_intermediate_pad_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv1511_shared[v_i0, v_i1, v_k] * var_decode_intermediate_pad_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_pad_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_pad_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = T.Cast("float32", var_matmul_intermediate_pad_local[v0, v1, v2])


@T.prim_func
def fused_decode4_fused_matmul5_add3_fp16_before(lv35: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv36: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv37: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv2: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv35[v_i // T.int64(8), v_j], lv36[v_i // T.int64(32), v_j], lv37[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv35[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv36[v_i // T.int64(32), v_j] + lv37[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2710[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv2710[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode4_fused_matmul5_add3_fp16_after(lv1143: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv36: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv37: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv3_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv3_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv3[v0, v1, v2])
                                T.writes(lv3_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv3_shared[v0, v1, v2] = lv3[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1143[v_j // T.int64(8), v_i], lv36[v_j // T.int64(32), v_i], lv37[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1143[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * lv36[v_j // T.int64(32), v_i] + lv37[v_j // T.int64(32), v_i]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv3_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv3_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv2710[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv2710[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode4_matmul5_fp16_before(lv11: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv12: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv13: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv11[v_i // T.int64(8), v_j], lv12[v_i // T.int64(32), v_j], lv13[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv11[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv12[v_i // T.int64(32), v_j] + lv13[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2712[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2712[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode4_matmul5_fp16_after(lv1128: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv12: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv13: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv2712_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2712_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2712[v0, v1, v2])
                                T.writes(lv2712_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2712_shared[v0, v1, v2] = lv2712[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1128[v_j // T.int64(8), v_i], lv12[v_j // T.int64(32), v_i], lv13[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1128[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * lv12[v_j // T.int64(32), v_i] + lv13[v_j // T.int64(32), v_i]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2712_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2712_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_fp16_before(lv51: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv52: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv53: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv51[v_i // T.int64(8), v_j], lv52[v_i // T.int64(32), v_j], lv53[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv51[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv52[v_i // T.int64(32), v_j] + lv53[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2749[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2749[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv5[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv5[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_fp16_after(lv1153: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv52: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv53: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(11008)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2749_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2749[v0, v1, v2])
                                T.writes(lv2749_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2749_shared[v0, v1, v2] = lv2749[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1153[v_j // T.int64(8), v_i], lv52[v_j // T.int64(32), v_i], lv53[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1153[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * lv52[v_j // T.int64(32), v_i] + lv53[v_j // T.int64(32), v_i]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv5[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv5[v0, v1, v2] * var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_fp16_before(lv43: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv44: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv45: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv43[v_i // T.int64(8), v_j], lv44[v_i // T.int64(32), v_j], lv45[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv43[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv44[v_i // T.int64(32), v_j] + lv45[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv2749[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv2749[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_fp16_after(lv1148: T.Buffer((T.int64(512), T.int64(11008)), "uint32"), lv44: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv45: T.Buffer((T.int64(128), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(11008)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(4)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                            with T.block("lv2749_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(4096), ax1_ax2_fused_0 * T.int64(1024) + ax1_ax2_fused_1 * T.int64(4) + ax1_ax2_fused_2)
                                T.reads(lv2749[v0, v1, v2])
                                T.writes(lv2749_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv2749_shared[v0, v1, v2] = lv2749[v0, v1, v2]
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(64)):
                    for ax0_0 in range(T.int64(8)):
                        for ax0_1 in T.unroll(T.int64(8)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(4096), k_0_0 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                    v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1148[v_j // T.int64(8), v_i], lv44[v_j // T.int64(32), v_i], lv45[v_j // T.int64(32), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1148[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * lv44[v_j // T.int64(32), v_i] + lv45[v_j // T.int64(32), v_i]
                    for k_0_1_k_1_fused in range(T.int64(64)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4096), k_0_0 * T.int64(64) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2] * T.sigmoid(var_matmul_intermediate_local[v0, v1, v2])


@T.prim_func
def fused_decode6_fused_matmul9_add3_fp16_before(lv59: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv60: T.Buffer((T.int64(344), T.int64(4096)), "float16"), lv61: T.Buffer((T.int64(344), T.int64(4096)), "float16"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv59[v_i // T.int64(8), v_j], lv60[v_i // T.int64(32), v_j], lv61[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = T.Cast("float16", T.bitwise_and(T.shift_right(lv59[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * lv60[v_i // T.int64(32), v_j] + lv61[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv5[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv5[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv3[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv3[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_fp16_after(lv1158: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv60: T.Buffer((T.int64(344), T.int64(4096)), "float16"), lv61: T.Buffer((T.int64(344), T.int64(4096)), "float16"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        # with T.block("root"):
        var_decode_intermediate_local = T.alloc_buffer((T.int64(11008), T.int64(4096)), scope="local", dtype="float16")
        var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
        lv6_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="shared", dtype="float16")
        for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                        T.reads()
                        T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                        var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0_0 in range(T.int64(2)):
                        for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(22)):
                            for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                                with T.block("lv6_shared"):
                                    v0 = T.axis.spatial(T.int64(1), ax0)
                                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2 = T.axis.spatial(T.int64(11008), k_0_0 * T.int64(5504) + (ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1))
                                    T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(5504))
                                    T.reads(lv6[v0, v1, v2])
                                    T.writes(lv6_shared[v0, v1, v2])
                                    T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                    lv6_shared[v0, v1, v2] = lv6[v0, v1, v2]
                        for k_0_1 in range(T.int64(86)):
                            for ax0_0 in range(T.int64(8)):
                                for ax0_1 in T.unroll(T.int64(8)):
                                    for ax1 in range(T.int64(1)):
                                        with T.block("decode"):
                                            v_j = T.axis.spatial(T.int64(11008), k_0_0 * T.int64(5504) + k_0_1 * T.int64(64) + ax0_0 * T.int64(8) + ax0_1)
                                            v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                            T.reads(lv1158[v_j // T.int64(8), v_i], lv60[v_j // T.int64(32), v_i], lv61[v_j // T.int64(32), v_i])
                                            T.writes(var_decode_intermediate_local[v_j, v_i])
                                            var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1158[v_j // T.int64(8), v_i], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * lv60[v_j // T.int64(32), v_i] + lv61[v_j // T.int64(32), v_i]
                            for k_0_2_k_1_fused in range(T.int64(64)):
                                with T.block("matmul_update"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                                    v_k = T.axis.reduce(T.int64(11008), k_0_0 * T.int64(5504) + k_0_1 * T.int64(64) + k_0_2_k_1_fused)
                                    T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv6_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv6_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("var_matmul_intermediate_local"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                            T.reads(lv4[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                            T.writes(p_output0_intermediate[v0, v1, v2])
                            p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode3_matmul1_cast_int3_fp16_before(lv2931: T.Buffer((T.int64(412), T.int64(32000)), "uint32"), lv2932: T.Buffer((T.int64(103), T.int64(32000)), "float16"), lv3025: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2931[v_i // T.int64(10), v_j], lv2932[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv2931[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv2932[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv3025[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv3025[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_decode3_matmul1_cast_int3_fp16_after(lv1123: T.Buffer((T.int64(412), T.int64(32000)), "uint32"), lv5866: T.Buffer((T.int64(103), T.int64(32000)), "float16"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_pad_local = T.alloc_buffer((T.int64(4120), T.int64(32000)), scope="local", dtype="float16")
    var_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), scope="local", dtype="float16")
    lv1511_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(125), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv1511_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv1511[v0, v1, v2])
                            T.writes(lv1511_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv1511_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv1511[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("var_decode_intermediate_pad"):
                                v0 = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v1 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1123[v0 // T.int64(10), v1], lv5866[v0 // T.int64(40), v1])
                                T.writes(var_decode_intermediate_pad_local[v0, v1])
                                var_decode_intermediate_pad_local[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1123[v0 // T.int64(10), v1], T.Cast("uint32", v0 % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv1511_shared[v_i0, v_i1, v_k], var_decode_intermediate_pad_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv1511_shared[v_i0, v_i1, v_k] * var_decode_intermediate_pad_local[v_k, v_i2] * lv5866[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_pad_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_pad_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = T.Cast("float32", var_matmul_intermediate_pad_local[v0, v1, v2])


@T.prim_func
def fused_decode4_fused_matmul5_add3_int3_fp16_before(lv1605: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), lv1606: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv164: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv1518: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1605[v_i // T.int64(10), v_j], lv1606[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1605[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1606[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv164[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv164[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv1518[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv1518[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode4_fused_matmul5_add3_int3_fp16_after(lv1143: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), lv36: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv3_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv3_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv3[v0, v1, v2])
                            T.writes(lv3_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv3_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv3[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1143[v_j // T.int64(10), v_i], lv36[v_j // T.int64(40), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1143[v_j // T.int64(10), v_i], T.Cast("uint32", v_j % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv3_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv3_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * lv36[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv2710[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv2710[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode4_matmul5_int3_fp16_before(lv1587: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), lv1588: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv1520: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1587[v_i // T.int64(10), v_j], lv1588[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1587[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1588[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1520[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1520[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode4_matmul5_int3_fp16_after(lv1128: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), lv12: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv2712_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2712_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2712[v0, v1, v2])
                            T.writes(lv2712_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2712_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2712[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1128[v_j // T.int64(10), v_i], lv12[v_j // T.int64(40), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1128[v_j // T.int64(10), v_i], T.Cast("uint32", v_j % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2712_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2712_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * lv12[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_int3_fp16_before(lv1617: T.Buffer((T.int64(412), T.int64(11008)), "uint32"), lv1618: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv1557: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1617[v_i // T.int64(10), v_j], lv1618[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1617[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1618[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1557[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1557[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv3[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv3[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_int3_fp16_after(lv1153: T.Buffer((T.int64(412), T.int64(11008)), "uint32"), lv52: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(11008)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2749[v0, v1, v2])
                            T.writes(lv2749_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2749_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2749[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1153[v_j // T.int64(10), v_i], lv52[v_j // T.int64(40), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1153[v_j // T.int64(10), v_i], T.Cast("uint32", v_j % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * lv52[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv5[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv5[v0, v1, v2] * var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_int3_fp16_before(lv1611: T.Buffer((T.int64(412), T.int64(11008)), "uint32"), lv1612: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv1557: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1611[v_i // T.int64(10), v_j], lv1612[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1611[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1612[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1557[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1557[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_int3_fp16_after(lv1148: T.Buffer((T.int64(412), T.int64(11008)), "uint32"), lv44: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(11008)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2749[v0, v1, v2])
                            T.writes(lv2749_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2749_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2749[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1148[v_j // T.int64(10), v_i], lv44[v_j // T.int64(40), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1148[v_j // T.int64(10), v_i], T.Cast("uint32", v_j % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * lv44[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2] * T.sigmoid(var_matmul_intermediate_local[v0, v1, v2])


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_fp16_before(lv1623: T.Buffer((T.int64(1104), T.int64(4096)), "uint32"), lv1624: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv167: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv165: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1623[v_i // T.int64(10), v_j], lv1624[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1623[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1624[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv167[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv167[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv165[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv165[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_fp16_after(lv1158: T.Buffer((T.int64(1104), T.int64(4096)), "uint32"), lv60: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(11040), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv6_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11040)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(2)):
                    for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(22)):
                        for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                            with T.block("lv6_shared"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v2 = T.axis.spatial(T.int64(11040), k_0_0 * T.int64(5520) + (ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1))
                                T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(5520))
                                T.reads(lv6[v0, v1, v2])
                                T.writes(lv6_shared[v0, v1, v2])
                                T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                                lv6_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(11008), lv6[v0, v1, v2], T.float16(0))
                    for k_0_1 in range(T.int64(69)):
                        for ax0_0 in T.unroll(T.int64(80)):
                            for ax1 in range(T.int64(1)):
                                with T.block("decode"):
                                    v_j = T.axis.spatial(T.int64(11040), k_0_0 * T.int64(5520) + k_0_1 * T.int64(80) + ax0_0)
                                    v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                    T.reads(lv1158[v_j // T.int64(10), v_i], lv60[v_j // T.int64(40), v_i])
                                    T.writes(var_decode_intermediate_local[v_j, v_i])
                                    var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.bitwise_and(T.shift_right(lv1158[v_j // T.int64(10), v_i], T.Cast("uint32", v_j % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)
                        for k_0_2_k_1_fused in range(T.int64(80)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                                v_k = T.axis.reduce(T.int64(11040), k_0_0 * T.int64(5520) + k_0_1 * T.int64(80) + k_0_2_k_1_fused)
                                T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv6_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2])
                                T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv6_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * lv60[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv4[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode3_matmul1_cast_int3_int16_fp16_before(lv2931: T.Buffer((T.int64(824), T.int64(32000)), "uint16"), lv2932: T.Buffer((T.int64(103), T.int64(32000)), "float16"), lv3025: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(32000)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv2931[v_i // T.int64(5), v_j], lv2932[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv2931[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv2932[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv3025[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv3025[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_decode3_matmul1_cast_int3_int16_fp16_after(lv1123: T.Buffer((T.int64(824), T.int64(32000)), "uint16"), lv5866: T.Buffer((T.int64(103), T.int64(32000)), "float16"), lv1511: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_pad_local = T.alloc_buffer((T.int64(4120), T.int64(32000)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(103), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), scope="local", dtype="float16")
    lv1511_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(125), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv1511_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv1511[v0, v1, v2])
                            T.writes(lv1511_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv1511_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv1511[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("var_decode_intermediate_pad"):
                                v0 = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v1 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1123[v0 // T.int64(5), v1])
                                T.writes(var_decode_intermediate_pad_local[v0, v1])
                                var_decode_intermediate_pad_local[v0, v1] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1123[v0 // T.int64(5), v1]), T.Cast("uint16", v0 % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in range(T.int64(1)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(103), k_0_0 + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv5866[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv5866[v_j, v_i]
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2], lv1511_shared[v_i0, v_i1, v_k], var_decode_intermediate_pad_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + lv1511_shared[v_i0, v_i1, v_k] * var_decode_intermediate_pad_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_pad_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(32000), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_pad_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = T.Cast("float32", var_matmul_intermediate_pad_local[v0, v1, v2])


@T.prim_func
def fused_decode4_fused_matmul5_add3_int3_int16_fp16_before(lv1605: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), lv1606: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv164: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv1518: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1605[v_i // T.int64(5), v_j], lv1606[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1605[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1606[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv164[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv164[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv1518[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv1518[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode4_fused_matmul5_add3_int3_int16_fp16_after(lv1143: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), lv36: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv2710: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(4096)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(103), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv3_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv3_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv3[v0, v1, v2])
                            T.writes(lv3_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv3_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv3[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1143[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1143[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in range(T.int64(1)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(103), k_0_0 + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv36[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv36[v_j, v_i]
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv3_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv3_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv2710[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv2710[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode4_matmul5_int3_int16_fp16_before(lv1587: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), lv1588: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv1520: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1587[v_i // T.int64(5), v_j], lv1588[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1587[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1588[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1520[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1520[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode4_matmul5_int3_int16_fp16_after(lv1128: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), lv12: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv2712: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(4096)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(103), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv2712_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2712_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2712[v0, v1, v2])
                            T.writes(lv2712_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2712_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2712[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1128[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1128[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in range(T.int64(1)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(103), k_0_0 + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv12[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv12[v_j, v_i]
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2712_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2712_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(var_matmul_intermediate[v0, v1, v2])
                        var_matmul_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_int3_int16_fp16_before(lv1617: T.Buffer((T.int64(824), T.int64(11008)), "uint16"), lv1618: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv1557: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv3: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1617[v_i // T.int64(5), v_j], lv1618[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1617[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1618[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1557[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1557[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv3[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv3[v_ax0, v_ax1, v_ax2] * var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_multiply1_int3_int16_fp16_after(lv1153: T.Buffer((T.int64(824), T.int64(11008)), "uint16"), lv52: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(11008)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(103), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2749[v0, v1, v2])
                            T.writes(lv2749_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2749_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2749[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1153[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1153[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in range(T.int64(1)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(103), k_0_0 + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv52[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv52[v_j, v_i]
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv5[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv5[v0, v1, v2] * var_matmul_intermediate_local[v0, v1, v2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_int3_int16_fp16_before(lv1611: T.Buffer((T.int64(824), T.int64(11008)), "uint16"), lv1612: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv1557: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1611[v_i // T.int64(5), v_j], lv1612[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1611[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1612[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1557[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1557[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_matmul_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode5_fused_matmul8_silu1_int3_int16_fp16_after(lv1148: T.Buffer((T.int64(824), T.int64(11008)), "uint16"), lv44: T.Buffer((T.int64(103), T.int64(11008)), "float16"), lv2749: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(4120), T.int64(11008)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(103), T.int64(11008)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), scope="local", dtype="float16")
    lv2749_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4120)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(43), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(17)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(4120), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv2749[v0, v1, v2])
                            T.writes(lv2749_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(4120))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv2749_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(4096), lv2749[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(103)):
                    for ax0_0 in T.unroll(T.int64(40)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(4120), k_0_0 * T.int64(40) + ax0_0)
                                v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1148[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1148[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in range(T.int64(1)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(103), k_0_0 + ax0_0)
                                v_i = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv44[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv44[v_j, v_i]
                    for k_0_1_k_1_fused in range(T.int64(40)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(4120), k_0_0 * T.int64(40) + k_0_1_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv2749_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv2749_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(11008), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = var_matmul_intermediate_local[v0, v1, v2] * T.sigmoid(var_matmul_intermediate_local[v0, v1, v2])


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_int16_fp16_before(lv1623: T.Buffer((T.int64(2208), T.int64(4096)), "uint16"), lv1624: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv167: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv165: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1623[v_i // T.int64(5), v_j], lv1624[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1623[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1624[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv167[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv167[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv165[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv165[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_int16_fp16_after(lv1158: T.Buffer((T.int64(2208), T.int64(4096)), "uint16"), lv60: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(11040), T.int64(4096)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(276), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv6_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11040)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(44)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(11040), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(11040))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv6_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(11008), lv6[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(138)):
                    for ax0_0 in T.unroll(T.int64(80)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(11040), k_0_0 * T.int64(80) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1158[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1158[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in T.unroll(T.int64(2)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(276), k_0_0 * T.int64(2) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv60[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv60[v_j, v_i]
                    for k_0_2_k_1_fused in range(T.int64(80)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(11040), k_0_0 * T.int64(80) + k_0_2_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv6_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv6_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv4[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]
################################################

def get_dict_key(func):
    return tvm.ir.structural_hash(func), func


tir_dispatch_dict = {
    get_dict_key(fused_min_max_triu_te_broadcast_to): fused_min_max_triu_te_broadcast_to_sch_func(),
    get_dict_key(rms_norm_before): rms_norm_after,
    get_dict_key(rms_norm_fp16_before): rms_norm_fp16_after,
    get_dict_key(softmax_before): softmax_after,
    get_dict_key(softmax_mxn_before): softmax_mxn_after,
    get_dict_key(softmax_cast_mxn_before): softmax_cast_mxn_after,
    get_dict_key(softmax_fp16_before): softmax_fp16_after,
    get_dict_key(softmax_mxn_fp16_before): softmax_mxn_fp16_after,
    get_dict_key(softmax_1xn_before): softmax_1xn_sch_func(softmax_1xn_before),
    get_dict_key(softmax_cast_1xn_before): softmax_1xn_sch_func(softmax_cast_1xn_before, cast_to_fp16=True),
    get_dict_key(softmax_1xn_fp16_before): softmax_1xn_sch_func(softmax_1xn_fp16_before),
    get_dict_key(matmul1_before): matmul1_after,
    get_dict_key(matmul2_before): matmul2_sch_func(),
    get_dict_key(matmul5_before): matmul5_after,
    get_dict_key(matmul5_with_m_before): matmul5_with_m_after,
    get_dict_key(NT_matmul_before): NT_matmul_after,
    get_dict_key(NT_matmul4_before): NT_matmul4_sch_func(),
    get_dict_key(NT_matmul9_before): NT_matmul9_sch_func(),
    get_dict_key(fused_matmul1_add1): fused_matmul1_add1_sch_func(),
    get_dict_key(fused_matmul3_multiply): fused_matmul3_multiply_sch_func(),
    get_dict_key(fused_matmul3_silu): fused_matmul3_silu_sch_func(),
    get_dict_key(fused_matmul4_add1): fused_matmul4_add1_sch_func(),
    get_dict_key(fused_NT_matmul_add1_before): fused_NT_matmul_add1_after,
    get_dict_key(fused_NT_matmul1_divide_add_maximum_before): fused_NT_matmul1_divide_add_maximum_after,
    get_dict_key(fused_NT_matmul1_divide_add_maximum_with_m_before): fused_NT_matmul1_divide_add_maximum_with_m_after,
    get_dict_key(fused_NT_matmul6_divide1_add2_maximum1_before): fused_NT_matmul6_divide1_add2_maximum1_after,
    get_dict_key(fused_NT_matmul2_multiply_before): fused_NT_matmul2_multiply_after,
    get_dict_key(fused_NT_matmul2_silu_before): fused_NT_matmul2_silu_after,
    get_dict_key(fused_NT_matmul3_add1_before): fused_NT_matmul3_add1_after,
    get_dict_key(fused_NT_matmul_divide_maximum_minimum_cast_before): fused_NT_matmul_divide_maximum_minimum_cast_sch_func(),
    get_dict_key(fused_NT_matmul1_add3_before): fused_NT_matmul1_add3_sch_func(),
    get_dict_key(fused_NT_matmul2_divide1_add2_maximum1_before): fused_NT_matmul2_divide1_add2_maximum1_sch_func(fused_NT_matmul2_divide1_add2_maximum1_before),
    get_dict_key(fused_NT_matmul2_divide1_maximum1_minimum1_cast3_before): fused_NT_matmul2_divide1_maximum1_minimum1_cast3_after,
    get_dict_key(fused_NT_matmul3_multiply1_before): fused_NT_matmul3_multiply1_sch_func(),
    get_dict_key(fused_NT_matmul3_silu1_before): fused_NT_matmul3_silu1_sch_func(),
    get_dict_key(fused_NT_matmul4_add3_before): fused_NT_matmul4_add3_sch_func(),
    get_dict_key(matmul1_fp16_before): matmul1_fp16_sch_func(),
    get_dict_key(matmul8_fp16_before): matmul8_fp16_sch_func(matmul8_fp16_before),
    get_dict_key(matmul8_with_m_fp16_before): matmul8_fp16_sch_func(matmul8_with_m_fp16_before),
    get_dict_key(NT_matmul1_fp16_before): NT_matmul1_fp16_sch_func(),
    get_dict_key(decode6): decode_sch_func(decode6),
    get_dict_key(decode7): decode_sch_func(decode7),
    get_dict_key(decode8): decode_sch_func(decode8),
    get_dict_key(decode4_fp16): decode_sch_func(decode4_fp16),
    get_dict_key(decode5_fp16): decode_sch_func(decode5_fp16),
    get_dict_key(decode6_fp16): decode_sch_func(decode6_fp16),
    get_dict_key(decode_int3_fp16): decode_sch_func(decode_int3_fp16),
    get_dict_key(decode1_int3_fp16): decode_sch_func(decode1_int3_fp16),
    get_dict_key(decode2_int3_fp16): decode_sch_func(decode2_int3_fp16),
    get_dict_key(decode_int3_int16_fp16): decode_sch_func(decode_int3_int16_fp16),
    get_dict_key(decode1_int3_int16_fp16): decode_sch_func(decode1_int3_int16_fp16),
    get_dict_key(decode2_int3_int16_fp16): decode_sch_func(decode2_int3_int16_fp16),
    get_dict_key(fused_decode3_matmul1_before): fused_decode3_matmul1_after,
    get_dict_key(fused_decode4_fused_matmul5_add3_before): fused_decode4_fused_matmul5_add3_after,
    get_dict_key(fused_decode4_matmul5_before): fused_decode4_matmul5_after,
    get_dict_key(fused_decode5_fused_matmul8_multiply1_before): fused_decode5_fused_matmul8_multiply1_after,
    get_dict_key(fused_decode5_fused_matmul8_silu1_before): fused_decode5_fused_matmul8_silu1_after,
    get_dict_key(fused_decode6_fused_matmul9_add3_before): fused_decode6_fused_matmul9_add3_after,
    get_dict_key(fused_decode3_matmul1_fp16_before): fused_decode3_matmul1_fp16_after,
    get_dict_key(fused_decode3_matmul1_cast_fp16_before): fused_decode3_matmul1_cast_fp16_after,
    get_dict_key(fused_decode4_fused_matmul5_add3_fp16_before): fused_decode4_fused_matmul5_add3_fp16_after,
    get_dict_key(fused_decode4_matmul5_fp16_before): fused_decode4_matmul5_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_multiply1_fp16_before): fused_decode5_fused_matmul8_multiply1_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_silu1_fp16_before): fused_decode5_fused_matmul8_silu1_fp16_after,
    get_dict_key(fused_decode6_fused_matmul9_add3_fp16_before): fused_decode6_fused_matmul9_add3_fp16_after,
    get_dict_key(fused_decode3_matmul1_cast_int3_fp16_before): fused_decode3_matmul1_cast_int3_fp16_after,
    get_dict_key(fused_decode4_fused_matmul5_add3_int3_fp16_before): fused_decode4_fused_matmul5_add3_int3_fp16_after,
    get_dict_key(fused_decode4_matmul5_int3_fp16_before): fused_decode4_matmul5_int3_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_multiply1_int3_fp16_before): fused_decode5_fused_matmul8_multiply1_int3_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_silu1_int3_fp16_before): fused_decode5_fused_matmul8_silu1_int3_fp16_after,
    get_dict_key(fused_decode6_fused_matmul9_add3_int3_fp16_before): fused_decode6_fused_matmul9_add3_int3_fp16_after,
    get_dict_key(fused_decode3_matmul1_cast_int3_int16_fp16_before): fused_decode3_matmul1_cast_int3_int16_fp16_after,
    get_dict_key(fused_decode4_fused_matmul5_add3_int3_int16_fp16_before): fused_decode4_fused_matmul5_add3_int3_int16_fp16_after,
    get_dict_key(fused_decode4_matmul5_int3_int16_fp16_before): fused_decode4_matmul5_int3_int16_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_multiply1_int3_int16_fp16_before): fused_decode5_fused_matmul8_multiply1_int3_int16_fp16_after,
    get_dict_key(fused_decode5_fused_matmul8_silu1_int3_int16_fp16_before): fused_decode5_fused_matmul8_silu1_int3_int16_fp16_after,
    get_dict_key(fused_decode6_fused_matmul9_add3_int3_int16_fp16_before): fused_decode6_fused_matmul9_add3_int3_int16_fp16_after,
}
# fmt: on


def lookup_func(func):
    for (hash_value, func_before), f_after in tir_dispatch_dict.items():
        if tvm.ir.structural_hash(func) == hash_value and tvm.ir.structural_equal(
            func, func_before
        ):
            return f_after
    return None


@tvm.transform.module_pass(opt_level=0, name="DispatchTIROperator")
class DispatchTIROperator:
    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        for gv in mod.functions:
            scheduled_func = lookup_func(mod[gv])
            if scheduled_func is not None:
                mod[gv] = scheduled_func

        return mod
