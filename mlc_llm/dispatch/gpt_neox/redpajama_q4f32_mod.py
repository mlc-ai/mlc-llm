# pylint: disable=pointless-string-statement,invalid-name,missing-docstring,line-too-long,too-many-locals,too-many-arguments,too-many-statements
from tvm.script import ir as I
from tvm.script import tir as T

# fmt: off

@I.ir_module
class Module:
    @T.prim_func
    def extend_te(var_A: T.handle, var_concat_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(1), n, n))
        m = T.int64()
        concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), n, m))
        # with T.block("root"):
        for b, _, i, j in T.grid(T.int64(1), T.int64(1), n, m):
            with T.block("concat_te"):
                v_b, v__, v_i, v_j = T.axis.remap("SSSS", [b, _, i, j])
                T.reads(A[v_b, v__, v_i, v_j + n - m])
                T.writes(concat_te[v_b, v__, v_i, v_j])
                concat_te[v_b, v__, v_i, v_j] = T.if_then_else(v_j < m - n, T.float32(3.4028234663852886e+38), A[v_b, v__, v_i, v_j + n - m])

    @T.prim_func
    def full(var_T_full: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(1), T.int64(1), n))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), n):
            with T.block("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(3.4028234663852886e+38)

    @T.prim_func
    def fused_NT_matmul1_divide_maximum_minimum(p_lv34: T.handle, p_lv35: T.handle, p_lv5: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv34 = T.match_buffer(p_lv34, (T.int64(1), T.int64(32), n, T.int64(80)))
        m = T.int64()
        lv35 = T.match_buffer(p_lv35, (T.int64(1), T.int64(32), m, T.int64(80)))
        lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m))
        var_T_minimum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv34[v_i0, v_i1, v_i2, v_k], lv35[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv34[v_i0, v_i1, v_i2, v_k] * lv35[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.11180339723346898)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])

    @T.prim_func
    def fused_NT_matmul2_add2_gelu(p_lv51: T.handle, lv38: T.Buffer((T.int64(10240), T.int64(2560)), "float32"), linear_bias4: T.Buffer((T.int64(10240),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv51 = T.match_buffer(p_lv51, (T.int64(1), n, T.int64(2560)))
        var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_multiply = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        compute = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_add = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(10240), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv51[v_i0, v_i1, v_k], lv38[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv51[v_i0, v_i1, v_k] * lv38[v_i2, v_k]
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

    @T.prim_func
    def fused_NT_matmul3_add_cast_add1(p_lv56: T.handle, lv45: T.Buffer((T.int64(2560), T.int64(10240)), "float32"), linear_bias5: T.Buffer((T.int64(2560),), "float32"), p_lv49: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv56 = T.match_buffer(p_lv56, (T.int64(1), n, T.int64(10240)))
        lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv56[v_i0, v_i1, v_k], lv45[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv56[v_i0, v_i1, v_k] * lv45[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias5[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias5[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = var_T_add_intermediate_1[v_i0, v_i1, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv49[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv49[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_NT_matmul4_divide2_maximum1_minimum1(lv1835: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32"), p_lv1836: T.handle, p_lv1806: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv1836 = T.match_buffer(p_lv1836, (T.int64(1), T.int64(32), n, T.int64(80)))
        lv1806 = T.match_buffer(p_lv1806, (T.int64(1), T.int64(1), T.int64(1), n))
        var_T_minimum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
        var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv1835[v_i0, v_i1, v_i2, v_k], lv1836[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1835[v_i0, v_i1, v_i2, v_k] * lv1836[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.11180339723346898)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1806[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1806[v_ax0, T.int64(0), v_ax2, v_ax3])

    @T.prim_func
    def fused_NT_matmul_add(p_lv7: T.handle, lv10: T.Buffer((T.int64(2560), T.int64(2560)), "float32"), linear_bias: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv7 = T.match_buffer(p_lv7, (T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv7[v_i0, v_i1, v_k], lv10[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv7[v_i0, v_i1, v_k] * lv10[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias[v_ax2]

    @T.prim_func
    def fused_NT_matmul_add_add1(p_lv45: T.handle, lv31: T.Buffer((T.int64(2560), T.int64(2560)), "float32"), linear_bias3: T.Buffer((T.int64(2560),), "float32"), p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv45 = T.match_buffer(p_lv45, (T.int64(1), n, T.int64(2560)))
        lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv45[v_i0, v_i1, v_k], lv31[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv45[v_i0, v_i1, v_k] * lv31[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias3[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias3[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv2[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv2[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_min_max_triu_te_broadcast_to(p_output0: T.handle, n: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        var_T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n))
        # with T.block("root"):
        var_make_diag_mask_te_intermediate = T.alloc_buffer((n, n))
        for i, j in T.grid(n, n):
            with T.block("make_diag_mask_te"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads()
                T.writes(var_make_diag_mask_te_intermediate[v_i, v_j])
                var_make_diag_mask_te_intermediate[v_i, v_j] = T.Select(v_i < v_j, T.float32(-3.4028234663852886e+38), T.float32(3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_make_diag_mask_te_intermediate[v_ax2, v_ax3])
                T.writes(var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_make_diag_mask_te_intermediate[v_ax2, v_ax3]

    @T.prim_func
    def layer_norm(var_A: T.handle, B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), var_T_layer_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        T_layer_norm = T.match_buffer(var_T_layer_norm, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), n))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), n))
        for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func
    def matmul(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m))
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)))
        matmul_1 = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(80)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(80), m):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func
    def matmul8(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n))
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(80)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80), n):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
        # with T.block("root"):
        for ax0 in range(n):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(n, ax0)
                T.reads(A[T.int64(0), v_ax0 % n])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]

    @T.prim_func
    def reshape1(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (n, T.int64(2560)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560)]

    @T.prim_func
    def reshape2(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(80)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), ((v_ax2 * T.int64(80) + v_ax3) // T.int64(2560) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(80) + v_ax3) // T.int64(2560) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)]

    @T.prim_func
    def reshape3(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(var_A, (m, T.int64(32), T.int64(80)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(80)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[((v_ax3 // T.int64(80) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(80) + v_ax2) % T.int64(32), v_ax3 % T.int64(80)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax3 // T.int64(80) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(80) + v_ax2) % T.int64(32), v_ax3 % T.int64(80)]

    @T.prim_func
    def reshape4(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), (v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), (v_ax2 // T.int64(2560) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)]

    @T.prim_func
    def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(2048), T.int64(80)), "float32"), C: T.Buffer((T.int64(2048), T.int64(80)), "float32"), var_rotary: T.handle, m: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)))
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(80)))
        # with T.block("root"):
        for i_batch_size, i_seq_len, i_num_heads, i_head_dim in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("rotary"):
                v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim = T.axis.remap("SSSS", [i_batch_size, i_seq_len, i_num_heads, i_head_dim])
                T.reads(B[m + v_i_seq_len - n, v_i_head_dim], A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(40):v_i_head_dim - T.int64(40) + T.int64(81)], C[m + v_i_seq_len - n, v_i_head_dim])
                T.writes(rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])
                rotary[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] = T.Select(v_i_head_dim < T.int64(80), B[m + v_i_seq_len - n, v_i_head_dim] * A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim] + C[m + v_i_seq_len - n, v_i_head_dim] * T.Select(v_i_head_dim < T.int64(40), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim + T.int64(40)] * T.float32(-1), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim - T.int64(40)]), A[v_i_batch_size, v_i_seq_len, v_i_num_heads, v_i_head_dim])

    @T.prim_func
    def slice(var_A: T.handle, slice_1: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        for i, _, k in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("slice"):
                v_i, v__, v_k = T.axis.remap("SSS", [i, _, k])
                T.reads(A[v_i, n - T.int64(1), v_k])
                T.writes(slice_1[v_i, v__, v_k])
                slice_1[v_i, v__, v_k] = A[v_i, n - T.int64(1), v_k]

    @T.prim_func
    def softmax(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
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
    def softmax2(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), T.int64(1), n))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
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
    def squeeze(var_A: T.handle, var_T_squeeze: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)))
        T_squeeze = T.match_buffer(var_T_squeeze, (n, T.int64(32), T.int64(80)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(n, T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def take_decode(A: T.Buffer((T.int64(50432), T.int64(320)), "uint32"), B: T.Buffer((T.int64(50432), T.int64(80)), "uint32"), var_C: T.handle, var_take_decode: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        C = T.match_buffer(var_C, (n,), "int32")
        take_decode_1 = T.match_buffer(var_take_decode, (n, T.int64(2560)))
        # with T.block("root"):
        for i, j in T.grid(n, T.int64(2560)):
            with T.block("take_decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[C[v_i], v_j // T.int64(8)], C[v_i], B[C[v_i], v_j // T.int64(32)])
                T.writes(take_decode_1[v_i, v_j])
                take_decode_1[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(A[C[v_i], v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(B[C[v_i], v_j // T.int64(32)], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(B[C[v_i], v_j // T.int64(32)], T.uint32(16)), T.uint32(65535)), T.uint32(16)))

    @T.prim_func
    def transpose(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)))
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), n, T.int64(80)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func
    def transpose1(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, T.int64(80)))
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), n, T.int64(32), T.int64(80)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]
# fmt: on
