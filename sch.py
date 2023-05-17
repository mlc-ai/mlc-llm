import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ir as I
from tvm.script import tir as T

# fmt: off

@I.ir_module
class Module:
    ########## Dynamic shape ##########

    @T.prim_func
    def fused_NT_matmul1_divide_maximum_minimum(p_lv34: T.handle, p_lv35: T.handle, p_lv5: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        m = T.meta_var(T.int64(128))
        lv34 = T.match_buffer(p_lv34, (T.int64(1), T.int64(32), n, T.int64(80)))
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
        n = T.meta_var(T.int64(128))
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
        n = T.meta_var(T.int64(128))
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
        n = T.meta_var(T.int64(128))
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
        n = T.meta_var(T.int64(128))
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
        n = T.meta_var(T.int64(128))
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
    def layer_norm(var_A: T.handle, B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), var_T_layer_norm: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
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
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        m = T.meta_var(T.int64(32))
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
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(32))
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



def main():
    db = ms.database.JSONDatabase(
        path_workload="/Users/jshao/Projects/mlc-llm/log_db/RedPajama-INCITE-Chat-3B-v1/database_workload.json",
        path_tuning_record="/Users/jshao/Projects/mlc-llm/log_db/RedPajama-INCITE-Chat-3B-v1/database_tuning_record.json",
    )
    LIST = [
        "fused_NT_matmul1_divide_maximum_minimum",
        "fused_NT_matmul2_add2_gelu",
        "fused_NT_matmul3_add_cast_add1",
        "fused_NT_matmul4_divide2_maximum1_minimum1",
        "fused_NT_matmul_add",
        "fused_NT_matmul_add_add1",
        "layer_norm",
        "matmul",
        "matmul8",
        "softmax",
        "softmax2",
    ]
    for k, v in Module.functions.items():
        k = str(k.name_hint)
        if k not in LIST:
            continue
        print(f"=============== {k} ===============")
        mod = ms.TuneContext(mod=v).mod
        assert db.has_workload(mod)
        workload = db.commit_workload(mod)
        (record,) = db.get_top_k(workload, top_k=1)
        sch = tir.Schedule(mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)
        # workload.mod.show(black_format=False)
        sch.show(black_format=False)


if __name__ == "__main__":
    main()




=============== fused_NT_matmul_add ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv7: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), lv10: T.Buffer((T.int64(2560), T.int64(2560)), "float32"), linear_bias: T.Buffer((T.int64(2560),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="local")
        lv7_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="shared")
        lv10_shared = T.alloc_buffer((T.int64(2560), T.int64(2560)), scope="shared")
        for i0_0_i1_0_i2_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_fused in T.thread_binding(T.int64(20), thread="vthread.x"):
                for i0_2_i1_2_i2_2_fused in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i0_4_init, i1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(64) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(5) * T.int64(16) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(2) + i1_3_init * T.int64(2) + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(64) * T.int64(40) + i0_1_i1_1_i2_1_fused % T.int64(5) * T.int64(8) + i0_2_i1_2_i2_2_fused % T.int64(8) + i2_3_init + i2_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0 in range(T.int64(320)):
                        for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv7_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(64) * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) // T.int64(8))
                                        v2 = T.axis.spatial(T.int64(2560), k_0 * T.int64(8) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) % T.int64(8))
                                        T.reads(lv7[v0, v1, v2])
                                        T.writes(lv7_shared[v0, v1, v2])
                                        lv7_shared[v0, v1, v2] = lv7[v0, v1, v2]
                        for ax0_ax1_fused_0 in range(T.int64(5)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                with T.block("lv10_shared"):
                                    v0 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(64) * T.int64(40) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1) // T.int64(8))
                                    v1 = T.axis.spatial(T.int64(2560), k_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1) % T.int64(8))
                                    T.reads(lv10[v0, v1])
                                    T.writes(lv10_shared[v0, v1])
                                    lv10_shared[v0, v1] = lv10[v0, v1]
                        for k_1, i0_3, i1_3, i2_3, k_2, i0_4, i1_4, i2_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(2), T.int64(1)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(64) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(5) * T.int64(16) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(2) + i1_3 * T.int64(2) + i1_4)
                                v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(64) * T.int64(40) + i0_1_i1_1_i2_1_fused % T.int64(5) * T.int64(8) + i0_2_i1_2_i2_2_fused % T.int64(8) + i2_3 + i2_4)
                                v_k = T.axis.reduce(T.int64(2560), k_0 * T.int64(8) + k_1 * T.int64(4) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2], lv7_shared[v_i0, v_i1, v_k], lv10_shared[v_i2, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv7_shared[v_i0, v_i1, v_k] * lv10_shared[v_i2, v_k]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(1)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(64) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(5) * T.int64(16) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(2) + ax1)
                            v2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(64) * T.int64(40) + i0_1_i1_1_i2_1_fused % T.int64(5) * T.int64(8) + i0_2_i1_2_i2_2_fused % T.int64(8) + ax2)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2], linear_bias[v2])
                            T.writes(var_T_add_intermediate[v0, v1, v2])
                            var_T_add_intermediate[v0, v1, v2] = var_NT_matmul_intermediate_local[v0, v1, v2] + linear_bias[v2]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="NT_matmul", func_name="main")
  b1 = sch.get_block(name="T_add", func_name="main")
  b2 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l3, l4, l5, l6 = sch.get_loops(block=b0)
  v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l12, l13, l14, l15, l16 = sch.split(loop=l3, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
  v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[2, 4, 8, 1, 2])
  l22, l23, l24, l25, l26 = sch.split(loop=l4, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
  v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[64, 5, 8, 1, 1])
  l32, l33, l34, l35, l36 = sch.split(loop=l5, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
  v37, v38, v39 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[320, 2, 4])
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
  l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b47)
  l55 = sch.fuse(l52, l53, l54, preserve_unit_iters=True)
  v56 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch", ann_val=v56)
  b57 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b57, loop=l40, preserve_unit_loops=True, index=-1)
  l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b57)
  l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
  v65 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
  sch.reverse_compute_inline(block=b1)
  v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b47, ann_key="meta_schedule.cooperative_fetch")
  l67, l68, l69, l70, l71 = sch.get_loops(block=b47)
  l72, l73, l74 = sch.split(loop=l71, factors=[None, 64, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l74)
  sch.bind(loop=l73, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
  l75, l76, l77, l78, l79 = sch.get_loops(block=b57)
  l80, l81 = sch.split(loop=l79, factors=[None, 64], preserve_unit_iters=True)
  sch.bind(loop=l81, thread_axis="threadIdx.x")
  b82 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.unroll_explicit")
  b83, b84, b85, b86 = sch.get_child_blocks(b82)
  l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b83)
  l94, l95, l96, l97, l98, l99 = sch.get_loops(block=b84)
  l100, l101, l102, l103, l104, l105, l106, l107, l108, l109, l110, l111 = sch.get_loops(block=b85)
  sch.annotate(block_or_loop=l100, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l100, ann_key="pragma_unroll_explicit", ann_val=1)
  l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b86)
  b118 = sch.get_block(name="NT_matmul", func_name="main")
  l119, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129, l130 = sch.get_loops(block=b118)
  b131 = sch.decompose_reduction(block=b118, loop=l122)

=============== matmul ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(80)), "float32"), matmul_1: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), "float32")):
        T.func_attr({"global_symbol": "main", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_1_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), scope="local")
        A_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(32)), scope="shared")
        B_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(80)), scope="shared")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(T.int64(16), thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(16) * T.int64(4) + i0_1_i1_1_i2_1_i3_1_fused // T.int64(4) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(16) * T.int64(8) + i0_1_i1_1_i2_1_i3_1_fused % T.int64(4) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(80) + i2_3_init + i2_4_init)
                            v_i3 = T.axis.spatial(T.int64(80), i0_2_i1_2_i2_2_i3_2_fused % T.int64(80) + i3_3_init + i3_4_init)
                            T.reads()
                            T.writes(matmul_1_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            matmul_1_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0 in range(T.int64(8)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(16) * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(640) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(16) * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(640) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(32) // T.int64(4))
                                        v3 = T.axis.spatial(T.int64(32), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(640) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(4))
                                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(160) + ax0_ax1_ax2_ax3_fused_1) * T.int64(4) + ax0_ax1_ax2_ax3_fused_2 < T.int64(128))
                                        T.reads(A[v0, v1, v2, v3])
                                        T.writes(A_shared[v0, v1, v2, v3])
                                        A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(160), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(16) * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(320) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(320))
                                        v2 = T.axis.spatial(T.int64(32), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(320) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(80))
                                        v3 = T.axis.spatial(T.int64(80), (ax0_ax1_ax2_ax3_fused_0 * T.int64(320) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(80))
                                        T.reads(B[v0, v1, v2, v3])
                                        T.writes(B_shared[v0, v1, v2, v3])
                                        B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
                        for k_1, i0_3, i1_3, i2_3, i3_3, k_2, i0_4, i1_4, i2_4, i3_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(16) * T.int64(4) + i0_1_i1_1_i2_1_i3_1_fused // T.int64(4) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(16) * T.int64(8) + i0_1_i1_1_i2_1_i3_1_fused % T.int64(4) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(80) + i2_3 + i2_4)
                                v_i3 = T.axis.spatial(T.int64(80), i0_2_i1_2_i2_2_i3_2_fused % T.int64(80) + i3_3 + i3_4)
                                v_k = T.axis.reduce(T.int64(32), k_0 * T.int64(4) + k_1 + k_2)
                                T.reads(matmul_1_local[v_i0, v_i1, v_i2, v_i3], A_shared[v_i0, v_i1, v_i2, v_k], B_shared[v_i0, v_i1, v_k, v_i3])
                                T.writes(matmul_1_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                matmul_1_local[v_i0, v_i1, v_i2, v_i3] = matmul_1_local[v_i0, v_i1, v_i2, v_i3] + A_shared[v_i0, v_i1, v_i2, v_k] * B_shared[v_i0, v_i1, v_k, v_i3]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_1_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(16) * T.int64(4) + i0_1_i1_1_i2_1_i3_1_fused // T.int64(4) + ax1)
                            v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(16) * T.int64(8) + i0_1_i1_1_i2_1_i3_1_fused % T.int64(4) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(80) + ax2)
                            v3 = T.axis.spatial(T.int64(80), i0_2_i1_2_i2_2_i3_2_fused % T.int64(80) + ax3)
                            T.reads(matmul_1_local[v0, v1, v2, v3])
                            T.writes(matmul_1[v0, v1, v2, v3])
                            matmul_1[v0, v1, v2, v3] = matmul_1_local[v0, v1, v2, v3]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="matmul", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l2, l3, l4, l5, l6 = sch.get_loops(block=b0)
  v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
  v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[8, 4, 1, 1, 1])
  l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
  v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[16, 4, 2, 1, 1])
  l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
  v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 80, 1, 1])
  l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
  v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[8, 4, 1])
  l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
  sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
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
  l58, l59, l60, l61, l62, l63, l64, l65 = sch.get_loops(block=b57)
  l66 = sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
  v67 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
  b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
  l69, l70, l71, l72, l73, l74, l75, l76 = sch.get_loops(block=b68)
  l77 = sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
  v78 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
  v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
  l80, l81, l82, l83, l84 = sch.get_loops(block=b57)
  l85, l86, l87 = sch.split(loop=l84, factors=[None, 160, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l87)
  sch.bind(loop=l86, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
  l88, l89, l90, l91, l92 = sch.get_loops(block=b68)
  l93, l94, l95 = sch.split(loop=l92, factors=[None, 160, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l95)
  sch.bind(loop=l94, thread_axis="threadIdx.x")
  b96 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b96, ann_key="meta_schedule.unroll_explicit")
  b97, b98, b99, b100 = sch.get_child_blocks(b96)
  l101, l102, l103, l104, l105, l106, l107 = sch.get_loops(block=b97)
  l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(block=b98)
  l115, l116, l117, l118, l119, l120, l121, l122, l123, l124, l125, l126, l127, l128 = sch.get_loops(block=b99)
  sch.annotate(block_or_loop=l115, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l115, ann_key="pragma_unroll_explicit", ann_val=1)
  l129, l130, l131, l132, l133, l134, l135 = sch.get_loops(block=b100)
  b136 = sch.get_block(name="matmul", func_name="main")
  l137, l138, l139, l140, l141, l142, l143, l144, l145, l146, l147, l148, l149, l150 = sch.get_loops(block=b136)
  b151 = sch.decompose_reduction(block=b136, loop=l140)

=============== matmul8 ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(32)), "float32"), B: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(80)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32")):
        T.func_attr({"global_symbol": "main", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), scope="local")
        A_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(32)), scope="shared")
        B_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(80)), scope="shared")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(T.int64(80), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(40) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(1), i2_3_init + i2_4_init)
                            v_i3 = T.axis.spatial(T.int64(80), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(40) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(40) + i3_3_init + i3_4_init)
                            T.reads()
                            T.writes(matmul_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            matmul_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0 in range(T.int64(8)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(80), thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1) // T.int64(4))
                                    v2 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v3 = T.axis.spatial(T.int64(32), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1) % T.int64(4))
                                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1 < T.int64(8))
                                    T.reads(A[v0, v1, v2, v3])
                                    T.writes(A_shared[v0, v1, v2, v3])
                                    A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(80), thread="threadIdx.x"):
                                with T.block("B_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1) // T.int64(160))
                                    v2 = T.axis.spatial(T.int64(32), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1) % T.int64(160) // T.int64(40))
                                    v3 = T.axis.spatial(T.int64(80), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(40) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(80) + ax0_ax1_ax2_ax3_fused_1) % T.int64(40))
                                    T.reads(B[v0, v1, v2, v3])
                                    T.writes(B_shared[v0, v1, v2, v3])
                                    B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
                        for k_1, i0_3, i1_3, i2_3, i3_3, k_2, i0_4, i1_4, i2_4, i3_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(40) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(1), i2_3 + i2_4)
                                v_i3 = T.axis.spatial(T.int64(80), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(40) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(40) + i3_3 + i3_4)
                                v_k = T.axis.reduce(T.int64(32), k_0 * T.int64(4) + k_1 * T.int64(2) + k_2)
                                T.reads(matmul_local[v_i0, v_i1, v_i2, v_i3], A_shared[v_i0, v_i1, v_i2, v_k], B_shared[v_i0, v_i1, v_k, v_i3])
                                T.writes(matmul_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                matmul_local[v_i0, v_i1, v_i2, v_i3] = matmul_local[v_i0, v_i1, v_i2, v_i3] + A_shared[v_i0, v_i1, v_i2, v_k] * B_shared[v_i0, v_i1, v_k, v_i3]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(40) + ax1)
                            v2 = T.axis.spatial(T.int64(1), ax2)
                            v3 = T.axis.spatial(T.int64(80), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(40) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(40) + ax3)
                            T.reads(matmul_local[v0, v1, v2, v3])
                            T.writes(matmul[v0, v1, v2, v3])
                            matmul[v0, v1, v2, v3] = matmul_local[v0, v1, v2, v3]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="matmul", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l2, l3, l4, l5, l6 = sch.get_loops(block=b0)
  v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l12, l13, l14, l15, l16 = sch.split(loop=l2, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
  v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 2, 1, 1])
  l22, l23, l24, l25, l26 = sch.split(loop=l3, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
  v27, v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l32, l33, l34, l35, l36 = sch.split(loop=l4, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True)
  v37, v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[2, 1, 40, 1, 1])
  l42, l43, l44, l45, l46 = sch.split(loop=l5, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True)
  v47, v48, v49 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[8, 2, 2])
  l50, l51, l52 = sch.split(loop=l6, factors=[v47, v48, v49], preserve_unit_iters=True)
  sch.reorder(l12, l22, l32, l42, l13, l23, l33, l43, l14, l24, l34, l44, l50, l51, l15, l25, l35, l45, l52, l16, l26, l36, l46)
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
  l58, l59, l60, l61, l62, l63, l64, l65 = sch.get_loops(block=b57)
  l66 = sch.fuse(l62, l63, l64, l65, preserve_unit_iters=True)
  v67 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
  b68 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b68, loop=l50, preserve_unit_loops=True, index=-1)
  l69, l70, l71, l72, l73, l74, l75, l76 = sch.get_loops(block=b68)
  l77 = sch.fuse(l73, l74, l75, l76, preserve_unit_iters=True)
  v78 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
  v79 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v79)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b57, ann_key="meta_schedule.cooperative_fetch")
  l80, l81, l82, l83, l84 = sch.get_loops(block=b57)
  l85, l86 = sch.split(loop=l84, factors=[None, 80], preserve_unit_iters=True)
  sch.bind(loop=l86, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b68, ann_key="meta_schedule.cooperative_fetch")
  l87, l88, l89, l90, l91 = sch.get_loops(block=b68)
  l92, l93 = sch.split(loop=l91, factors=[None, 80], preserve_unit_iters=True)
  sch.bind(loop=l93, thread_axis="threadIdx.x")
  b94 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b94, ann_key="meta_schedule.unroll_explicit")
  b95, b96, b97, b98 = sch.get_child_blocks(b94)
  l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b95)
  l105, l106, l107, l108, l109, l110 = sch.get_loops(block=b96)
  l111, l112, l113, l114, l115, l116, l117, l118, l119, l120, l121, l122, l123, l124 = sch.get_loops(block=b97)
  l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b98)
  b132 = sch.get_block(name="matmul", func_name="main")
  l133, l134, l135, l136, l137, l138, l139, l140, l141, l142, l143, l144, l145, l146 = sch.get_loops(block=b132)
  b147 = sch.decompose_reduction(block=b132, loop=l136)

=============== fused_NT_matmul1_divide_maximum_minimum ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv34: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), "float32"), lv35: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), "float32"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(128), T.int64(128)), "float32"), var_T_minimum_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(128)), scope="local")
        lv34_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), scope="shared")
        lv35_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), scope="shared")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(T.int64(1024), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(4), T.int64(2)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(64) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(16) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(64) // T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(16) // T.int64(4) * T.int64(4) + i2_3_init * T.int64(4) + i2_4_init)
                            v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(4) * T.int64(4) + i3_3_init * T.int64(2) + i3_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0 in range(T.int64(10)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("lv34_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(64) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(128))
                                        v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(64) // T.int64(8) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(128) // T.int64(8))
                                        v3 = T.axis.spatial(T.int64(80), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(8))
                                        T.reads(lv34[v0, v1, v2, v3])
                                        T.writes(lv34_shared[v0, v1, v2, v3])
                                        lv34_shared[v0, v1, v2, v3] = lv34[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(8)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                with T.block("lv35_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(64) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(8) * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) % T.int64(128) // T.int64(8))
                                    v3 = T.axis.spatial(T.int64(80), k_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) % T.int64(8))
                                    T.reads(lv35[v0, v1, v2, v3])
                                    T.writes(lv35_shared[v0, v1, v2, v3])
                                    lv35_shared[v0, v1, v2, v3] = lv35[v0, v1, v2, v3]
                        for k_1, i0_3, i1_3, i2_3, i3_3, k_2, i0_4, i1_4, i2_4, i3_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(1), T.int64(1), T.int64(4), T.int64(2)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(64) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(16) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(64) // T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(16) // T.int64(4) * T.int64(4) + i2_3 * T.int64(4) + i2_4)
                                v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(4) * T.int64(4) + i3_3 * T.int64(2) + i3_4)
                                v_k = T.axis.reduce(T.int64(80), k_0 * T.int64(8) + k_1 * T.int64(2) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3], lv34_shared[v_i0, v_i1, v_i2, v_k], lv35_shared[v_i0, v_i1, v_i3, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] + lv34_shared[v_i0, v_i1, v_i2, v_k] * lv35_shared[v_i0, v_i1, v_i3, v_k]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(4)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(64) * T.int64(2) + i0_2_i1_2_i2_2_i3_2_fused // T.int64(16) + ax1)
                            v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(64) // T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(16) // T.int64(4) * T.int64(4) + ax2)
                            v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(8) * T.int64(16) + i0_2_i1_2_i2_2_i3_2_fused % T.int64(4) * T.int64(4) + ax3)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2, v3], lv5[v0, T.int64(0), v2, v3])
                            T.writes(var_T_minimum_intermediate[v0, v1, v2, v3])
                            var_T_minimum_intermediate[v0, v1, v2, v3] = T.min(T.max(var_NT_matmul_intermediate_local[v0, v1, v2, v3] * T.float32(0.11180339723346898), T.float32(-3.4028234663852886e+38)), lv5[v0, T.int64(0), v2, v3])

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="NT_matmul", func_name="main")
  b1 = sch.get_block(name="T_divide", func_name="main")
  b2 = sch.get_block(name="T_maximum", func_name="main")
  b3 = sch.get_block(name="T_minimum", func_name="main")
  b4 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l5, l6, l7, l8, l9 = sch.get_loops(block=b0)
  v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l15, l16, l17, l18, l19 = sch.split(loop=l5, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
  v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[16, 1, 2, 1, 1])
  l25, l26, l27, l28, l29 = sch.split(loop=l6, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
  v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[8, 1, 4, 1, 4])
  l35, l36, l37, l38, l39 = sch.split(loop=l7, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
  v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[8, 1, 4, 2, 2])
  l45, l46, l47, l48, l49 = sch.split(loop=l8, factors=[v40, v41, v42, v43, v44], preserve_unit_iters=True)
  v50, v51, v52 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[10, 4, 2])
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
  l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b60)
  l69 = sch.fuse(l65, l66, l67, l68, preserve_unit_iters=True)
  v70 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
  b71 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b71, loop=l53, preserve_unit_loops=True, index=-1)
  l72, l73, l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b71)
  l80 = sch.fuse(l76, l77, l78, l79, preserve_unit_iters=True)
  v81 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
  sch.reverse_compute_inline(block=b3)
  sch.reverse_compute_inline(block=b2)
  sch.reverse_compute_inline(block=b1)
  v82 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
  sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v82)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
  l83, l84, l85, l86, l87 = sch.get_loops(block=b60)
  l88, l89, l90 = sch.split(loop=l87, factors=[None, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l90)
  sch.bind(loop=l89, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch")
  l91, l92, l93, l94, l95 = sch.get_loops(block=b71)
  l96, l97 = sch.split(loop=l95, factors=[None, 32], preserve_unit_iters=True)
  sch.bind(loop=l97, thread_axis="threadIdx.x")
  b98 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b98, ann_key="meta_schedule.unroll_explicit")
  b99, b100, b101, b102 = sch.get_child_blocks(b98)
  l103, l104, l105, l106, l107, l108, l109 = sch.get_loops(block=b99)
  l110, l111, l112, l113, l114, l115 = sch.get_loops(block=b100)
  l116, l117, l118, l119, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129 = sch.get_loops(block=b101)
  sch.annotate(block_or_loop=l116, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l116, ann_key="pragma_unroll_explicit", ann_val=1)
  l130, l131, l132, l133, l134, l135, l136 = sch.get_loops(block=b102)
  b137 = sch.get_block(name="NT_matmul", func_name="main")
  l138, l139, l140, l141, l142, l143, l144, l145, l146, l147, l148, l149, l150, l151 = sch.get_loops(block=b137)
  b152 = sch.decompose_reduction(block=b137, loop=l141)

=============== fused_NT_matmul3_add_cast_add1 ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv56: T.Buffer((T.int64(1), T.int64(128), T.int64(10240)), "float32"), lv45: T.Buffer((T.int64(2560), T.int64(10240)), "float32"), linear_bias5: T.Buffer((T.int64(2560),), "float32"), lv49: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="local")
        lv56_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(10240)), scope="shared")
        lv45_shared = T.alloc_buffer((T.int64(2560), T.int64(10240)), scope="shared")
        for i0_0_i1_0_i2_0_fused in T.thread_binding(T.int64(40), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                for i0_2_i1_2_i2_2_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i0_4_init, i1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(1), T.int64(16), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused // T.int64(4) + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_2_i1_2_i2_2_fused % T.int64(4) * T.int64(16) + i2_3_init + i2_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0 in range(T.int64(640)):
                        for ax0_ax1_ax2_fused_0 in range(T.int64(8)):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv56_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) // T.int64(16))
                                        v2 = T.axis.spatial(T.int64(10240), k_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) % T.int64(16))
                                        T.reads(lv56[v0, v1, v2])
                                        T.writes(lv56_shared[v0, v1, v2])
                                        lv56_shared[v0, v1, v2] = lv56[v0, v1, v2]
                        for ax0_ax1_fused_0 in range(T.int64(4)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv45_shared"):
                                        v0 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(10240), k_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(2) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(lv45[v0, v1])
                                        T.writes(lv45_shared[v0, v1])
                                        lv45_shared[v0, v1] = lv45[v0, v1]
                        for k_1, i0_3, i1_3, i2_3, k_2, i0_4, i1_4, i2_4 in T.grid(T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(4), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused // T.int64(4) + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_2_i1_2_i2_2_fused % T.int64(4) * T.int64(16) + i2_3 + i2_4)
                                v_k = T.axis.reduce(T.int64(10240), k_0 * T.int64(16) + k_1 * T.int64(4) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2], lv56_shared[v_i0, v_i1, v_k], lv45_shared[v_i2, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv56_shared[v_i0, v_i1, v_k] * lv45_shared[v_i2, v_k]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(16)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(32) + i0_2_i1_2_i2_2_fused // T.int64(4) + ax1)
                            v2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused * T.int64(64) + i0_2_i1_2_i2_2_fused % T.int64(4) * T.int64(16) + ax2)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2], linear_bias5[v2], lv49[v0, v1, v2])
                            T.writes(var_T_add_intermediate[v0, v1, v2])
                            var_T_add_intermediate[v0, v1, v2] = var_NT_matmul_intermediate_local[v0, v1, v2] + linear_bias5[v2] + lv49[v0, v1, v2]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="NT_matmul", func_name="main")
  b1 = sch.get_block(name="T_add", func_name="main")
  b2 = sch.get_block(name="compute", func_name="main")
  b3 = sch.get_block(name="T_add_1", func_name="main")
  b4 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l5, l6, l7, l8 = sch.get_loops(block=b0)
  v9, v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l14, l15, l16, l17, l18 = sch.split(loop=l5, factors=[v9, v10, v11, v12, v13], preserve_unit_iters=True)
  v19, v20, v21, v22, v23 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[1, 4, 32, 1, 1])
  l24, l25, l26, l27, l28 = sch.split(loop=l6, factors=[v19, v20, v21, v22, v23], preserve_unit_iters=True)
  v29, v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[40, 1, 4, 16, 1])
  l34, l35, l36, l37, l38 = sch.split(loop=l7, factors=[v29, v30, v31, v32, v33], preserve_unit_iters=True)
  v39, v40, v41 = sch.sample_perfect_tile(loop=l8, n=3, max_innermost_factor=64, decision=[640, 4, 4])
  l42, l43, l44 = sch.split(loop=l8, factors=[v39, v40, v41], preserve_unit_iters=True)
  sch.reorder(l14, l24, l34, l15, l25, l35, l16, l26, l36, l42, l43, l17, l27, l37, l44, l18, l28, l38)
  l45 = sch.fuse(l14, l24, l34, preserve_unit_iters=True)
  sch.bind(loop=l45, thread_axis="blockIdx.x")
  l46 = sch.fuse(l15, l25, l35, preserve_unit_iters=True)
  sch.bind(loop=l46, thread_axis="vthread.x")
  l47 = sch.fuse(l16, l26, l36, preserve_unit_iters=True)
  sch.bind(loop=l47, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256)
  b48 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b48, loop=l47, preserve_unit_loops=True, index=-1)
  b49 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b49, loop=l42, preserve_unit_loops=True, index=-1)
  l50, l51, l52, l53, l54, l55, l56 = sch.get_loops(block=b49)
  l57 = sch.fuse(l54, l55, l56, preserve_unit_iters=True)
  v58 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b49, ann_key="meta_schedule.cooperative_fetch", ann_val=v58)
  b59 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b59, loop=l42, preserve_unit_loops=True, index=-1)
  l60, l61, l62, l63, l64, l65 = sch.get_loops(block=b59)
  l66 = sch.fuse(l64, l65, preserve_unit_iters=True)
  v67 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b59, ann_key="meta_schedule.cooperative_fetch", ann_val=v67)
  sch.reverse_compute_inline(block=b3)
  sch.reverse_compute_inline(block=b2)
  sch.reverse_compute_inline(block=b1)
  v68 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
  sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v68)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b49, ann_key="meta_schedule.cooperative_fetch")
  l69, l70, l71, l72, l73 = sch.get_loops(block=b49)
  l74, l75, l76 = sch.split(loop=l73, factors=[None, 128, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l76)
  sch.bind(loop=l75, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b59, ann_key="meta_schedule.cooperative_fetch")
  l77, l78, l79, l80, l81 = sch.get_loops(block=b59)
  l82, l83, l84 = sch.split(loop=l81, factors=[None, 128, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l84)
  sch.bind(loop=l83, thread_axis="threadIdx.x")
  b85 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.unroll_explicit")
  b86, b87, b88, b89 = sch.get_child_blocks(b85)
  l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b86)
  l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b87)
  l104, l105, l106, l107, l108, l109, l110, l111, l112, l113, l114, l115 = sch.get_loops(block=b88)
  sch.annotate(block_or_loop=l104, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l104, ann_key="pragma_unroll_explicit", ann_val=1)
  l116, l117, l118, l119, l120, l121 = sch.get_loops(block=b89)
  b122 = sch.get_block(name="NT_matmul", func_name="main")
  l123, l124, l125, l126, l127, l128, l129, l130, l131, l132, l133, l134 = sch.get_loops(block=b122)
  b135 = sch.decompose_reduction(block=b122, loop=l126)

=============== fused_NT_matmul2_add2_gelu ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv51: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), lv38: T.Buffer((T.int64(10240), T.int64(2560)), "float32"), linear_bias4: T.Buffer((T.int64(10240),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(10240)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(10240)), scope="local")
        lv51_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="shared")
        lv38_shared = T.alloc_buffer((T.int64(10240), T.int64(2560)), scope="shared")
        for i0_0_i1_0_i2_0_fused in T.thread_binding(T.int64(320), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_fused in T.thread_binding(T.int64(2), thread="vthread.x"):
                for i0_2_i1_2_i2_2_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i0_4_init, i1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(1), T.int64(2), T.int64(1)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(64) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_3_init * T.int64(2) + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(10240), i0_0_i1_0_i2_0_fused * T.int64(32) + i0_2_i1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3_init + i2_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0 in range(T.int64(160)):
                        for ax0_ax1_ax2_fused_0 in range(T.int64(8)):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv51_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) // T.int64(16))
                                        v2 = T.axis.spatial(T.int64(2560), k_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(2) + ax0_ax1_ax2_fused_2) % T.int64(16))
                                        T.reads(lv51[v0, v1, v2])
                                        T.writes(lv51_shared[v0, v1, v2])
                                        lv51_shared[v0, v1, v2] = lv51[v0, v1, v2]
                        for ax0_ax1_fused_0 in range(T.int64(4)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                with T.block("lv38_shared"):
                                    v0 = T.axis.spatial(T.int64(10240), i0_0_i1_0_i2_0_fused * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(16))
                                    v1 = T.axis.spatial(T.int64(2560), k_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(16))
                                    T.reads(lv38[v0, v1])
                                    T.writes(lv38_shared[v0, v1])
                                    lv38_shared[v0, v1] = lv38[v0, v1]
                        for k_1, i0_3, i1_3, i2_3, k_2, i0_4, i1_4, i2_4 in T.grid(T.int64(4), T.int64(1), T.int64(2), T.int64(4), T.int64(4), T.int64(1), T.int64(2), T.int64(1)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(64) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(4) + i1_3 * T.int64(2) + i1_4)
                                v_i2 = T.axis.spatial(T.int64(10240), i0_0_i1_0_i2_0_fused * T.int64(32) + i0_2_i1_2_i2_2_fused % T.int64(8) * T.int64(4) + i2_3 + i2_4)
                                v_k = T.axis.reduce(T.int64(2560), k_0 * T.int64(16) + k_1 * T.int64(4) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2], lv51_shared[v_i0, v_i1, v_k], lv38_shared[v_i2, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv51_shared[v_i0, v_i1, v_k] * lv38_shared[v_i2, v_k]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(128), i0_1_i1_1_i2_1_fused * T.int64(64) + i0_2_i1_2_i2_2_fused // T.int64(8) * T.int64(4) + ax1)
                            v2 = T.axis.spatial(T.int64(10240), i0_0_i1_0_i2_0_fused * T.int64(32) + i0_2_i1_2_i2_2_fused % T.int64(8) * T.int64(4) + ax2)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2], linear_bias4[v2])
                            T.writes(var_T_multiply_intermediate[v0, v1, v2])
                            var_T_multiply_intermediate[v0, v1, v2] = (var_NT_matmul_intermediate_local[v0, v1, v2] + linear_bias4[v2]) * (T.float32(0.5) + T.erf((var_NT_matmul_intermediate_local[v0, v1, v2] + linear_bias4[v2]) * T.float32(0.70710678118654757)) * T.float32(0.5))

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
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
  v22, v23, v24, v25, v26 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64, decision=[1, 2, 16, 2, 2])
  l27, l28, l29, l30, l31 = sch.split(loop=l9, factors=[v22, v23, v24, v25, v26], preserve_unit_iters=True)
  v32, v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64, decision=[320, 1, 8, 4, 1])
  l37, l38, l39, l40, l41 = sch.split(loop=l10, factors=[v32, v33, v34, v35, v36], preserve_unit_iters=True)
  v42, v43, v44 = sch.sample_perfect_tile(loop=l11, n=3, max_innermost_factor=64, decision=[160, 4, 4])
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
  v61 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v61)
  b62 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b62, loop=l45, preserve_unit_loops=True, index=-1)
  l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b62)
  l69 = sch.fuse(l67, l68, preserve_unit_iters=True)
  v70 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b62, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
  sch.compute_inline(block=b5)
  sch.compute_inline(block=b4)
  sch.compute_inline(block=b3)
  sch.compute_inline(block=b2)
  sch.compute_inline(block=b1)
  sch.reverse_compute_inline(block=b6)
  v71 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
  sch.annotate(block_or_loop=b7, ann_key="meta_schedule.unroll_explicit", ann_val=v71)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch")
  l72, l73, l74, l75, l76 = sch.get_loops(block=b52)
  l77, l78, l79 = sch.split(loop=l76, factors=[None, 128, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l79)
  sch.bind(loop=l78, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b62, ann_key="meta_schedule.cooperative_fetch")
  l80, l81, l82, l83, l84 = sch.get_loops(block=b62)
  l85, l86 = sch.split(loop=l84, factors=[None, 128], preserve_unit_iters=True)
  sch.bind(loop=l86, thread_axis="threadIdx.x")
  b87 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b87, ann_key="meta_schedule.unroll_explicit")
  b88, b89, b90, b91 = sch.get_child_blocks(b87)
  l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b88)
  l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b89)
  l105, l106, l107, l108, l109, l110, l111, l112, l113, l114, l115, l116 = sch.get_loops(block=b90)
  sch.annotate(block_or_loop=l105, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l105, ann_key="pragma_unroll_explicit", ann_val=1)
  l117, l118, l119, l120, l121, l122 = sch.get_loops(block=b91)
  b123 = sch.get_block(name="NT_matmul", func_name="main")
  l124, l125, l126, l127, l128, l129, l130, l131, l132, l133, l134, l135 = sch.get_loops(block=b123)
  b136 = sch.decompose_reduction(block=b123, loop=l127)

=============== layer_norm ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32")):
        T.func_attr({"global_symbol": "main", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(128)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(128)))
        for ax0_ax1_fused in T.thread_binding(T.int64(128), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for k2_0 in range(T.int64(40)):
                for k2_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                    with T.block("A_red_temp"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_fused)
                        v_k2 = T.axis.reduce(T.int64(2560), k2_0 * T.int64(64) + k2_1)
                        T.reads(A[v_ax0, v_ax1, v_k2])
                        T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                        with T.init():
                            A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                            A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                        v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                        v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                        A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                        A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0_ax1_ax2_fused_0 in range(T.int64(5)):
                    with T.block("T_layer_norm"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(65536) + ax0_ax1_ax2_fused_1 * T.int64(256) + ax0_ax1_ax2_fused_2) // T.int64(2560))
                        v_ax2 = T.axis.spatial(T.int64(2560), (ax0_ax1_ax2_fused_0 * T.int64(65536) + ax0_ax1_ax2_fused_1 * T.int64(256) + ax0_ax1_ax2_fused_2) % T.int64(2560))
                        T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                        T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                        T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="A_red_temp", func_name="main")
  b1 = sch.get_block(name="T_layer_norm", func_name="main")
  b2 = sch.get_block(name="root", func_name="main")
  v3 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], decision=4)
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

=============== fused_NT_matmul_add_add1 ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv45: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), lv31: T.Buffer((T.int64(2560), T.int64(2560)), "float32"), linear_bias3: T.Buffer((T.int64(2560),), "float32"), lv2: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(2560)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="local")
        lv45_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(2560)), scope="shared")
        lv31_shared = T.alloc_buffer((T.int64(2560), T.int64(2560)), scope="shared")
        for i0_0_i1_0_i2_0_fused in T.thread_binding(T.int64(160), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for i0_1_i1_1_i2_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                for i0_2_i1_2_i2_2_fused in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i0_4_init, i1_4_init, i2_4_init in T.grid(T.int64(1), T.int64(1), T.int64(16), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(80) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(2) * T.int64(32) + i0_2_i1_2_i2_2_fused + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(80) * T.int64(32) + i0_1_i1_1_i2_1_fused % T.int64(2) * T.int64(16) + i2_3_init + i2_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                    for k_0 in range(T.int64(320)):
                        for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("lv45_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(80) * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) // T.int64(8))
                                        v2 = T.axis.spatial(T.int64(2560), k_0 * T.int64(8) + (ax0_ax1_ax2_fused_0 * T.int64(128) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2) % T.int64(8))
                                        T.reads(lv45[v0, v1, v2])
                                        T.writes(lv45_shared[v0, v1, v2])
                                        lv45_shared[v0, v1, v2] = lv45[v0, v1, v2]
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("lv31_shared"):
                                        v0 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(80) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(8))
                                        v1 = T.axis.spatial(T.int64(2560), k_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(8))
                                        T.reads(lv31[v0, v1])
                                        T.writes(lv31_shared[v0, v1])
                                        lv31_shared[v0, v1] = lv31[v0, v1]
                        for k_1, i0_3, i1_3, i2_3, k_2, i0_4, i1_4, i2_4 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(16), T.int64(8), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(80) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(2) * T.int64(32) + i0_2_i1_2_i2_2_fused + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(80) * T.int64(32) + i0_1_i1_1_i2_1_fused % T.int64(2) * T.int64(16) + i2_3 + i2_4)
                                v_k = T.axis.reduce(T.int64(2560), k_0 * T.int64(8) + k_1 * T.int64(8) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2], lv45_shared[v_i0, v_i1, v_k], lv31_shared[v_i2, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv45_shared[v_i0, v_i1, v_k] * lv31_shared[v_i2, v_k]
                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(16)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_fused // T.int64(80) * T.int64(64) + i0_1_i1_1_i2_1_fused // T.int64(2) * T.int64(32) + i0_2_i1_2_i2_2_fused + ax1)
                            v2 = T.axis.spatial(T.int64(2560), i0_0_i1_0_i2_0_fused % T.int64(80) * T.int64(32) + i0_1_i1_1_i2_1_fused % T.int64(2) * T.int64(16) + ax2)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2], linear_bias3[v2], lv2[v0, v1, v2])
                            T.writes(var_T_add_intermediate[v0, v1, v2])
                            var_T_add_intermediate[v0, v1, v2] = var_NT_matmul_intermediate_local[v0, v1, v2] + linear_bias3[v2] + lv2[v0, v1, v2]

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="NT_matmul", func_name="main")
  b1 = sch.get_block(name="T_add", func_name="main")
  b2 = sch.get_block(name="T_add_1", func_name="main")
  b3 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l4, l5, l6, l7 = sch.get_loops(block=b0)
  v8, v9, v10, v11, v12 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l13, l14, l15, l16, l17 = sch.split(loop=l4, factors=[v8, v9, v10, v11, v12], preserve_unit_iters=True)
  v18, v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[2, 2, 32, 1, 1])
  l23, l24, l25, l26, l27 = sch.split(loop=l5, factors=[v18, v19, v20, v21, v22], preserve_unit_iters=True)
  v28, v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[80, 2, 1, 16, 1])
  l33, l34, l35, l36, l37 = sch.split(loop=l6, factors=[v28, v29, v30, v31, v32], preserve_unit_iters=True)
  v38, v39, v40 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64, decision=[320, 1, 8])
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
  l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b48)
  l56 = sch.fuse(l53, l54, l55, preserve_unit_iters=True)
  v57 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch", ann_val=v57)
  b58 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b58, loop=l41, preserve_unit_loops=True, index=-1)
  l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b58)
  l65 = sch.fuse(l63, l64, preserve_unit_iters=True)
  v66 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v66)
  sch.reverse_compute_inline(block=b2)
  sch.reverse_compute_inline(block=b1)
  v67 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v67)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.cooperative_fetch")
  l68, l69, l70, l71, l72 = sch.get_loops(block=b48)
  l73, l74, l75 = sch.split(loop=l72, factors=[None, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l75)
  sch.bind(loop=l74, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
  l76, l77, l78, l79, l80 = sch.get_loops(block=b58)
  l81, l82, l83 = sch.split(loop=l80, factors=[None, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l83)
  sch.bind(loop=l82, thread_axis="threadIdx.x")
  b84 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b84, ann_key="meta_schedule.unroll_explicit")
  b85, b86, b87, b88 = sch.get_child_blocks(b84)
  l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)
  l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b86)
  l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(block=b87)
  sch.annotate(block_or_loop=l103, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l103, ann_key="pragma_unroll_explicit", ann_val=1)
  l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b88)
  b121 = sch.get_block(name="NT_matmul", func_name="main")
  l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132, l133 = sch.get_loops(block=b121)
  b134 = sch.decompose_reduction(block=b121, loop=l125)

=============== fused_NT_matmul4_divide2_maximum1_minimum1 ===============
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv1835: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32"), lv1836: T.Buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), "float32"), lv1806: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(128)), "float32"), var_T_minimum_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), scope="local")
        lv1835_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), scope="shared")
        lv1836_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(128), T.int64(80)), scope="shared")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(T.int64(2), thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2)):
                        with T.block("NT_matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_1_i1_1_i2_1_i3_1_fused + i1_3_init + i1_4_init)
                            v_i2 = T.axis.spatial(T.int64(1), i2_3_init + i2_4_init)
                            v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_2_i1_2_i2_2_i3_2_fused * T.int64(2) + i3_3_init * T.int64(2) + i3_4_init)
                            T.reads()
                            T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                            var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    for k_0 in range(T.int64(20)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("lv1835_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(4))
                                        v2 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3 = T.axis.spatial(T.int64(80), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(4))
                                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) * T.int64(2) + ax0_ax1_ax2_ax3_fused_2 < T.int64(8))
                                        T.reads(lv1835[v0, v1, v2, v3])
                                        T.writes(lv1835_shared[v0, v1, v2, v3])
                                        lv1835_shared[v0, v1, v2, v3] = lv1835[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(16)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                with T.block("lv1836_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) // T.int64(256))
                                    v2 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(64) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) % T.int64(256) // T.int64(4))
                                    v3 = T.axis.spatial(T.int64(80), k_0 * T.int64(4) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(32) + ax0_ax1_ax2_ax3_fused_1) % T.int64(4))
                                    T.reads(lv1836[v0, v1, v2, v3])
                                    T.writes(lv1836_shared[v0, v1, v2, v3])
                                    lv1836_shared[v0, v1, v2, v3] = lv1836[v0, v1, v2, v3]
                        for k_1, i0_3, i1_3, i2_3, i3_3, k_2, i0_4, i1_4, i2_4, i3_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(2)):
                            with T.block("NT_matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_1_i1_1_i2_1_i3_1_fused + i1_3 + i1_4)
                                v_i2 = T.axis.spatial(T.int64(1), i2_3 + i2_4)
                                v_i3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_2_i1_2_i2_2_i3_2_fused * T.int64(2) + i3_3 * T.int64(2) + i3_4)
                                v_k = T.axis.reduce(T.int64(80), k_0 * T.int64(4) + k_1 * T.int64(2) + k_2)
                                T.reads(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3], lv1835_shared[v_i0, v_i1, v_i2, v_k], lv1836_shared[v_i0, v_i1, v_i3, v_k])
                                T.writes(var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(256), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "meta_schedule.tiling_structure": "SSSRRSRS"})
                                var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate_local[v_i0, v_i1, v_i2, v_i3] + lv1835_shared[v_i0, v_i1, v_i2, v_k] * lv1836_shared[v_i0, v_i1, v_i3, v_k]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(2)):
                        with T.block("var_NT_matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(32), i0_0_i1_0_i2_0_i3_0_fused // T.int64(2) * T.int64(2) + i0_1_i1_1_i2_1_i3_1_fused + ax1)
                            v2 = T.axis.spatial(T.int64(1), ax2)
                            v3 = T.axis.spatial(T.int64(128), i0_0_i1_0_i2_0_i3_0_fused % T.int64(2) * T.int64(64) + i0_2_i1_2_i2_2_i3_2_fused * T.int64(2) + ax3)
                            T.reads(var_NT_matmul_intermediate_local[v0, v1, v2, v3], lv1806[v0, T.int64(0), v2, v3])
                            T.writes(var_T_minimum_intermediate[v0, v1, v2, v3])
                            var_T_minimum_intermediate[v0, v1, v2, v3] = T.min(T.max(var_NT_matmul_intermediate_local[v0, v1, v2, v3] * T.float32(0.11180339723346898), T.float32(-3.4028234663852886e+38)), lv1806[v0, T.int64(0), v2, v3])

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="NT_matmul", func_name="main")
  b1 = sch.get_block(name="T_divide", func_name="main")
  b2 = sch.get_block(name="T_maximum", func_name="main")
  b3 = sch.get_block(name="T_minimum", func_name="main")
  b4 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l5, l6, l7, l8, l9 = sch.get_loops(block=b0)
  v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l15, l16, l17, l18, l19 = sch.split(loop=l5, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
  v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[16, 2, 1, 1, 1])
  l25, l26, l27, l28, l29 = sch.split(loop=l6, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
  v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l35, l36, l37, l38, l39 = sch.split(loop=l7, factors=[v30, v31, v32, v33, v34], preserve_unit_iters=True)
  v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64, decision=[2, 1, 32, 1, 2])
  l45, l46, l47, l48, l49 = sch.split(loop=l8, factors=[v40, v41, v42, v43, v44], preserve_unit_iters=True)
  v50, v51, v52 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64, decision=[20, 2, 2])
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
  l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b60)
  l69 = sch.fuse(l65, l66, l67, l68, preserve_unit_iters=True)
  v70 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch", ann_val=v70)
  b71 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b0])
  sch.compute_at(block=b71, loop=l53, preserve_unit_loops=True, index=-1)
  l72, l73, l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b71)
  l80 = sch.fuse(l76, l77, l78, l79, preserve_unit_iters=True)
  v81 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
  sch.reverse_compute_inline(block=b3)
  sch.reverse_compute_inline(block=b2)
  sch.reverse_compute_inline(block=b1)
  v82 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
  sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v82)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b60, ann_key="meta_schedule.cooperative_fetch")
  l83, l84, l85, l86, l87 = sch.get_loops(block=b60)
  l88, l89, l90 = sch.split(loop=l87, factors=[None, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l90)
  sch.bind(loop=l89, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch")
  l91, l92, l93, l94, l95 = sch.get_loops(block=b71)
  l96, l97 = sch.split(loop=l95, factors=[None, 32], preserve_unit_iters=True)
  sch.bind(loop=l97, thread_axis="threadIdx.x")
  b98 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b98, ann_key="meta_schedule.unroll_explicit")
  b99, b100, b101, b102 = sch.get_child_blocks(b98)
  l103, l104, l105, l106, l107, l108, l109 = sch.get_loops(block=b99)
  l110, l111, l112, l113, l114, l115 = sch.get_loops(block=b100)
  l116, l117, l118, l119, l120, l121, l122, l123, l124, l125, l126, l127, l128, l129 = sch.get_loops(block=b101)
  l130, l131, l132, l133, l134, l135, l136 = sch.get_loops(block=b102)
  b137 = sch.get_block(name="NT_matmul", func_name="main")
  l138, l139, l140, l141, l142, l143, l144, l145, l146, l147, l148, l149, l150, l151 = sch.get_loops(block=b137)
  b152 = sch.decompose_reduction(block=b137, loop=l141)