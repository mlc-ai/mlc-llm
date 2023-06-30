# pylint: disable=pointless-string-statement,invalid-name,missing-docstring,line-too-long,too-many-locals,too-many-arguments,too-many-statements
from tvm.script import ir as I
from tvm.script import tir as T

# fmt: off

@I.ir_module
class Module:
    @T.prim_func
    def cast1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = A[v_i0, v_i1, v_i2]

    @T.prim_func
    def decode(A: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), B: T.Buffer((T.int64(80), T.int64(2560)), "uint32"), T_transpose: T.Buffer((T.int64(2560), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode_1 = T.alloc_buffer((T.int64(2560), T.int64(2560)))
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode_1[v_i, v_j])
                decode_1[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(B[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(B[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for ax0, ax1 in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode_1[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode_1[v_ax1, v_ax0]

    @T.prim_func
    def decode1(A: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), B: T.Buffer((T.int64(80), T.int64(10240)), "uint32"), T_transpose: T.Buffer((T.int64(10240), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode = T.alloc_buffer((T.int64(2560), T.int64(10240)))
        for i, j in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(B[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(B[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for ax0, ax1 in T.grid(T.int64(10240), T.int64(2560)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @T.prim_func
    def decode2(A: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), B: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), T_transpose: T.Buffer((T.int64(2560), T.int64(10240)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode = T.alloc_buffer((T.int64(10240), T.int64(2560)))
        for i, j in T.grid(T.int64(10240), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(B[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(B[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for ax0, ax1 in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @T.prim_func
    def divide1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[()]

    @T.prim_func
    def fused_decode3_matmul1(lv1352: T.Buffer((T.int64(320), T.int64(50432)), "uint32"), lv1353: T.Buffer((T.int64(80), T.int64(50432)), "uint32"), lv1800: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(50432)))
        for i, j in T.grid(T.int64(2560), T.int64(50432)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1352[v_i // T.int64(8), v_j], lv1353[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1352[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1353[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1353[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50432), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1800[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1800[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]

    @T.prim_func
    def fused_decode4_fused_matmul7_add3(lv1363: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv1364: T.Buffer((T.int64(80), T.int64(2560)), "uint32"), lv1808: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), linear_bias192: T.Buffer((T.int64(2560),), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1363[v_i // T.int64(8), v_j], lv1364[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1363[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1364[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1364[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1808[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1808[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias192[v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias192[v_ax2]

    @T.prim_func
    def fused_decode4_fused_matmul7_add3_add4(lv1381: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv1382: T.Buffer((T.int64(80), T.int64(2560)), "uint32"), lv5: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), linear_bias195: T.Buffer((T.int64(2560),), "float32"), lv1805: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1381[v_i // T.int64(8), v_j], lv1382[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1381[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1382[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1382[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv5[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv5[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias195[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias195[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], lv1805[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] + lv1805[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_decode5_fused_matmul9_add5_gelu1(lv1387: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), lv1388: T.Buffer((T.int64(80), T.int64(10240)), "uint32"), lv1852: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), linear_bias196: T.Buffer((T.int64(10240),), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(10240)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(10240)))
        for i, j in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1387[v_i // T.int64(8), v_j], lv1388[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1387[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1388[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1388[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(10240), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1852[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1852[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
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
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_decode6_fused_matmul10_add3_cast1_add4(lv1393: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv1394: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv1857: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float32"), linear_bias197: T.Buffer((T.int64(2560),), "float32"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(10240), T.int64(2560)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, j in T.grid(T.int64(10240), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1393[v_i // T.int64(8), v_j], lv1394[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(lv1393[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(lv1394[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(lv1394[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(10240)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1857[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1857[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
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
                var_compute_intermediate[v_i0, v_i1, v_i2] = var_T_add_intermediate[v_i0, v_i1, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate[v_ax0, v_ax1, v_ax2], lv6[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate[v_ax0, v_ax1, v_ax2] + lv6[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_reshape7_squeeze1(lv1821: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1821[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1821[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_slice1_cast1(lv3599: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_slice_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, _, k in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("slice"):
                v_i, v__, v_k = T.axis.remap("SSS", [i, _, k])
                T.reads(lv3599[v_i, T.int64(0), v_k])
                T.writes(var_slice_intermediate[v_i, v__, v_k])
                var_slice_intermediate[v_i, v__, v_k] = lv3599[v_i, T.int64(0), v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_slice_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = var_slice_intermediate[v_i0, v_i1, v_i2]

    @T.prim_func
    def fused_transpose7_reshape8(lv1844: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1844[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1844[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)]

    @T.prim_func
    def layer_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
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
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func
    def reshape5(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(A[T.int64(0), T.int64(0)])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]

    @T.prim_func
    def reshape6(A: T.Buffer((T.int64(1), T.int64(2560)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax2 % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(2560)]

    @T.prim_func
    def reshape7(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)]

    @T.prim_func
    def softmax1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(50432)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func
    def squeeze1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float32"), T_squeeze: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def take_decode1(A: T.Buffer((T.int64(50432), T.int64(320)), "uint32"), B: T.Buffer((T.int64(50432), T.int64(80)), "uint32"), C: T.Buffer((T.int64(1),), "int32"), take_decode: T.Buffer((T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(1), T.int64(2560)):
            with T.block("take_decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[C[v_i], v_j // T.int64(8)], C[v_i], B[C[v_i], v_j // T.int64(32)])
                T.writes(take_decode[v_i, v_j])
                take_decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(A[C[v_i], v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(B[C[v_i], v_j // T.int64(32)], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(B[C[v_i], v_j // T.int64(32)], T.uint32(16)), T.uint32(65535)), T.uint32(16)))

    @T.prim_func
    def transpose6(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

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
        T.func_attr({"tir.noalias": T.bool(True)})
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
        T.func_attr({"tir.noalias": T.bool(True)})
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
        T.func_attr({"tir.noalias": T.bool(True)})
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

# fmt: on
