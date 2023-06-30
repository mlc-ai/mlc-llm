from tvm.script import ir as I
from tvm.script import tir as T

"""
 ID |                                                   Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done
-----------------------------------------------------------------------------------------------------------------------------------------------------------
  0 |                                                   cast |          1 |      1 |         0.0000 |      27.7422 |               27.7422 |      4 |    Y
  1 |                                                  cast6 |          1 |      1 |         0.0000 |      27.6800 |               27.6800 |      4 |    Y
  2 |                                                decode4 |   26214400 |      1 |       167.6862 |     156.3301 |              156.3301 |    172 |    Y
  3 |                                                decode5 |  104857600 |      1 |       128.5783 |     815.5153 |              815.5153 |    172 |    Y
  4 |                                                decode6 |  104857600 |      1 |       128.6586 |     815.0066 |              815.0066 |    179 |    Y
  5 |                                                divide2 |      50432 |      1 |         1.8169 |      27.7575 |               27.7575 |      4 |    Y
  6 |                                  fused_NT_matmul1_add4 | 1678049280 |      1 |      2178.8097 |     770.1679 |              770.1679 |   1088 |    Y
  7 |                             fused_NT_matmul1_add4_add5 | 1678376960 |      1 |      2130.5374 |     787.7717 |              787.7717 |   1215 |    Y
  8 |       fused_NT_matmul2_divide1_maximum1_minimum1_cast9 |   85458944 |      1 |      1211.9454 |      70.5139 |               70.5139 |    192 |    Y
  9 |                     fused_NT_matmul3_add6_gelu1_cast11 | 6717440000 |      1 |      2129.3171 |    3154.7391 |             3154.7391 |   4416 |    Y
 10 |                fused_NT_matmul4_add7_cast8_cast12_add5 | 6711541760 |      1 |      2072.7296 |    3238.0208 |             3238.0208 |   4544 |    Y
 11 |          fused_NT_matmul4_add7_cast8_cast12_add5_cast7 | 6711541760 |      1 |      2091.5892 |    3208.8241 |             3208.8241 |   4416 |    Y
 12 |           fused_NT_matmul_divide_maximum_minimum_cast2 |     667648 |      1 |        23.3021 |      28.6519 |               28.6519 |     64 |    Y
 13 |            fused_decode1_fused_matmul4_add2_gelu_cast4 |  157337600 |      1 |       812.5380 |     193.6372 |              193.6372 |    319 |    Y
 14 |      fused_decode2_fused_matmul5_add3_cast1_cast5_add1 |  157291520 |      1 |       730.8166 |     215.2271 |              215.2271 |    320 |    Y
 15 | fused_decode2_fused_matmul5_add3_cast1_cast5_add1_cast |  157291520 |      1 |       729.0229 |     215.7566 |              215.7566 |    319 |    Y
 16 |                                  fused_decode3_matmul6 |  774635520 |      1 |       868.1608 |     892.2719 |              892.2719 |   1331 |    Y
 17 |                         fused_decode_fused_matmul2_add |   39324160 |      1 |       733.2646 |      53.6289 |               53.6289 |    191 |    Y
 18 |                    fused_decode_fused_matmul2_add_add1 |   39326720 |      1 |       740.8926 |      53.0802 |               53.0802 |    192 |    Y
 19 |                                fused_layer_norm1_cast8 |    4587520 |      1 |        76.3188 |      60.1099 |               60.1099 |     50 |    Y
 20 |                                 fused_layer_norm_cast1 |      35840 |      1 |         0.6533 |      54.8634 |               54.8634 |    159 |    Y
 21 |                                 fused_reshape2_squeeze |          1 |      1 |         0.0000 |      27.5470 |               27.5470 |      4 |    Y
 22 |                                     fused_slice1_cast6 |          1 |      1 |         0.0000 |      27.5899 |               27.5899 |      4 |    Y
 23 |                              fused_transpose4_reshape4 |          1 |      1 |         0.0000 |      27.5157 |               27.5157 |      4 |    Y
 24 |                                             layer_norm |      35840 |      1 |         0.6506 |      55.0910 |               55.0910 |    160 |    Y
 25 |                                            layer_norm1 |    4587520 |      1 |        74.6941 |      61.4174 |               61.4174 |     50 |    Y
 26 |                                                matmul3 |     163840 |      1 |         5.8011 |      28.2428 |               28.2428 |     64 |    Y
 27 |                                                matmul9 |   20971520 |      1 |       571.2811 |      36.7096 |               36.7096 |    192 |    Y
 28 |                                                reshape |          1 |      1 |         0.0000 |      27.9399 |               27.9399 |      1 |    Y
 29 |                                               reshape1 |          1 |      1 |         0.0000 |      27.6659 |               27.6659 |      4 |    Y
 30 |                                               reshape2 |          1 |      1 |         0.0000 |      27.6446 |               27.6446 |      4 |    Y
 31 |                                               softmax2 |     201728 |      1 |         2.8631 |      70.4578 |               70.4578 |    186 |    Y
 32 |                                                squeeze |          1 |      1 |         0.0000 |      27.3156 |               27.3156 |      4 |    Y
 33 |                                            take_decode |      10240 |      1 |         0.3712 |      27.5835 |               27.5835 |      4 |    Y
 34 |                                             transpose2 |          1 |      1 |         0.0000 |      27.6975 |               27.6975 |      4 |    Y
-----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# fmt: off

@I.ir_module
class Module:
    @T.prim_func
    def cast(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", A[v_i0, v_i1, v_i2])

    @T.prim_func
    def cast6(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = A[v_i0, v_i1, v_i2]

    @T.prim_func
    def decode4(A: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), B: T.Buffer((T.int64(80), T.int64(2560)), "float16"), T_transpose: T.Buffer((T.int64(2560), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i // T.int64(32), v_j]
        for ax0, ax1 in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @T.prim_func
    def decode5(A: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), B: T.Buffer((T.int64(80), T.int64(10240)), "float16"), T_transpose: T.Buffer((T.int64(10240), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode = T.alloc_buffer((T.int64(2560), T.int64(10240)), "float16")
        for i, j in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i // T.int64(32), v_j]
        for ax0, ax1 in T.grid(T.int64(10240), T.int64(2560)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @T.prim_func
    def decode6(A: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), B: T.Buffer((T.int64(320), T.int64(2560)), "float16"), T_transpose: T.Buffer((T.int64(2560), T.int64(10240)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        decode = T.alloc_buffer((T.int64(10240), T.int64(2560)), "float16")
        for i, j in T.grid(T.int64(10240), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i // T.int64(32), v_j]
        for ax0, ax1 in T.grid(T.int64(2560), T.int64(10240)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(decode[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @T.prim_func
    def divide2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(50432)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[()]

    @T.prim_func
    def fused_decode1_fused_matmul4_add2_gelu_cast4(lv32: T.Buffer((T.int64(320), T.int64(10240)), "uint32"), lv33: T.Buffer((T.int64(80), T.int64(10240)), "float16"), lv2115: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias196: T.Buffer((T.int64(10240),), "float32"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16")):
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
                T.reads(lv32[v_i // T.int64(8), v_j], lv33[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv32[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv33[v_i // T.int64(32), v_j]
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

    @T.prim_func
    def fused_decode2_fused_matmul5_add3_cast1_cast5_add1(lv38: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv39: T.Buffer((T.int64(320), T.int64(2560)), "float16"), lv2121: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), linear_bias197: T.Buffer((T.int64(2560),), "float32"), lv8: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
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
                T.reads(lv38[v_i // T.int64(8), v_j], lv39[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv38[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv39[v_i // T.int64(32), v_j]
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
                T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv8[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv8[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_decode2_fused_matmul5_add3_cast1_cast5_add1_cast(lv1154: T.Buffer((T.int64(1280), T.int64(2560)), "uint32"), lv1155: T.Buffer((T.int64(320), T.int64(2560)), "float16"), lv4105: T.Buffer((T.int64(1), T.int64(1), T.int64(10240)), "float16"), linear_bias383: T.Buffer((T.int64(2560),), "float32"), lv380: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
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
                T.reads(lv1154[v_i // T.int64(8), v_j], lv1155[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1154[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1155[v_i // T.int64(32), v_j]
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
                T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv380[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv380[v_ax0, v_ax1, v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2])

    @T.prim_func
    def fused_decode3_matmul6(lv1160: T.Buffer((T.int64(320), T.int64(50432)), "uint32"), lv1161: T.Buffer((T.int64(80), T.int64(50432)), "float32"), lv384: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(50432)))
        for i, j in T.grid(T.int64(2560), T.int64(50432)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1160[v_i // T.int64(8), v_j], lv1161[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.Cast("float16", T.bitwise_and(T.shift_right(lv1160[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1161[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(50432), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv384[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv384[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]

    @T.prim_func
    def fused_decode_fused_matmul2_add(lv8: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv9: T.Buffer((T.int64(80), T.int64(2560)), "float16"), lv2067: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias192: T.Buffer((T.int64(2560),), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv8[v_i // T.int64(8), v_j], lv9[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv8[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv9[v_i // T.int64(32), v_j]
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

    @T.prim_func
    def fused_decode_fused_matmul2_add_add1(lv26: T.Buffer((T.int64(320), T.int64(2560)), "uint32"), lv27: T.Buffer((T.int64(80), T.int64(2560)), "float16"), lv7: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), linear_bias195: T.Buffer((T.int64(2560),), "float16"), lv2062: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_decode_intermediate = T.alloc_buffer((T.int64(2560), T.int64(2560)), "float16")
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")
        for i, j in T.grid(T.int64(2560), T.int64(2560)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv26[v_i // T.int64(8), v_j], lv27[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv26[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv27[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(2560), T.int64(2560)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv7[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv7[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
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

    @T.prim_func
    def fused_layer_norm_cast1(lv2064: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), weight67: T.Buffer((T.int64(2560),), "float32"), bias65: T.Buffer((T.int64(2560),), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1)))
        var_T_layer_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(lv2064[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + lv2064[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + lv2064[v_ax0, v_ax1, v_k2] * lv2064[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv2064[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], weight67[v_ax2], bias65[v_ax2])
                T.writes(var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv2064[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight67[v_ax2] + bias65[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def fused_reshape2_squeeze(lv2080: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), var_T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv2080[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2080[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_slice1_cast6(lv4113: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_slice_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2560)))
        for i, _, k in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("slice"):
                v_i, v__, v_k = T.axis.remap("SSS", [i, _, k])
                T.reads(lv4113[v_i, T.int64(0), v_k])
                T.writes(var_slice_intermediate[v_i, v__, v_k])
                var_slice_intermediate[v_i, v__, v_k] = lv4113[v_i, T.int64(0), v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_slice_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = var_slice_intermediate[v_i0, v_i1, v_i2]

    @T.prim_func
    def fused_transpose4_reshape4(lv2105: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv2105[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2105[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(2560) // T.int64(80), v_ax2 % T.int64(80)]

    @T.prim_func
    def layer_norm(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32"), B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float32")):
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
    def reshape(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(A[T.int64(0), T.int64(0)])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]

    @T.prim_func
    def reshape1(A: T.Buffer((T.int64(1), T.int64(2560)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2560)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax2 % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(2560)]

    @T.prim_func
    def reshape2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2560)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), T.int64(0), (v_ax2 * T.int64(80) + v_ax3) % T.int64(2560)]

    @T.prim_func
    def softmax2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(50432)), "float32")):
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
    def squeeze(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_squeeze: T.Buffer((T.int64(1), T.int64(32), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(80)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def take_decode(A: T.Buffer((T.int64(50432), T.int64(320)), "uint32"), B: T.Buffer((T.int64(50432), T.int64(80)), "float16"), C: T.Buffer((T.int64(1),), "int32"), take_decode_1: T.Buffer((T.int64(1), T.int64(2560)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(1), T.int64(2560)):
            with T.block("take_decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[C[v_i], v_j // T.int64(8)], C[v_i], B[C[v_i], v_j // T.int64(32)])
                T.writes(take_decode_1[v_i, v_j])
                take_decode_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[C[v_i], v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[C[v_i], v_j // T.int64(32)]

    @T.prim_func
    def transpose2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(80)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    ####################################### Dynamic Shape #######################################

    @T.prim_func
    def fused_NT_matmul1_add4(p_lv9: T.handle, lv1173: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), linear_bias: T.Buffer((T.int64(2560),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv9[v_i0, v_i1, v_k], lv1173[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv9[v_i0, v_i1, v_k] * lv1173[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], linear_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + linear_bias[v_ax2]

    @T.prim_func
    def fused_NT_matmul1_add4_add5(p_lv49: T.handle, lv1194: T.Buffer((T.int64(2560), T.int64(2560)), "float16"), linear_bias3: T.Buffer((T.int64(2560),), "float16"), p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv49 = T.match_buffer(p_lv49, (T.int64(1), n, T.int64(2560)), "float16")
        lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv49[v_i0, v_i1, v_k], lv1194[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv49[v_i0, v_i1, v_k] * lv1194[v_i2, v_k]
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
    def fused_NT_matmul2_divide1_maximum1_minimum1_cast9(p_lv36: T.handle, p_lv37: T.handle, p_lv5: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        m = T.meta_var(T.int64(128))
        lv36 = T.match_buffer(p_lv36, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        lv37 = T.match_buffer(p_lv37, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv36[v_i0, v_i1, v_i2, v_k], lv37[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv36[v_i0, v_i1, v_i2, v_k] * lv37[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.11179039301310044)
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
    def fused_NT_matmul3_add6_gelu1_cast11(p_lv57: T.handle, lv1201: T.Buffer((T.int64(10240), T.int64(2560)), "float16"), linear_bias4: T.Buffer((T.int64(10240),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv57 = T.match_buffer(p_lv57, (T.int64(1), n, T.int64(2560)), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(10240)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_multiply = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        compute = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        T_add = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        var_T_multiply_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(10240)))
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(10240), T.int64(2560)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv57[v_i0, v_i1, v_k], lv1201[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv57[v_i0, v_i1, v_k] * lv1201[v_i2, v_k])
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
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_multiply_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def fused_NT_matmul4_add7_cast8_cast12_add5(p_lv63: T.handle, lv1208: T.Buffer((T.int64(2560), T.int64(10240)), "float16"), linear_bias5: T.Buffer((T.int64(2560),), "float32"), p_lv53: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv63 = T.match_buffer(p_lv63, (T.int64(1), n, T.int64(10240)), "float16")
        lv53 = T.match_buffer(p_lv53, (T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_compute_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv63[v_i0, v_i1, v_k], lv1208[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv63[v_i0, v_i1, v_k] * lv1208[v_i2, v_k])
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
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate_1[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_compute_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
                var_compute_intermediate_1[v_i0, v_i1, v_i2] = var_compute_intermediate[v_i0, v_i1, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv53[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv53[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def fused_NT_matmul4_add7_cast8_cast12_add5_cast7(p_lv2047: T.handle, lv2510: T.Buffer((T.int64(2560), T.int64(10240)), "float16"), linear_bias191: T.Buffer((T.int64(2560),), "float32"), p_lv2037: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv2047 = T.match_buffer(p_lv2047, (T.int64(1), n, T.int64(10240)), "float16")
        lv2037 = T.match_buffer(p_lv2037, (T.int64(1), n, T.int64(2560)), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        var_compute_intermediate_2 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(2560)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(2560), T.int64(10240)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv2047[v_i0, v_i1, v_k], lv2510[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv2047[v_i0, v_i1, v_k] * lv2510[v_i2, v_k])
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
                T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
                var_compute_intermediate_1[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_compute_intermediate_1[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate_2[v_i0, v_i1, v_i2])
                var_compute_intermediate_2[v_i0, v_i1, v_i2] = var_compute_intermediate_1[v_i0, v_i1, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate_2[v_ax0, v_ax1, v_ax2], lv2037[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_2[v_ax0, v_ax1, v_ax2] + lv2037[v_ax0, v_ax1, v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_T_add_intermediate_1[v_i0, v_i1, v_i2])

    @T.prim_func
    def fused_NT_matmul_divide_maximum_minimum_cast2(lv2094: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16"), p_lv2095: T.handle, p_lv2063: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv2095 = T.match_buffer(p_lv2095, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        lv2063 = T.match_buffer(p_lv2063, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
        var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
        var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(80)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv2094[v_i0, v_i1, v_i2, v_k], lv2095[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv2094[v_i0, v_i1, v_i2, v_k] * lv2095[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.11179039301310044)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2063[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2063[v_ax0, T.int64(0), v_ax2, v_ax3])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def fused_layer_norm1_cast8(p_lv6: T.handle, weight1: T.Buffer((T.int64(2560),), "float32"), bias: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
        var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), n))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), n))
        var_T_layer_norm_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
        for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(lv6[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], weight1[v_ax2], bias[v_ax2])
                T.writes(var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_ax2] + bias[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def layer_norm1(var_A: T.handle, B: T.Buffer((T.int64(2560),), "float32"), C: T.Buffer((T.int64(2560),), "float32"), var_T_layer_norm: T.handle):
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
    def matmul3(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(80)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(32))
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(80), n):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func
    def matmul9(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.meta_var(T.int64(128))
        m = T.meta_var(T.int64(32))
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(80)), "float16")
        matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(80), m):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

# fmt: on
