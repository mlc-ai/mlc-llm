# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def extend_te(var_A: T.handle, var_concat_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(1), n, n), "float16")
        m = T.int64()
        concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), n, m), "float16")
        # with T.block("root"):
        for b, _, i, j in T.grid(T.int64(1), T.int64(1), n, m):
            with T.block("concat_te"):
                v_b, v__, v_i, v_j = T.axis.remap("SSSS", [b, _, i, j])
                T.reads(A[v_b, v__, v_i, v_j + n - m])
                T.writes(concat_te[v_b, v__, v_i, v_j])
                concat_te[v_b, v__, v_i, v_j] = T.if_then_else(v_j < m - n, T.float16(65504), A[v_b, v__, v_i, v_j + n - m])

    @T.prim_func
    def min_max_triu_te(var_make_diag_mask_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        make_diag_mask_te = T.match_buffer(var_make_diag_mask_te, (n, n), "float16")
        # with T.block("root"):
        for i, j in T.grid(n, n):
            with T.block("make_diag_mask_te"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads()
                T.writes(make_diag_mask_te[v_i, v_j])
                make_diag_mask_te[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(65504))

    @T.prim_func
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

    @T.prim_func
    def rms_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), T.int64(1)))
        for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm[v_bsz, v_i, v_k])
                rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

    @T.prim_func
    def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(2048), T.int64(128)), "float16"), C: T.Buffer((T.int64(2048), T.int64(128)), "float16"), var_rotary: T.handle, m: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(B[m + v_i1 - n, v_i3], A[v_i0, v_i1, v_i2, v_i3 - T.int64(64):v_i3 - T.int64(64) + T.int64(129)], C[m + v_i1 - n, v_i3])
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - n, v_i3] * A[v_i0, v_i1, v_i2, v_i3] + C[m + v_i1 - n, v_i3] * T.Select(T.int64(64) <= v_i3, A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)], A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1))

    @T.prim_func
    def rotary_embedding1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), B: T.Buffer((T.int64(2048), T.int64(128)), "float16"), C: T.Buffer((T.int64(2048), T.int64(128)), "float16"), rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), n: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(B[n + v_i1 - T.int64(1), v_i3], A[v_i0, v_i1, v_i2, v_i3 - T.int64(64):v_i3 - T.int64(64) + T.int64(129)], C[n + v_i1 - T.int64(1), v_i3])
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[n + v_i1 - T.int64(1), v_i3] * A[v_i0, v_i1, v_i2, v_i3] + C[n + v_i1 - T.int64(1), v_i3] * T.Select(T.int64(64) <= v_i3, A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)], A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1))

    @T.prim_func
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

    @T.prim_func
    def slice1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), slice: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with T.block("slice"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(A[v_i, T.int64(0), v_k])
                T.writes(slice[v_i, v_j, v_k])
                slice[v_i, v_j, v_k] = A[v_i, T.int64(0), v_k]

    @R.function
    def create_kv_cache() -> R.Tuple(R.Object, R.Object):
        R.func_attr({"tir_var_upper_bound": {"m": 2048, "n": 2048}})
        with R.dataflow():
            lv118: R.Tensor((2048, 32, 128), dtype="float16") = R.zeros(R.shape([2048, 32, 128]), dtype="float16")
            lv119: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv118, R.shape([2048, 32, 128]), R.prim_value(0), sinfo_args=(R.Object,))
            lv120: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv118, R.shape([2048, 32, 128]), R.prim_value(0), sinfo_args=(R.Object,))
            gv2: R.Tuple(R.Object, R.Object) = lv119, lv120
            R.output(gv2)
        return gv2

    @R.function
    def decoding(input_ids1: R.Tensor((1, 1), dtype="int32"), all_seq_len: R.Shape(["n"]), kv_cache: R.Tuple(R.Object, R.Object), embedding_weight1: R.Tensor((32000, 4096), dtype="float16"), q_proj_weight1: R.Tensor((4096, 4096), dtype="float16"), k_proj_weight1: R.Tensor((4096, 4096), dtype="float16"), v_proj_weight1: R.Tensor((4096, 4096), dtype="float16"), o_proj_weight1: R.Tensor((4096, 4096), dtype="float16"), gate_proj_weight1: R.Tensor((11008, 4096), dtype="float16"), down_proj_weight1: R.Tensor((4096, 11008), dtype="float16"), up_proj_weight1: R.Tensor((11008, 4096), dtype="float16"), rms_norm_weight3: R.Tensor((4096,), dtype="float16"), rms_norm_weight4: R.Tensor((4096,), dtype="float16"), rms_norm_weight5: R.Tensor((4096,), dtype="float16"), lm_head_weight1: R.Tensor((32000, 4096), dtype="float16"), cos_cached1: R.Tensor((2048, 128), dtype="float16"), sin_cached1: R.Tensor((2048, 128), dtype="float16")) -> R.Tuple(R.Tensor((1, 1, 32000), dtype="float32"), R.Tuple(R.Object, R.Object)):
        n = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 2048, "n": 2048}})
        cls = Module
        with R.dataflow():
            lv60: R.Tensor((1,), dtype="int32") = R.reshape(input_ids1, R.shape([1]))
            lv61: R.Tensor((1, 4096), dtype="float16") = R.take(embedding_weight1, lv60, axis=0)
            lv62: R.Tensor((1, 1, 4096), dtype="float16") = R.reshape(lv61, R.shape([1, 1, 4096]))
            lv63: R.Tensor((1, 1, 1, n), dtype="float16") = R.full(R.shape([1, 1, 1, n]), metadata["relax.expr.Constant"][0], dtype="float16")
            lv64 = R.call_tir(cls.rms_norm1, (lv62, rms_norm_weight3), out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"))
            lv65: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(q_proj_weight1, axes=None)
            lv66: R.Tensor((1, 1, 4096), dtype="float16") = R.matmul(lv64, lv65, out_dtype="void")
            lv67: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv66, R.shape([1, 1, 32, 128]))
            lv68: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(k_proj_weight1, axes=None)
            lv69: R.Tensor((1, 1, 4096), dtype="float16") = R.matmul(lv64, lv68, out_dtype="void")
            lv70: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv69, R.shape([1, 1, 32, 128]))
            lv71: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(v_proj_weight1, axes=None)
            lv72: R.Tensor((1, 1, 4096), dtype="float16") = R.matmul(lv64, lv71, out_dtype="void")
            lv73: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv72, R.shape([1, 1, 32, 128]))
            lv74 = R.call_tir(cls.rotary_embedding1, (lv67, cos_cached1, sin_cached1), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float16"), tir_vars=R.shape([n]))
            lv75 = R.call_tir(cls.rotary_embedding1, (lv70, cos_cached1, sin_cached1), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float16"), tir_vars=R.shape([n]))
            lv76: R.Tensor((1, 32, 128), dtype="float16") = R.squeeze(lv75, axis=[0])
            lv77: R.Tensor((1, 32, 128), dtype="float16") = R.squeeze(lv73, axis=[0])
            lv78: R.Object = kv_cache[0]
            lv79: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv78, lv76, sinfo_args=(R.Object,))
            lv80: R.Object = kv_cache[1]
            lv81: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv80, lv77, sinfo_args=(R.Object,))
            lv82: R.Tensor((n, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv79, R.shape([n, 32, 128]), sinfo_args=(R.Tensor((n, 32, 128), dtype="float16"),))
            lv83: R.Tensor((n, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv81, R.shape([n, 32, 128]), sinfo_args=(R.Tensor((n, 32, 128), dtype="float16"),))
            lv84: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv82, R.shape([1, n, 32, 128]))
            lv85: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv83, R.shape([1, n, 32, 128]))
            lv86: R.Tensor((1, 32, 1, 128), dtype="float16") = R.permute_dims(lv74, axes=[0, 2, 1, 3])
            lv87: R.Tensor((1, 32, n, 128), dtype="float16") = R.permute_dims(lv84, axes=[0, 2, 1, 3])
            lv88: R.Tensor((1, 32, n, 128), dtype="float16") = R.permute_dims(lv85, axes=[0, 2, 1, 3])
            lv89: R.Tensor((1, 32, 128, n), dtype="float16") = R.permute_dims(lv87, axes=[0, 1, 3, 2])
            lv90: R.Tensor((1, 32, 1, n), dtype="float16") = R.matmul(lv86, lv89, out_dtype="void")
            lv91: R.Tensor((1, 32, 1, n), dtype="float16") = R.divide(lv90, metadata["relax.expr.Constant"][1])
            lv92: R.Tensor((1, 32, 1, n), dtype="float16") = R.maximum(lv91, metadata["relax.expr.Constant"][2])
            lv93: R.Tensor((1, 32, 1, n), dtype="float16") = R.minimum(lv92, lv63)
            lv94: R.Tensor((1, 32, 1, n), dtype="float32") = R.astype(lv93, dtype="float32")
            lv95: R.Tensor((1, 32, 1, n), dtype="float32") = R.nn.softmax(lv94, axis=-1)
            lv96: R.Tensor((1, 32, 1, n), dtype="float16") = R.astype(lv95, dtype="float16")
            lv97: R.Tensor((1, 32, 1, 128), dtype="float16") = R.matmul(lv96, lv88, out_dtype="void")
            lv98: R.Tensor((1, 1, 32, 128), dtype="float16") = R.permute_dims(lv97, axes=[0, 2, 1, 3])
            lv99: R.Tensor((1, 1, 4096), dtype="float16") = R.reshape(lv98, R.shape([1, 1, 4096]))
            lv100: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(o_proj_weight1, axes=None)
            lv101: R.Tensor((1, 1, 4096), dtype="float16") = R.matmul(lv99, lv100, out_dtype="void")
            lv102: R.Tensor((1, 1, 4096), dtype="float16") = R.add(lv62, lv101)
            lv103 = R.call_tir(cls.rms_norm1, (lv102, rms_norm_weight4), out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"))
            lv104: R.Tensor((4096, 11008), dtype="float16") = R.permute_dims(gate_proj_weight1, axes=None)
            lv105: R.Tensor((1, 1, 11008), dtype="float16") = R.matmul(lv103, lv104, out_dtype="void")
            lv106: R.Tensor((4096, 11008), dtype="float16") = R.permute_dims(up_proj_weight1, axes=None)
            lv107: R.Tensor((1, 1, 11008), dtype="float16") = R.matmul(lv103, lv106, out_dtype="void")
            lv108: R.Tensor((1, 1, 11008), dtype="float16") = R.nn.silu(lv105)
            lv109: R.Tensor((1, 1, 11008), dtype="float16") = R.multiply(lv108, lv107)
            lv110: R.Tensor((11008, 4096), dtype="float16") = R.permute_dims(down_proj_weight1, axes=None)
            lv111: R.Tensor((1, 1, 4096), dtype="float16") = R.matmul(lv109, lv110, out_dtype="void")
            lv112: R.Tensor((1, 1, 4096), dtype="float16") = R.add(lv102, lv111)
            lv113 = R.call_tir(cls.rms_norm1, (lv112, rms_norm_weight5), out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"))
            lv114 = R.call_tir(cls.slice1, (lv113,), out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"))
            lv115: R.Tensor((4096, 32000), dtype="float16") = R.permute_dims(lm_head_weight1, axes=None)
            lv116: R.Tensor((1, 1, 32000), dtype="float16") = R.matmul(lv114, lv115, out_dtype="void")
            lv117: R.Tensor((1, 1, 32000), dtype="float32") = R.astype(lv116, dtype="float32")
            gv1: R.Tuple(R.Tensor((1, 1, 32000), dtype="float32"), R.Tuple(R.Object, R.Object)) = lv117, (lv79, lv81)
            R.output(gv1)
        return gv1

    @R.function
    def encoding(input_ids: R.Tensor((1, "n"), dtype="int32"), all_seq_len: R.Shape(["m"]), kv_cache: R.Tuple(R.Object, R.Object), embedding_weight: R.Tensor((32000, 4096), dtype="float16"), q_proj_weight: R.Tensor((4096, 4096), dtype="float16"), k_proj_weight: R.Tensor((4096, 4096), dtype="float16"), v_proj_weight: R.Tensor((4096, 4096), dtype="float16"), o_proj_weight: R.Tensor((4096, 4096), dtype="float16"), gate_proj_weight: R.Tensor((11008, 4096), dtype="float16"), down_proj_weight: R.Tensor((4096, 11008), dtype="float16"), up_proj_weight: R.Tensor((11008, 4096), dtype="float16"), rms_norm_weight: R.Tensor((4096,), dtype="float16"), rms_norm_weight1: R.Tensor((4096,), dtype="float16"), rms_norm_weight2: R.Tensor((4096,), dtype="float16"), lm_head_weight: R.Tensor((32000, 4096), dtype="float16"), cos_cached: R.Tensor((2048, 128), dtype="float16"), sin_cached: R.Tensor((2048, 128), dtype="float16")) -> R.Tuple(R.Tensor((1, 1, 32000), dtype="float32"), R.Tuple(R.Object, R.Object)):
        n = T.int64()
        m = T.int64()
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"m": 2048, "n": 2048}})
        cls = Module
        with R.dataflow():
            lv: R.Tensor((n,), dtype="int32") = R.reshape(input_ids, R.shape([n]))
            lv1: R.Tensor((n, 4096), dtype="float16") = R.take(embedding_weight, lv, axis=0)
            lv2: R.Tensor((1, n, 4096), dtype="float16") = R.reshape(lv1, R.shape([1, n, 4096]))
            lv3 = R.call_tir(cls.min_max_triu_te, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float16"))
            lv4: R.Tensor((1, 1, n, n), dtype="float16") = R.broadcast_to(lv3, R.shape([1, 1, n, n]))
            lv5 = R.call_tir(cls.extend_te, (lv4,), out_sinfo=R.Tensor((1, 1, n, m), dtype="float16"))
            lv6 = R.call_tir(cls.rms_norm, (lv2, rms_norm_weight), out_sinfo=R.Tensor((1, n, 4096), dtype="float16"))
            lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(q_proj_weight, axes=None)
            lv8: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv7, out_dtype="void")
            lv9: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv8, R.shape([1, n, 32, 128]))
            lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(k_proj_weight, axes=None)
            lv11: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv10, out_dtype="void")
            lv12: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv11, R.shape([1, n, 32, 128]))
            lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(v_proj_weight, axes=None)
            lv14: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv13, out_dtype="void")
            lv15: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv14, R.shape([1, n, 32, 128]))
            lv16 = R.call_tir(cls.rotary_embedding, (lv9, cos_cached, sin_cached), out_sinfo=R.Tensor((1, n, 32, 128), dtype="float16"), tir_vars=R.shape([m]))
            lv17 = R.call_tir(cls.rotary_embedding, (lv12, cos_cached, sin_cached), out_sinfo=R.Tensor((1, n, 32, 128), dtype="float16"), tir_vars=R.shape([m]))
            lv18: R.Tensor((n, 32, 128), dtype="float16") = R.squeeze(lv17, axis=[0])
            lv19: R.Tensor((n, 32, 128), dtype="float16") = R.squeeze(lv15, axis=[0])
            lv20: R.Object = kv_cache[0]
            lv21: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv20, lv18, sinfo_args=(R.Object,))
            lv22: R.Object = kv_cache[1]
            lv23: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv22, lv19, sinfo_args=(R.Object,))
            lv24: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv21, R.shape([m, 32, 128]), sinfo_args=(R.Tensor((m, 32, 128), dtype="float16"),))
            lv25: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv23, R.shape([m, 32, 128]), sinfo_args=(R.Tensor((m, 32, 128), dtype="float16"),))
            lv26: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(lv24, R.shape([1, m, 32, 128]))
            lv27: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(lv25, R.shape([1, m, 32, 128]))
            lv28: R.Tensor((1, 32, n, 128), dtype="float16") = R.permute_dims(lv16, axes=[0, 2, 1, 3])
            lv29: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(lv26, axes=[0, 2, 1, 3])
            lv30: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(lv27, axes=[0, 2, 1, 3])
            lv31: R.Tensor((1, 32, 128, m), dtype="float16") = R.permute_dims(lv29, axes=[0, 1, 3, 2])
            lv32: R.Tensor((1, 32, n, m), dtype="float16") = R.matmul(lv28, lv31, out_dtype="void")
            lv33: R.Tensor((1, 32, n, m), dtype="float16") = R.divide(lv32, metadata["relax.expr.Constant"][3])
            lv34: R.Tensor((1, 32, n, m), dtype="float16") = R.maximum(lv33, metadata["relax.expr.Constant"][4])
            lv35: R.Tensor((1, 32, n, m), dtype="float16") = R.minimum(lv34, lv5)
            lv36: R.Tensor((1, 32, n, m), dtype="float32") = R.astype(lv35, dtype="float32")
            lv37: R.Tensor((1, 32, n, m), dtype="float32") = R.nn.softmax(lv36, axis=-1)
            lv38: R.Tensor((1, 32, n, m), dtype="float16") = R.astype(lv37, dtype="float16")
            lv39: R.Tensor((1, 32, n, 128), dtype="float16") = R.matmul(lv38, lv30, out_dtype="void")
            lv40: R.Tensor((1, n, 32, 128), dtype="float16") = R.permute_dims(lv39, axes=[0, 2, 1, 3])
            lv41: R.Tensor((1, n, 4096), dtype="float16") = R.reshape(lv40, R.shape([1, n, 4096]))
            lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(o_proj_weight, axes=None)
            lv43: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv41, lv42, out_dtype="void")
            lv44: R.Tensor((1, n, 4096), dtype="float16") = R.add(lv2, lv43)
            lv45 = R.call_tir(cls.rms_norm, (lv44, rms_norm_weight1), out_sinfo=R.Tensor((1, n, 4096), dtype="float16"))
            lv46: R.Tensor((4096, 11008), dtype="float16") = R.permute_dims(gate_proj_weight, axes=None)
            lv47: R.Tensor((1, n, 11008), dtype="float16") = R.matmul(lv45, lv46, out_dtype="void")
            lv48: R.Tensor((4096, 11008), dtype="float16") = R.permute_dims(up_proj_weight, axes=None)
            lv49: R.Tensor((1, n, 11008), dtype="float16") = R.matmul(lv45, lv48, out_dtype="void")
            lv50: R.Tensor((1, n, 11008), dtype="float16") = R.nn.silu(lv47)
            lv51: R.Tensor((1, n, 11008), dtype="float16") = R.multiply(lv50, lv49)
            lv52: R.Tensor((11008, 4096), dtype="float16") = R.permute_dims(down_proj_weight, axes=None)
            lv53: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv51, lv52, out_dtype="void")
            lv54: R.Tensor((1, n, 4096), dtype="float16") = R.add(lv44, lv53)
            lv55 = R.call_tir(cls.rms_norm, (lv54, rms_norm_weight2), out_sinfo=R.Tensor((1, n, 4096), dtype="float16"))
            lv56 = R.call_tir(cls.slice, (lv55,), out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"))
            lv57: R.Tensor((4096, 32000), dtype="float16") = R.permute_dims(lm_head_weight, axes=None)
            lv58: R.Tensor((1, 1, 32000), dtype="float16") = R.matmul(lv56, lv57, out_dtype="void")
            lv59: R.Tensor((1, 1, 32000), dtype="float32") = R.astype(lv58, dtype="float32")
            gv: R.Tuple(R.Tensor((1, 1, 32000), dtype="float32"), R.Tuple(R.Object, R.Object)) = lv59, (lv21, lv23)
            R.output(gv)
        return gv

    @R.function
    def softmax_with_temperature(logits: R.Tensor((1, 1, 32000), dtype="float32"), temperature: R.Tensor((), dtype="float32")) -> R.Tensor((1, 1, 32000), dtype="float32"):
        R.func_attr({"tir_var_upper_bound": {"m": 2048, "n": 2048}})
        with R.dataflow():
            lv121: R.Tensor((1, 1, 32000), dtype="float32") = R.divide(logits, temperature)
            lv122: R.Tensor((1, 1, 32000), dtype="float32") = R.nn.softmax(lv121, axis=-1)
            gv3: R.Tensor((1, 1, 32000), dtype="float32") = lv122
            R.output(gv3)
        return gv3

# Metadata omitted. Use show_meta=True in script() method to show it.