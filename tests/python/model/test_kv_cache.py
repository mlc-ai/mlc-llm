# pylint: disable=line-too-long,missing-docstring
import tvm
from tvm import tir
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

from mlc_chat.nn.kv_cache import FlashInferPagedKVCache, PagedKVCache, RopeMode

# mypy: disable-error-code="attr-defined"
# pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-statements


def test_nn_module_paged_kv_cache():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def fused_rope(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, apply_rope: T.int32):  # pylint: disable=too-many-arguments
            T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
            seq_len = T.int64()
            qkv = T.match_buffer(var_qkv, (seq_len, 96, 128), "float16")
            position_map = T.match_buffer(var_position_map, (seq_len,), "int32")
            q = T.match_buffer(var_q, (seq_len, 32, 128), "float16")
            k = T.match_buffer(var_k, (seq_len, 32, 128), "float16")
            v = T.match_buffer(var_v, (seq_len, 32, 128), "float16")
            for iters_0, iters_1, iters_2 in T.grid(seq_len, 96, 128):
                with T.block("llama_fused_rope"):
                    s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                    T.reads(position_map[s], qkv[s, h, d - 64:d - 64 + 129])
                    T.writes(q[s, h, d], k[s, h - 32, d], v[s, h - 64, d])
                    if h < 32:
                        q[s, h, d] = T.if_then_else(apply_rope > 0 and d < 128, T.Cast("float16", T.cos(T.Cast("float32", T.Cast("float16", position_map[s])) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * qkv[s, h, d] + T.Cast("float16", T.sin(T.Cast("float32", T.Cast("float16", position_map[s])) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * T.if_then_else(d < 64, qkv[s, h, d + 64] * T.float16(-1), qkv[s, h, d - 64]), qkv[s, h, d])
                    else:
                        if h < 64:
                            k[s, h - 32, d] = T.if_then_else(apply_rope > 0 and d < 128, T.Cast("float16", T.cos(T.Cast("float32", T.Cast("float16", position_map[s])) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * qkv[s, h, d] + T.Cast("float16", T.sin(T.Cast("float32", T.Cast("float16", position_map[s])) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * T.if_then_else(d < 64, qkv[s, h, d + 64] * T.float16(-1), qkv[s, h, d - 64]), qkv[s, h, d])
                        else:
                            v[s, h - 64, d] = qkv[s, h, d]

        @T.prim_func
        def tir_kv_cache_debug_get_kv(var_pages: T.handle, var_position_map: T.handle, var_k_data: T.handle, var_v_data: T.handle, layer_id: T.int64):
            T.func_attr({"tir.noalias": T.bool(True)})
            num_pages, page_size = T.int64(), T.int64(is_size_var=True)
            pages = T.match_buffer(var_pages, (num_pages, 2, 32, page_size, 128), "float16")
            seqlen = T.int64(is_size_var=True)
            position_map = T.match_buffer(var_position_map, (seqlen,), "int32")
            k_data = T.match_buffer(var_k_data, (32, seqlen, 32, 128), "float16")
            v_data = T.match_buffer(var_v_data, (32, seqlen, 32, 128), "float16")
            for p, h, d in T.grid(seqlen, 32, 128):
                with T.block("copy0"):
                    vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                    T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, 0:2, vh, T.Cast("int64", position_map[vp]) % page_size, vd])
                    T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                    position: T.int32 = position_map[vp] # type: ignore[name-defined]
                    k_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 0, vh, T.Cast("int64", position) % page_size, vd]
                    v_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 1, vh, T.Cast("int64", position) % page_size, vd]

        @T.prim_func
        def tir_kv_cache_transpose_append(var_pages: T.handle, var_k_data: T.handle, var_v_data: T.handle, var_position_map: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            num_pages = T.int64()
            pages = T.match_buffer(var_pages, (num_pages, 2, 32, 16, 128), "float16")
            ntoken = T.int64(is_size_var=True)
            k_data = T.match_buffer(var_k_data, (ntoken, 32, 128), "float16")
            v_data = T.match_buffer(var_v_data, (ntoken, 32, 128), "float16")
            position_map = T.match_buffer(var_position_map, (ntoken,), "int32")
            # with T.block("root"):
            for global_pos, h, f in T.grid(ntoken, 32, 128):
                with T.block("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore[no-redef]
                    pages[position // 16, 0, vh, position % 16, vf] = k_data[vgpos, vh, vf]
                with T.block("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore[no-redef]
                    pages[position // 16, 1, vh, position % 16, vf] = v_data[vgpos, vh, vf]

        @T.prim_func
        def tir_rotary(var_q: T.handle, var_k: T.handle, var_append_len_indptr: T.handle, var_rope_offsets: T.handle, _0: T.int32, _1: T.int32, _2: T.int32, _3: T.int32, _4: T.int32, _5: T.float32, _6: T.float32):
            T.func_attr({"tir.is_scheduled": 1})
            total_len = T.int32()
            q = T.match_buffer(var_q, (total_len, 32, 128), "float16")
            k = T.match_buffer(var_k, (total_len, 32, 128), "float16")
            batch_size = T.int32()
            append_len_indptr = T.match_buffer(var_append_len_indptr, (batch_size + 1,), "int32")
            rope_offsets = T.match_buffer(var_rope_offsets, (batch_size,), "int32")
            with T.block(""):
                T.reads()
                T.writes()
                for b_h in T.thread_binding(batch_size * 64, thread="blockIdx.x"):  # pylint: disable=too-many-nested-blocks
                    b: T.int32 = b_h // 64
                    h: T.int32 = b_h % 64
                    instance_offset: T.int32 = append_len_indptr[b]
                    rope_offset: T.int32 = rope_offsets[b]
                    append_len: T.int32 = append_len_indptr[b + 1] - append_len_indptr[b]
                    for s0 in range((append_len + 31) // 32):
                        for s1 in T.thread_binding(32, thread="threadIdx.y"):
                            for d0 in T.thread_binding(32, thread="threadIdx.x"):
                                for d1 in T.vectorized(4):
                                    s: T.int32 = s0 * 32 + s1
                                    d: T.int32 = d0 * 4 + d1
                                    if s < append_len and d < 128:
                                        if h < 32:
                                            q[s + instance_offset, h, d] = T.Cast("float16", T.cos(T.Cast("float32", s + rope_offset) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * q[s + instance_offset, h, d] + T.Cast("float16", T.sin(T.Cast("float32", s + rope_offset) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * T.if_then_else(d < 64, q[s + instance_offset, h, d + 64] * T.float16(-1), q[s + instance_offset, h, d - 64])
                                        else:
                                            k[s + instance_offset, h - 32, d] = T.Cast("float16", T.cos(T.Cast("float32", s + rope_offset) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * k[s + instance_offset, h - 32, d] + T.Cast("float16", T.sin(T.Cast("float32", s + rope_offset) / T.pow(T.float32(10000), T.Cast("float32", d * 2 % 128) / T.float32(128)))) * T.if_then_else(d < 64, k[s + instance_offset, h - 32, d + 64] * T.float16(-1), k[s + instance_offset, h - 32, d - 64])

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()  # type: ignore
                lv: R.Tuple(R.Object) = (_io,)  # type: ignore
                gv: R.Tuple(R.Object) = lv  # type: ignore
                R.output(gv)
            return gv

        @R.function
        def create_flashinfer_paged_kv_cache(max_batch_size: R.Shape(["max_batch_size_1"]), max_total_seq_len: R.Shape(["max_total_seq_len_1"]), prefill_chunk_size: R.Shape(["prefill_chunk_size_1"]), page_size: R.Shape(["page_size_1"]), _io: R.Object) -> R.Tuple(R.Object, R.Tuple(R.Object)):
            max_batch_size_1 = T.int64()
            max_total_seq_len_1 = T.int64()
            prefill_chunk_size_1 = T.int64()
            page_size_1 = T.int64()
            R.func_attr({"num_input": 5})
            cls = Module
            with R.dataflow():
                lv2: R.Tensor((), dtype="float16") = R.zeros(R.shape([]), dtype="float16")  # type: ignore
                paged_kv_cache: R.Object = R.call_packed("vm.builtin.paged_attention_kv_cache_create", R.shape([max_batch_size_1, max_total_seq_len_1, prefill_chunk_size_1, page_size_1]), R.prim_value(32), R.prim_value(32), R.prim_value(32), R.prim_value(128), R.prim_value(0), R.prim_value(1), R.prim_value(10000), lv2, cls.tir_kv_cache_transpose_append, R.ExternFunc("paged_kv_cache.attention_kernel_prefill"), R.ExternFunc("paged_kv_cache.attention_kernel_decode"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_prefill_begin_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_prefill_end_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_decode_begin_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_decode_end_forward"), R.ExternFunc("flashinfer.merge_state_in_place"), cls.fused_rope, cls.tir_rotary, cls.tir_kv_cache_debug_get_kv, sinfo_args=(R.Object,))
                gv2: R.Tuple(R.Object, R.Tuple(R.Object)) = paged_kv_cache, (_io,)  # type: ignore
                R.output(gv2)
            return gv2

        @R.function
        def forward(cache: R.Object, q: R.Tensor((1, 100, 32, 128), dtype="float16"), k: R.Tensor((1, 100, 32, 128), dtype="float16"), v: R.Tensor((1, 100, 32, 128), dtype="float16"), _io: R.Object) -> R.Tuple(R.Tensor((1, 100, 32, 128), dtype="float16"), R.Tuple(R.Object)):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                reshape: R.Tensor((100, 32, 128), dtype="float16") = R.reshape(q, R.shape([100, 32, 128]))  # type: ignore
                reshape1: R.Tensor((100, 32, 128), dtype="float16") = R.reshape(k, R.shape([100, 32, 128]))  # type: ignore
                reshape2: R.Tensor((100, 32, 128), dtype="float16") = R.reshape(v, R.shape([100, 32, 128]))  # type: ignore
                lv1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (cache, R.prim_value(0), reshape, reshape1, reshape2), out_sinfo=R.Tensor((100, 32, 128), dtype="float16"))
                reshape3: R.Tensor((1, 100, 32, 128), dtype="float16") = R.reshape(lv1, R.shape([1, 100, 32, 128]))  # type: ignore
                gv1: R.Tuple(R.Tensor((1, 100, 32, 128), dtype="float16"), R.Tuple(R.Object)) = reshape3, (_io,)  # type: ignore
                R.output(gv1)
            return gv1
    # fmt: on

    class PagedKVCacheTest(modules.Module):
        def forward(
            self,
            cache: PagedKVCache,
            q: core.Tensor,
            k: core.Tensor,
            v: core.Tensor,
        ) -> core.Tensor:
            return cache.attention(0, q, k, v)

        def create_flashinfer_paged_kv_cache(
            self,
            max_batch_size: tir.Var,
            max_total_seq_len: tir.Var,
            prefill_chunk_size: tir.Var,
            page_size: tir.Var,
        ) -> PagedKVCache:
            return FlashInferPagedKVCache(
                max_batch_size=max_batch_size,
                max_total_seq_len=max_total_seq_len,
                prefill_chunk_size=prefill_chunk_size,
                page_size=page_size,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                rope_mode=RopeMode.NORMAL,
                rope_scale=1,
                rope_theta=10000,
                rotary_dim=128,
                dtype="float16",
                target=tvm.target.Target("cuda"),
            )

    export_results = PagedKVCacheTest().export_tvm(
        spec={
            "forward": {
                "cache": spec.Object(object_type=PagedKVCache),
                "q": spec.Tensor((1, 100, 32, 128), "float16"),
                "k": spec.Tensor((1, 100, 32, 128), "float16"),
                "v": spec.Tensor((1, 100, 32, 128), "float16"),
            },
            "create_flashinfer_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
            },
        },
        debug=True,
    )
    tvm_mod = export_results[0]
    tvm.ir.assert_structural_equal(tvm_mod, Module, True)


if __name__ == "__main__":
    test_nn_module_paged_kv_cache()
