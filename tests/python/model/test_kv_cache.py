# pylint: disable=line-too-long,missing-docstring
import tvm
from tvm import tir
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

from mlc_chat.nn.kv_cache import FlashInferPagedKVCache, PagedKVCache

# mypy: disable-error-code="attr-defined"
# pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-statements


def test_nn_module_paged_kv_cache():
    # fmt: off
    @I.ir_module
    class Module:
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
            num_pages, page_size = T.int64(), T.int64(is_size_var=True)
            pages = T.match_buffer(var_pages, (num_pages, 2, 32, page_size, 128), "float16")
            ntoken = T.int64(is_size_var=True)
            k_data = T.match_buffer(var_k_data, (ntoken, 32, 128), "float16")
            v_data = T.match_buffer(var_v_data, (ntoken, 32, 128), "float16")
            position_map = T.match_buffer(var_position_map, (ntoken,), "int32")
            # with T.block("root"):
            for global_pos, h, f in T.grid(ntoken, 32, 128):
                with T.block("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[T.Cast("int64", position_map[vgpos]) // page_size, 0, vh, T.Cast("int64", position_map[vgpos]) % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.Cast("int64", position) // page_size, 0, vh, T.Cast("int64", position) % page_size, vf] = k_data[vgpos, vh, vf]
                with T.block("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[T.Cast("int64", position_map[vgpos]) // page_size, 1, vh, T.Cast("int64", position_map[vgpos]) % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.Cast("int64", position) // page_size, 1, vh, T.Cast("int64", position) % page_size, vf] = v_data[vgpos, vh, vf]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()  # type: ignore
                lv: R.Tuple(R.Object) = (_io,)  # type: ignore
                gv: R.Tuple(R.Object) = lv  # type: ignore
                R.output(gv)
            return gv

        @R.function
        def create_flashinfer_paged_kv_cache(max_batch_size: R.Shape(["max_batch_size_1"]), max_total_seq_len: R.Shape(["max_total_seq_len_1"]), page_size: R.Shape(["page_size_1"]), _io: R.Object) -> R.Tuple(R.Object, R.Tuple(R.Object)):
            max_batch_size_1 = T.int64()
            max_total_seq_len_1 = T.int64()
            page_size_1 = T.int64()
            R.func_attr({"num_input": 4})
            cls = Module
            with R.dataflow():
                lv2: R.Tensor((), dtype="float16") = R.zeros(R.shape([]), dtype="float16")  # type: ignore
                paged_kv_cache: R.Object = R.call_packed("vm.builtin.paged_attention_kv_cache_create", R.shape([max_batch_size_1, max_total_seq_len_1, page_size_1]), R.prim_value(32), R.prim_value(32), R.prim_value(32), R.prim_value(128), R.prim_value(1), R.prim_value(10000), lv2, cls.tir_kv_cache_transpose_append, R.ExternFunc("paged_kv_cache.attention_kernel_prefill"), R.ExternFunc("paged_kv_cache.attention_kernel_decode"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"), R.ExternFunc("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_prefill_begin_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_prefill_end_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_decode_begin_forward"), R.ExternFunc("paged_kv_cache.attention_kernel_decode_end_forward"), R.ExternFunc("flashinfer.batch_qk_apply_rotary_in_place"), R.ExternFunc("flashinfer.merge_state_in_place"), cls.tir_kv_cache_debug_get_kv, sinfo_args=(R.Object,)) # type: ignore
                gv2: R.Tuple(R.Object, R.Tuple(R.Object)) = paged_kv_cache, (_io,)  # type: ignore
                R.output(gv2)
            return gv2

        @R.function
        def forward(cache: R.Object, q: R.Tensor((1, 100, 32, 128), dtype="float16"), k: R.Tensor((1, 100, 32, 128), dtype="float16"), v: R.Tensor((1, 100, 32, 128), dtype="float16"), _io: R.Object) -> R.Tuple(R.Tensor((1, 100, 32, 128), dtype="float16"), R.Tuple(R.Object)):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                lv1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (cache, R.prim_value(0), q, k, v), out_sinfo=R.Tensor((1, 100, 32, 128), dtype="float16"))
                gv1: R.Tuple(R.Tensor((1, 100, 32, 128), dtype="float16"), R.Tuple(R.Object)) = lv1, (_io,)  # type: ignore
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
            self, max_batch_size: tir.Var, max_total_seq_len: tir.Var, page_size: tir.Var
        ) -> PagedKVCache:
            return FlashInferPagedKVCache(
                max_batch_size=max_batch_size,
                max_total_seq_len=max_total_seq_len,
                page_size=page_size,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                rope_scale=1,
                rope_theta=10000,
                dtype="float16",
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
                "page_size": int,
            },
        },
        debug=True,
    )
    tvm_mod = export_results[0]
    tvm.ir.assert_structural_equal(tvm_mod, Module, True)


if __name__ == "__main__":
    test_nn_module_paged_kv_cache()
