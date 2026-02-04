# pylint: disable=line-too-long,missing-docstring
import tvm
from tvm import tir
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

from mlc_llm.nn.kv_cache import PagedKVCache, RopeMode

# mypy: disable-error-code="attr-defined"
# pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-statements


def test_nn_module_paged_kv_cache():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def create_paged_kv_cache(
            max_batch_size: R.Shape(["max_batch_size_1"]),  # type: ignore
            max_total_seq_len: R.Shape(["max_total_seq_len_1"]),  # type: ignore
            prefill_chunk_size: R.Shape(["prefill_chunk_size_1"]),  # type: ignore
            page_size: R.Shape(["page_size_1"]),  # type: ignore
            support_sliding_window: R.Shape(["support_sliding_window_1"]),  # type: ignore
        ) -> R.Object:
            max_batch_size_1 = T.int64()
            max_total_seq_len_1 = T.int64()
            prefill_chunk_size_1 = T.int64()
            page_size_1 = T.int64()
            support_sliding_window_1 = T.int64()
            R.func_attr({"num_input": 5})
            with R.dataflow():
                paged_kv_cache: R.Object = R.call_pure_packed("mlc.create_paged_kv_cache_generic", R.shape([max_batch_size_1, max_total_seq_len_1, prefill_chunk_size_1, page_size_1, support_sliding_window_1]), R.prim_value(32), R.prim_value(32), R.prim_value(32), R.prim_value(128), R.prim_value(1), R.prim_value(1), R.prim_value(10000), R.prim_value(128), R.dtype("float16"), sinfo_args=(R.Object,))
                gv1: R.Object = paged_kv_cache
                R.output(gv1)
            return gv1

        @R.function
        def forward(
            cache: R.Object, qkv: R.Tensor((1, 100, 96, 128), dtype="float16")  # type: ignore
        ) -> R.Tensor((1, 100, 32, 128), dtype="float16"):  # type: ignore
            R.func_attr({"num_input": 2})
            with R.dataflow():
                reshape: R.Tensor((100, 96, 128), dtype="float16") = R.reshape(  # type: ignore
                    qkv, R.shape([100, 96, 128])
                )
                lv = R.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    (cache, R.prim_value(0), R.prim_value(T.float32(1)), reshape),
                    out_sinfo=R.Tensor((100, 32, 128), dtype="float16"),
                )
                reshape1: R.Tensor((1, 100, 32, 128), dtype="float16") = R.reshape(  # type: ignore
                    lv, R.shape([1, 100, 32, 128])
                )
                gv: R.Tensor((1, 100, 32, 128), dtype="float16") = reshape1  # type: ignore
                R.output(gv)
            return gv
    # fmt: on

    class PagedKVCacheTest(modules.Module):
        def forward(
            self,
            cache: PagedKVCache,
            qkv: core.Tensor,
        ) -> core.Tensor:
            return cache.attention_with_fused_qkv(0, qkv, num_qo_heads=32, sm_scale=128**-0.5)

        def create_paged_kv_cache(
            self,
            max_batch_size: tir.Var,
            max_total_seq_len: tir.Var,
            prefill_chunk_size: tir.Var,
            page_size: tir.Var,
            support_sliding_window: tir.Var,
        ) -> PagedKVCache:
            return PagedKVCache.create_generic(
                attn_kind="mha",
                max_batch_size=max_batch_size,
                max_total_seq_len=max_total_seq_len,
                prefill_chunk_size=prefill_chunk_size,
                page_size=page_size,
                support_sliding_window=support_sliding_window,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                qk_head_dim=128,
                v_head_dim=128,
                rope_mode=RopeMode.NORMAL,
                rope_scale=1,
                rope_theta=10000,
                rotary_dim=128,
                dtype="float16",
            )

    export_results = PagedKVCacheTest().export_tvm(
        spec={
            "forward": {
                "cache": spec.Object(object_type=PagedKVCache),
                "qkv": spec.Tensor((1, 100, 96, 128), "float16"),
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
            },
        },
    )
    tvm_mod = export_results[0]
    tvm.ir.assert_structural_equal(tvm_mod, Module, True)


if __name__ == "__main__":
    test_nn_module_paged_kv_cache()
