"""A pass that rewrites KV cache creation functions in IRModule."""

from typing import Any, Dict

import tvm
from tvm import IRModule, relax

from mlc_chat.nn import RopeMode, kv_cache


def extract_creation_args(func: relax.Function) -> Dict[str, Any]:
    """Extract the KV cache creation args from the given generic creation func."""
    assert isinstance(func.body, relax.SeqExpr)
    assert len(func.body.blocks) == 1
    assert isinstance(func.body.blocks[0], relax.DataflowBlock)
    assert len(func.body.blocks[0].bindings) == 2
    assert isinstance(func.body.blocks[0].bindings[0], relax.VarBinding)
    assert isinstance(func.body.blocks[0].bindings[0].value, relax.Call)
    assert isinstance(func.body.blocks[0].bindings[0].value.op, relax.ExternFunc)
    assert (
        func.body.blocks[0].bindings[0].value.op.global_symbol
        == "mlc.create_paged_kv_cache_generic"
    )

    args = func.body.blocks[0].bindings[0].value.args
    assert len(args) == 10
    assert isinstance(args[0], relax.ShapeExpr)
    assert len(args[0].values) == 4
    for i in range(1, 9):
        assert isinstance(args[i], relax.PrimValue)
        assert isinstance(args[i].value, (tvm.tir.IntImm, tvm.tir.FloatImm))
    assert isinstance(args[9], relax.DataTypeImm)

    return {
        "max_batch_size": args[0].values[0],
        "max_total_seq_len": args[0].values[1],
        "prefill_chunk_size": args[0].values[2],
        "page_size": args[0].values[3],
        "num_hidden_layers": args[1].value.value,
        "num_attention_heads": args[2].value.value,
        "num_key_value_heads": args[3].value.value,
        "head_dim": args[4].value.value,
        "rope_mode": args[5].value.value,
        "rope_scale": args[6].value.value,
        "rope_theta": args[7].value.value,
        "rotary_dim": args[8].value.value,
        "dtype": args[9].value,
    }


@tvm.transform.module_pass(opt_level=0, name="RewriteKVCacheCreation")
class RewriteKVCacheCreation:  # pylint: disable=too-many-instance-attributes
    """Rewrite KV cache creation functions to IRModule."""

    def __init__(self, target: tvm.target.Target, flashinfer: bool) -> None:
        """Initializer.

        Parameters
        ----------
        target : tvm.target.Target
            The target of the model compilation.

        flashinfer : bool
            A boolean indicating if flashinfer is enabled.
        """
        self.target = target
        self.flashinfer = flashinfer

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        func_dict = {}
        creation_func = None
        for g_var, func in mod.functions_items():
            # Try to find the `create_paged_kv_cache` func.
            if g_var.name_hint == "create_paged_kv_cache":
                creation_func = func
            else:
                func_dict[g_var] = func

        if creation_func is None:
            return mod

        new_mod = IRModule(func_dict)
        if mod.attrs is not None:
            new_mod = new_mod.with_attrs(mod.attrs)

        kwargs = extract_creation_args(creation_func)

        bb = relax.BlockBuilder(new_mod)
        self.create_tir_paged_kv_cache(bb, kwargs)
        self.create_flashinfer_paged_kv_cache(bb, kwargs)
        return bb.finalize()

    def create_tir_paged_kv_cache(self, bb: relax.BlockBuilder, kwargs: Dict[str, Any]) -> None:
        """Create the TIR-based PagedKVCache"""
        max_batch_size = relax.Var(
            "max_batch_size_", relax.ShapeStructInfo([kwargs["max_batch_size"]])
        )
        max_total_seq_len = relax.Var(
            "max_total_seq_len_", relax.ShapeStructInfo([kwargs["max_total_seq_len"]])
        )
        prefill_chunk_size = relax.Var(
            "prefill_chunk_size_", relax.ShapeStructInfo([kwargs["prefill_chunk_size"]])
        )
        page_size = relax.Var("page_size_", relax.ShapeStructInfo([kwargs["page_size"]]))

        with bb.function(
            name="create_tir_paged_kv_cache",
            params=[max_batch_size, max_total_seq_len, prefill_chunk_size, page_size],
        ):
            cache = kv_cache.TIRPagedKVCache(target=self.target, **kwargs)
            bb.emit_func_output(cache._expr)  # pylint: disable=protected-access

    def create_flashinfer_paged_kv_cache(
        self, bb: relax.BlockBuilder, kwargs: Dict[str, Any]
    ) -> None:
        """Create the FlashInfer-based PagedKVCache"""
        # Filter the cases which FlashInfer does not support.
        if (
            not self.flashinfer
            or str(kwargs["dtype"]) != "float16"
            or kwargs["head_dim"] != 128
            or (
                kwargs["rope_mode"] == RopeMode.INLINE
                and kwargs["rotary_dim"] != kwargs["head_dim"]
            )
        ):
            return

        max_batch_size = relax.Var(
            "max_batch_size_", relax.ShapeStructInfo([kwargs["max_batch_size"]])
        )
        max_total_seq_len = relax.Var(
            "max_total_seq_len_", relax.ShapeStructInfo([kwargs["max_total_seq_len"]])
        )
        prefill_chunk_size = relax.Var(
            "prefill_chunk_size_", relax.ShapeStructInfo([kwargs["prefill_chunk_size"]])
        )
        page_size = relax.Var("page_size_", relax.ShapeStructInfo([kwargs["page_size"]]))

        with bb.function(
            name="create_flashinfer_paged_kv_cache",
            params=[max_batch_size, max_total_seq_len, prefill_chunk_size, page_size],
        ):
            cache = kv_cache.FlashInferPagedKVCache(target=self.target, **kwargs)
            bb.emit_func_output(cache._expr)  # pylint: disable=protected-access
