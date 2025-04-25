"""A pass that rewrites KV cache creation functions in IRModule."""

import json
from typing import Any, Dict, List

import tvm
from tvm import IRModule, relax
from tvm.relax.frontend.nn.llm import kv_cache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode

from mlc_llm.support import logging

logger = logging.getLogger(__name__)


def extract_creation_args(func: relax.Function) -> Dict[str, Any]:
    """Extract the KV cache creation args from the given generic creation func."""
    assert isinstance(func.body, relax.SeqExpr)
    assert len(func.body.blocks) == 1
    assert isinstance(func.body.blocks[0], relax.DataflowBlock)
    assert isinstance(func.body.blocks[0].bindings[0], relax.VarBinding)
    assert isinstance(func.body.blocks[0].bindings[0].value, relax.Call)
    assert func.body.blocks[0].bindings[0].value.op == tvm.ir.Op.get("relax.call_pure_packed")
    call_args = func.body.blocks[0].bindings[0].value.args
    assert isinstance(call_args[0], relax.ExternFunc)
    assert call_args[0].global_symbol == "mlc.create_paged_kv_cache_generic"
    assert isinstance(call_args[1], relax.StringImm)

    args = call_args[1:]
    assert len(args) == 18
    assert isinstance(args[0], relax.StringImm)
    assert args[0].value in ["mha", "mla"]
    assert isinstance(args[1], relax.ShapeExpr)
    assert len(args[1].values) == 5
    assert isinstance(args[2], relax.ShapeExpr)
    for i in range(3, 18):
        if i in [13, 14, 17]:
            continue
        assert isinstance(args[i], relax.PrimValue), f"args[{i}] is {type(args[i])}"
        assert isinstance(args[i].value, (tvm.tir.IntImm, tvm.tir.FloatImm))
    assert isinstance(args[13], relax.StringImm)
    assert isinstance(args[16], (relax.Constant, relax.PrimValue))
    assert isinstance(args[17], relax.DataTypeImm)

    return {
        "attn_kind": args[0].value,
        "max_batch_size": args[1].values[0],
        "max_total_seq_len": args[1].values[1],
        "prefill_chunk_size": args[1].values[2],
        "page_size": args[1].values[3],
        "support_sliding_window": args[1].values[4],
        "layer_partition": args[2],
        "num_hidden_layers": args[3].value.value,
        "num_attention_heads": args[4].value.value,
        "num_key_value_heads": args[5].value.value,
        "qk_head_dim": args[6].value.value,
        "v_head_dim": args[7].value.value,
        "mla_original_qk_head_dim": args[8].value.value,
        "mla_original_v_head_dim": args[9].value.value,
        "rope_mode": args[10].value.value,
        "rope_scale": args[11].value.value,
        "rope_theta": args[12].value.value,
        "rope_scaling": json.loads(args[13].value),
        "rope_ext_factors": args[14],
        "rotary_dim": args[15].value.value,
        "enable_disaggregation": bool(args[16].value.value),
        "dtype": args[17].value,
    }


@tvm.transform.module_pass(opt_level=0, name="DispatchKVCacheCreation")
class DispatchKVCacheCreation:  # pylint: disable=too-many-instance-attributes
    """Rewrite KV cache creation functions to IRModule."""

    def __init__(
        self, target: tvm.target.Target, flashinfer: bool, metadata: Dict[str, Any]
    ) -> None:
        """Initializer.

        Parameters
        ----------
        target : tvm.target.Target
            The target of the model compilation.

        flashinfer : bool
            A boolean indicating if flashinfer is enabled.

        metadata : Dict[str, Any]
            The model's metadata for KV cache creation.
            Note that the metadata will be updated in this pass -- the
            KV cache metadata will be attached.
        """
        self.target = target
        self.flashinfer = flashinfer
        self.metadata = metadata

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
        self.attach_kv_cache_metadata(kwargs)

        bb = relax.BlockBuilder(new_mod)
        extern_mods = []
        extern_mods += self.create_tir_paged_kv_cache(bb, kwargs)
        extern_mods += self.create_flashinfer_paged_kv_cache(bb, kwargs)

        mod = bb.finalize()
        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        mod = mod.with_attr("external_mods", mod_attrs.get("external_mods", []) + extern_mods)
        return mod

    def attach_kv_cache_metadata(self, kwargs: Dict[str, Any]):
        """Attach the KV cache metadata to model metadata."""
        self.metadata["kv_cache"] = {
            "num_hidden_layers": kwargs["num_hidden_layers"],
            "num_attention_heads": kwargs["num_attention_heads"],
            "num_key_value_heads": kwargs["num_key_value_heads"],
            "head_dim": kwargs["qk_head_dim"],
        }

    def create_tir_paged_kv_cache(
        self, bb: relax.BlockBuilder, kwargs: Dict[str, Any]
    ) -> List[tvm.runtime.Module]:
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
        support_sliding_window = relax.Var(
            "support_sliding_window_", relax.ShapeStructInfo([kwargs["support_sliding_window"]])
        )

        # Ensure 'enable_disaggregation' is optional
        enable_disaggregation = kwargs.pop("enable_disaggregation", False)
        kwargs["enable_disaggregation"] = enable_disaggregation

        with bb.function(
            name="create_tir_paged_kv_cache",
            params=[
                max_batch_size,
                max_total_seq_len,
                prefill_chunk_size,
                page_size,
                support_sliding_window,
            ],
        ):
            cache = kv_cache.TIRPagedKVCache(target=self.target, **kwargs)
            bb.emit_func_output(cache._expr)  # pylint: disable=protected-access

        return cache.extern_mods

    def create_flashinfer_paged_kv_cache(
        self, bb: relax.BlockBuilder, kwargs: Dict[str, Any]
    ) -> List[tvm.runtime.Module]:
        """Create the FlashInfer-based PagedKVCache"""
        # Filter the cases which FlashInfer does not support.
        if (  # pylint: disable=too-many-boolean-expressions
            not self.flashinfer
            or self.target.kind.name != "cuda"
            or str(kwargs["dtype"]) not in ["float16", "bfloat16"]
            or (
                kwargs["rope_mode"] == RopeMode.INLINE
                and (
                    kwargs["rotary_dim"] != kwargs["qk_head_dim"]
                    or kwargs["qk_head_dim"] != kwargs["v_head_dim"]
                )
            )
        ):
            return []

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
        support_sliding_window = relax.Var(
            "support_sliding_window_", relax.ShapeStructInfo([kwargs["support_sliding_window"]])
        )

        try:
            with bb.function(
                name="create_flashinfer_paged_kv_cache",
                params=[
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                ],
            ):
                cache = kv_cache.FlashInferPagedKVCache(target=self.target, **kwargs)
                bb.emit_func_output(cache._expr)  # pylint: disable=protected-access
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.info(
                "Error caught when creating FlashInfer PagedKVCache: %s\n"
                "The model will fallback to TIR-based KV cache.",
                e,
            )
            return []

        return cache.extern_mods
