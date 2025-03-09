"""A pass that rewrites KV cache creation functions in IRModule."""

import json
from typing import Any, Dict, Literal, Tuple

import tvm
from tvm import IRModule, relax
from tvm.relax.frontend.nn.llm import kv_cache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode


def extract_creation_args(func: relax.Function) -> Tuple[Literal["mha", "mla"], Dict[str, Any]]:
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
    if args[0].value == "mha":
        assert len(args) == 15
        assert isinstance(args[1], relax.ShapeExpr)
        assert len(args[1].values) == 5
        assert isinstance(args[2], relax.ShapeExpr)
        for i in range(3, 14):
            if i in [10, 11]:
                continue
            assert isinstance(args[i], relax.PrimValue)
            assert isinstance(args[i].value, (tvm.tir.IntImm, tvm.tir.FloatImm))
        assert isinstance(args[10], relax.StringImm)
        assert isinstance(args[11], (relax.Constant, relax.PrimValue))
        assert isinstance(args[14], relax.DataTypeImm)

        return "mha", {
            "max_batch_size": args[1].values[0],
            "max_total_seq_len": args[1].values[1],
            "prefill_chunk_size": args[1].values[2],
            "page_size": args[1].values[3],
            "support_sliding_window": args[1].values[4],
            "layer_partition": args[2],
            "num_hidden_layers": args[3].value.value,
            "num_attention_heads": args[4].value.value,
            "num_key_value_heads": args[5].value.value,
            "head_dim": args[6].value.value,
            "rope_mode": args[7].value.value,
            "rope_scale": args[8].value.value,
            "rope_theta": args[9].value.value,
            "rope_scaling": json.loads(args[10].value),
            "rope_ext_factors": args[11],
            "rotary_dim": args[12].value.value,
            "enable_disaggregation": bool(args[13].value.value),
            "dtype": args[14].value,
        }
    if call_args[1].value == "mla":
        assert len(args) == 12
        assert isinstance(args[1], relax.ShapeExpr)
        assert len(args[1].values) == 5
        assert isinstance(args[2], relax.ShapeExpr)
        for i in range(3, 11):
            assert isinstance(args[i], relax.PrimValue)
            assert isinstance(args[i].value, tvm.tir.IntImm)
        assert isinstance(args[11], relax.DataTypeImm)

        return "mla", {
            "max_batch_size": args[1].values[0],
            "max_total_seq_len": args[1].values[1],
            "prefill_chunk_size": args[1].values[2],
            "page_size": args[1].values[3],
            "support_sliding_window": args[1].values[4],
            "layer_partition": args[2],
            "num_hidden_layers": args[3].value.value,
            "num_attention_heads": args[4].value.value,
            "num_key_value_heads": args[5].value.value,
            "qk_nope_head_dim": args[6].value.value,
            "qk_rope_head_dim": args[7].value.value,
            "v_head_dim": args[8].value.value,
            "kv_lora_rank": args[9].value.value,
            "enable_disaggregation": bool(args[10].value.value),
            "dtype": args[11].value,
        }

    raise ValueError("Cannot reach here")


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

        kv_cache_kind, kwargs = extract_creation_args(creation_func)
        self.attach_kv_cache_metadata(kv_cache_kind, kwargs)

        bb = relax.BlockBuilder(new_mod)
        self.create_tir_paged_kv_cache(bb, kv_cache_kind, kwargs)
        self.create_flashinfer_paged_kv_cache(bb, kv_cache_kind, kwargs)
        return bb.finalize()

    def attach_kv_cache_metadata(
        self, kv_cache_kind: Literal["mha", "mla"], kwargs: Dict[str, Any]
    ):
        """Attach the KV cache metadata to model metadata."""
        if kv_cache_kind == "mha":
            self.metadata["kv_cache"] = {
                "num_hidden_layers": kwargs["num_hidden_layers"],
                "num_attention_heads": kwargs["num_attention_heads"],
                "num_key_value_heads": kwargs["num_key_value_heads"],
                "head_dim": kwargs["head_dim"],
            }
        elif kv_cache_kind == "mla":
            self.metadata["kv_cache"] = {
                "num_hidden_layers": kwargs["num_hidden_layers"],
                "num_attention_heads": kwargs["num_attention_heads"],
                "num_key_value_heads": 1,
                "head_dim": kwargs["kv_lora_rank"] + kwargs["qk_rope_head_dim"],
            }
        else:
            raise ValueError("Cannot reach here.")

    def create_tir_paged_kv_cache(
        self, bb: relax.BlockBuilder, kv_cache_kind: Literal["mha", "mla"], kwargs: Dict[str, Any]
    ) -> None:
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

        # Remove the 'enable_disaggregation' argument
        kwargs.pop("enable_disaggregation", None)

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
            if kv_cache_kind == "mha":
                cache = kv_cache.TIRPagedKVCache(target=self.target, **kwargs)
            elif kv_cache_kind == "mla":
                cache = kv_cache.TIRPagedKVCache.create_mla_kv_cache(target=self.target, **kwargs)
            else:
                raise ValueError("Cannot reach here")
            bb.emit_func_output(cache._expr)  # pylint: disable=protected-access

    def create_flashinfer_paged_kv_cache(
        self, bb: relax.BlockBuilder, kv_cache_kind: Literal["mha", "mla"], kwargs: Dict[str, Any]
    ) -> None:
        """Create the FlashInfer-based PagedKVCache"""
        # Filter the cases which FlashInfer does not support.
        if (  # pylint: disable=too-many-boolean-expressions
            not self.flashinfer
            or kv_cache_kind != "mha"
            or str(kwargs["dtype"]) != "float16"
            or kwargs["head_dim"] != 128
            or (
                kwargs["rope_mode"] == RopeMode.INLINE
                and kwargs["rotary_dim"] != kwargs["head_dim"]
            )
            or (
                # bypass GPT-2 since it uses attn_score_scaling_factor
                "gpt2"
                in self.metadata["model_type"]
            )
            # filter by attention group size
            or kwargs["num_attention_heads"] // kwargs["num_key_value_heads"] not in [1, 4, 8]
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
        support_sliding_window = relax.Var(
            "support_sliding_window_", relax.ShapeStructInfo([kwargs["support_sliding_window"]])
        )

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
