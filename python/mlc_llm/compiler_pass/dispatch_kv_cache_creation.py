"""A pass that rewrites KV cache creation functions in IRModule."""

import json
from typing import Any, Dict, List, Optional

import tvm
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.frontend.nn.llm import kv_cache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode

from mlc_llm.support import logging

logger = logging.getLogger(__name__)

_OP_CALL_DPS_PACKED = tvm.ir.Op.get("relax.call_dps_packed")
_OP_CALL_PURE_PACKED = tvm.ir.Op.get("relax.call_pure_packed")
_ATTN_WITH_FUSED_QKV = "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
_ATTN_SELF = "vm.builtin.attention_kv_cache_self_attention"
_ATTN_CROSS = "vm.builtin.attention_kv_cache_cross_attention"
_APPEND_MLA_KV = "vm.builtin.attention_kv_cache_append_mla_kv"


@mutator
class _KVCacheCallDTypeRewriter(PyExprMutator):  # pylint: disable=abstract-method
    """Rewrite KV cache runtime calls to adapt tensor dtypes to KV cache dtype."""

    def __init__(self, mod: IRModule, kv_cache_dtype: str) -> None:
        super().__init__(mod)
        self.mod = mod
        self.kv_cache_dtype = kv_cache_dtype

    def transform(self) -> IRModule:
        """Entry point."""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                self.builder_.update_func(g_var, self.visit_expr(func))
        mod = self.builder_.finalize()
        if self.mod.attrs is not None:
            mod = mod.with_attrs(self.mod.attrs)
        return mod

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)

        if call.op == _OP_CALL_DPS_PACKED and isinstance(call.args[0], relax.ExternFunc):
            global_symbol = str(call.args[0].global_symbol)
            if global_symbol == _ATTN_WITH_FUSED_QKV:
                return self._rewrite_attention_with_fused_qkv(call, global_symbol)
            if global_symbol in [_ATTN_SELF, _ATTN_CROSS]:
                return self._rewrite_self_or_cross_attention(call, global_symbol)

        if call.op == _OP_CALL_PURE_PACKED and isinstance(call.args[0], relax.ExternFunc):
            global_symbol = str(call.args[0].global_symbol)
            if global_symbol == _APPEND_MLA_KV:
                return self._rewrite_append_mla_kv(call, global_symbol)

        return call

    def _rewrite_attention_with_fused_qkv(self, call: relax.Call, global_symbol: str) -> relax.Expr:
        if len(call.args) < 2 or not isinstance(call.args[1], relax.Tuple):
            return call
        packed_args = list(call.args[1].fields)
        if len(packed_args) != 4:
            return call
        qkv = packed_args[3]
        qkv_sinfo = qkv.struct_info
        if not isinstance(qkv_sinfo, relax.TensorStructInfo):
            return call
        if str(qkv_sinfo.dtype) == self.kv_cache_dtype:
            return call
        out_sinfo = call.struct_info
        if not isinstance(out_sinfo, relax.TensorStructInfo):
            return call

        packed_args[3] = relax.op.astype(qkv, self.kv_cache_dtype)
        kv_out_sinfo = relax.TensorStructInfo(out_sinfo.shape, self.kv_cache_dtype)
        rewritten_call = relax.call_dps_packed(global_symbol, packed_args, out_sinfo=kv_out_sinfo)
        return relax.op.astype(rewritten_call, str(qkv_sinfo.dtype))

    def _rewrite_self_or_cross_attention(  # pylint: disable=too-many-return-statements
        self, call: relax.Call, global_symbol: str
    ) -> relax.Expr:
        if len(call.args) < 2 or not isinstance(call.args[1], relax.Tuple):
            return call
        packed_args = list(call.args[1].fields)
        if len(packed_args) < 4:
            return call
        q_data = packed_args[3]
        q_sinfo = q_data.struct_info
        if not isinstance(q_sinfo, relax.TensorStructInfo):
            return call
        if str(q_sinfo.dtype) == self.kv_cache_dtype:
            return call
        out_sinfo = call.struct_info
        if not isinstance(out_sinfo, relax.TupleStructInfo):
            return call
        if len(out_sinfo.fields) != 2 or not isinstance(
            out_sinfo.fields[0], relax.TensorStructInfo
        ):
            return call

        packed_args[3] = relax.op.astype(q_data, self.kv_cache_dtype)
        kv_o_sinfo = relax.TensorStructInfo(out_sinfo.fields[0].shape, self.kv_cache_dtype)
        rewritten_call = relax.call_dps_packed(
            global_symbol,
            packed_args,
            out_sinfo=[kv_o_sinfo, out_sinfo.fields[1]],
        )
        return relax.Tuple(
            [
                relax.op.astype(relax.TupleGetItem(rewritten_call, 0), str(q_sinfo.dtype)),
                relax.TupleGetItem(rewritten_call, 1),
            ]
        )

    def _rewrite_append_mla_kv(self, call: relax.Call, global_symbol: str) -> relax.Expr:
        if len(call.args) < 4:
            return call
        kv_data = call.args[3]
        kv_sinfo = kv_data.struct_info
        if not isinstance(kv_sinfo, relax.TensorStructInfo):
            return call
        if str(kv_sinfo.dtype) == self.kv_cache_dtype:
            return call
        updated_args = list(call.args)
        updated_args[3] = relax.op.astype(kv_data, self.kv_cache_dtype)
        sinfo_args = list(call.sinfo_args)
        return relax.call_pure_packed(
            global_symbol,
            *updated_args[1:],
            sinfo_args=sinfo_args[0] if len(sinfo_args) == 1 else sinfo_args,
        )


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
    args = call_args[1:]
    assert len(args) == 18
    assert isinstance(args[0], (relax.StringImm, relax.Tuple))
    # Check if attn_kind is a single value or a list with length of hidden layers
    if isinstance(args[0], relax.StringImm):
        assert args[0].value in ["mha", "mla"]
        attn_kind = args[0].value
    else:
        assert len(args[0].fields) == args[3].value.value
        for i, attention_type in enumerate(args[0].fields):
            assert isinstance(attention_type, relax.StringImm)
            assert attention_type.value in ["mha", "mla", "mha_sliding"]
        attn_kind = [args[0].fields[i].value for i in range(len(args[0]))]
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
        "attn_kind": attn_kind,
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

    def _requested_kv_cache_dtype(self) -> Optional[str]:
        dtype = self.metadata.get("kv_cache_dtype")
        if dtype in [None, "", "auto"]:
            return None
        if not isinstance(dtype, str):
            dtype = str(dtype)
        return dtype

    def _apply_kv_cache_dtype_override(self, kwargs: Dict[str, Any]) -> None:
        requested_dtype = self._requested_kv_cache_dtype()
        if requested_dtype is None:
            return
        logger.info(
            "Overriding KV cache dtype from %s to %s",
            kwargs["dtype"],
            requested_dtype,
        )
        kwargs["dtype"] = requested_dtype

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
        self._apply_kv_cache_dtype_override(kwargs)
        self.attach_kv_cache_metadata(kwargs)

        bb = relax.BlockBuilder(new_mod)
        extern_mods = []
        extern_mods += self.create_tir_paged_kv_cache(bb, kwargs)
        extern_mods += self.create_flashinfer_paged_kv_cache(bb, kwargs)

        mod = bb.finalize()
        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        mod = mod.with_attr("external_mods", mod_attrs.get("external_mods", []) + extern_mods)
        requested_kv_cache_dtype = self._requested_kv_cache_dtype()
        if requested_kv_cache_dtype is not None:
            mod = _KVCacheCallDTypeRewriter(mod, requested_kv_cache_dtype).transform()
        return mod

    def attach_kv_cache_metadata(self, kwargs: Dict[str, Any]):
        """Attach the KV cache metadata to model metadata."""
        self.metadata["kv_cache"] = {
            "num_hidden_layers": kwargs["num_hidden_layers"],
            "num_attention_heads": kwargs["num_attention_heads"],
            "num_key_value_heads": kwargs["num_key_value_heads"],
            "head_dim": kwargs["qk_head_dim"],
            "dtype": str(kwargs["dtype"]),
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
            "support_sliding_window_",
            relax.ShapeStructInfo([kwargs["support_sliding_window"]]),
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
            "support_sliding_window_",
            relax.ShapeStructInfo([kwargs["support_sliding_window"]]),
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
