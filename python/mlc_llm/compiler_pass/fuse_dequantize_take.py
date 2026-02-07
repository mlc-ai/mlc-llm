"""A compiler pass that fuses dequantize + take."""

import tvm
from tvm import IRModule, relax, tir
from tvm.relax.dpl.pattern import (
    GlobalVarPattern,
    TuplePattern,
    is_const,
    is_op,
    wildcard,
)


@tvm.transform.module_pass(opt_level=0, name="FuseDequantizeTake")
class FuseDequantizeTake:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses dequantize + take."""

    def transform_module(  # pylint: disable=too-many-locals
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        seq = []
        for n_aux_tensor in [2, 3]:
            for match_tir_vars in [False, True]:
                seq.append(
                    relax.transform.FuseOpsByPattern(
                        [
                            (
                                "dequantize_take",
                                *_pattern(n_aux_tensor, match_tir_vars),
                            )
                        ]
                    )
                )
        seq.append(relax.transform.FuseTIR())
        mod = tvm.transform.Sequential(seq)(mod)
        for g_var, func in mod.functions_items():
            name = g_var.name_hint
            if isinstance(func, tir.PrimFunc) and (
                ("fused_dequantize" in name) and ("take" in name)
            ):
                sch_mod = tvm.IRModule({"main": func})
                sch_mod = tir.transform.ForceNarrowIndexToInt32()(sch_mod)
                sch = tir.Schedule(sch_mod)
                sch.compute_inline("dequantize")
                mod[g_var] = sch.mod["main"]
        return mod


def _pattern(n_aux_tensor: int, match_tir_vars: bool):
    dequantize = is_op("relax.call_tir")(
        GlobalVarPattern(),
        TuplePattern([wildcard() for _ in range(n_aux_tensor)]),
        add_constraint=False,
    )
    indices = ~is_const()
    if match_tir_vars:
        call_tir_args_take = [
            GlobalVarPattern(),
            TuplePattern([dequantize, indices]),
            wildcard(),
        ]
    else:
        call_tir_args_take = [
            GlobalVarPattern(),
            TuplePattern([dequantize, indices]),
        ]
    take = is_op("relax.call_tir")(
        *call_tir_args_take,
        add_constraint=False,
    )
    annotations = {
        "take": take,
        "dequantize": dequantize,
        "indices": indices,
    }

    def _check(ctx: relax.transform.PatternCheckContext) -> bool:
        take = ctx.annotated_expr["take"]
        dequantize = ctx.annotated_expr["dequantize"]
        if not isinstance(dequantize, relax.expr.Call):
            return False
        if not isinstance(take.args[0], relax.GlobalVar) or not isinstance(
            dequantize.args[0], relax.GlobalVar
        ):
            return False
        return "take" in take.args[0].name_hint and "dequantize" in dequantize.args[0].name_hint

    return take, annotations, _check
