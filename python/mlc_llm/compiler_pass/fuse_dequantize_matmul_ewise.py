"""A compiler pass that fuses dequantize + matmul + elementwise."""

import tvm
from tvm import IRModule, relax
from tvm.relax.dpl.pattern import GlobalVarPattern, TuplePattern, is_op, wildcard


@tvm.transform.module_pass(opt_level=0, name="FuseDequantizeMatmulEwise")
class FuseDequantizeMatmulEwise:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses dequantize + matmul + elementwise."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        seq = []
        for n_aux_tensor in [0, 1, 2, 3, 4]:
            for match_ewise in [0, 1, 2, 3, 6]:
                if match_ewise == 6 and n_aux_tensor != 4:
                    continue
                seq.append(
                    relax.transform.FuseOpsByPattern(
                        [
                            (
                                "dequantize_matmul",
                                *_pattern(match_ewise, n_aux_tensor),
                            )
                        ]
                    )
                )
        seq.append(relax.transform.FuseTIR())
        return tvm.transform.Sequential(seq)(mod)


def _pattern(match_ewise: int, n_aux_tensor: int):
    # pylint: disable=invalid-name
    w_scaled = wildcard()
    x = wildcard()
    w = is_op("relax.call_tir")(
        GlobalVarPattern(),
        TuplePattern([w_scaled] + [wildcard() for _ in range(n_aux_tensor)]),
        add_constraint=False,
    )
    matmul = is_op("relax.call_tir")(
        GlobalVarPattern(),
        TuplePattern([x, w] + [wildcard() for _ in range(match_ewise)]),
        add_constraint=False,
    )
    # pylint: enable=invalid-name
    annotations = {
        "w_scaled": w_scaled,
        "x": x,
        "w": w,
        "matmul": matmul,
    }

    def _check_decoding(ctx: relax.transform.PatternCheckContext) -> bool:
        call = ctx.annotated_expr["w"]
        if not isinstance(call, relax.Call):
            return False
        g_var = call.args[0]
        if not isinstance(g_var, relax.GlobalVar):
            return False
        return g_var.name_hint.startswith("dequantize") or g_var.name_hint.startswith(
            "fused_dequantize"
        )

    def _check_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
        call = ctx.annotated_expr["matmul"]
        if not isinstance(call, relax.Call):
            return False
        g_var = call.args[0]
        if not isinstance(g_var, relax.GlobalVar):
            return False
        return (
            g_var.name_hint.startswith("matmul")
            or g_var.name_hint.startswith("fused_matmul")
            or g_var.name_hint.startswith("NT_matmul")
            or g_var.name_hint.startswith("fused_NT_matmul")
        )

    def _check(ctx: relax.transform.PatternCheckContext) -> bool:
        return _check_decoding(ctx) and _check_matmul(ctx)

    return matmul, annotations, _check
