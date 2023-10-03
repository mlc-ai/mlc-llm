import tvm
from tvm import IRModule, relax, tir
from tvm.relax.dpl.pattern import GlobalVarPattern, TuplePattern, is_op, wildcard


def check_decoding(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["w"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    return gv.name_hint.startswith("decode") or gv.name_hint.startswith("fused_decode")


def check_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["matmul"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    return (
        gv.name_hint.startswith("matmul")
        or gv.name_hint.startswith("fused_matmul")
        or gv.name_hint.startswith("NT_matmul")
        or gv.name_hint.startswith("fused_NT_matmul")
    )


def pattern_check():
    def f_pattern_check(ctx: relax.transform.PatternCheckContext) -> bool:
        return check_decoding(ctx) and check_matmul(ctx)

    return f_pattern_check


def decode_matmul_pattern(match_ewise: int, n_aux_tensor: int):
    assert n_aux_tensor == 1 or n_aux_tensor == 2 or n_aux_tensor == 3 or n_aux_tensor == 4

    w_scaled = wildcard()
    aux_tensors = [wildcard(), wildcard(), wildcard(), wildcard()]
    x = wildcard()
    w = is_op("relax.call_tir")(
        GlobalVarPattern(),
        TuplePattern([w_scaled, *aux_tensors[0:n_aux_tensor]]),
        add_constraint=False,
    )
    matmul_args = [x, w]
    for _ in range(match_ewise):
        matmul_args.append(wildcard())
    matmul = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern(matmul_args), add_constraint=False
    )

    annotations = {
        "matmul": matmul,
        "w": w,
        "x": x,
        "w_scaled": w_scaled,
    }
    return matmul, annotations, pattern_check()


@tvm.transform.module_pass(opt_level=0, name="FuseDecodeMatmulEwise")
class FuseDecodeMatmulEwise:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext  # pylint: disable=unused-argument
    ) -> IRModule:
        for n_aux_tensor in [1, 2, 3, 4]:
            for match_ewise in [0, 1, 2, 6]:
                if match_ewise == 6 and n_aux_tensor != 4:
                    continue
                mod = relax.transform.FuseOpsByPattern(
                    [
                        (
                            "decode_matmul",
                            *decode_matmul_pattern(match_ewise, n_aux_tensor),
                        )
                    ]
                )(mod)
        mod = relax.transform.FuseTIR()(mod)

        return mod
