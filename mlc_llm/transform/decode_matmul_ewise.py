import tvm
from tvm import IRModule, relax, tir
from tvm.relax.dpl.pattern import GlobalVarPattern, TuplePattern, is_op, wildcard


def check_x_1dim(ctx: relax.transform.PatternCheckContext) -> bool:
    x = ctx.annotated_expr["x"]
    if len(x.struct_info.shape) == 1:
        return True
    n = x.struct_info.shape[-2]
    return isinstance(n, tir.IntImm) and n.value == 1


def check_decoding(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["w"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    return gv.name_hint.startswith("decode")


def check_matmul(ctx: relax.transform.PatternCheckContext, target_kind: str) -> bool:
    call = ctx.annotated_expr["matmul"]
    if not isinstance(call, relax.Call):
        return False
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return False
    is_matmul = gv.name_hint.startswith("matmul") or gv.name_hint.startswith(
        "fused_matmul"
    )
    is_NT_matmul = gv.name_hint.startswith("NT_matmul") or gv.name_hint.startswith(
        "fused_NT_matmul"
    )
    return (is_matmul or is_NT_matmul) if target_kind == "android" else is_matmul


def pattern_check(target_kind: str):
    def f_pattern_check(ctx: relax.transform.PatternCheckContext) -> bool:
        if target_kind != "android" and not check_x_1dim(ctx):
            return False
        return check_decoding(ctx) and check_matmul(ctx, target_kind)

    return f_pattern_check


def decode_matmul_pattern(match_ewise: int, n_aux_tensor: int, target_kind: str):
    assert n_aux_tensor == 1 or n_aux_tensor == 2 or n_aux_tensor == 4

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

    return matmul, annotations, pattern_check(target_kind)


@tvm.transform.module_pass(opt_level=0, name="FuseDecodeMatmulEwise")
class FuseDecodeMatmulEwise:
    def __init__(self, dtype: str, target_kind: str) -> None:
        self.dtype = dtype
        self.target_kind = target_kind

    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        for n_aux_tensor in [1, 2, 4]:
            for match_ewise in [0, 1, 2, 6]:
                if match_ewise == 6 and n_aux_tensor != 4:
                    continue
                mod = relax.transform.FuseOpsByPattern(
                    [
                        (
                            "decode_matmul",
                            *decode_matmul_pattern(
                                match_ewise, n_aux_tensor, self.target_kind
                            ),
                        )
                    ]
                )(mod)
        mod = relax.transform.FuseTIR()(mod)

        return mod
