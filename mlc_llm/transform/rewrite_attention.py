import tvm
from tvm.relax.dpl import PatternContext, is_const, is_op, rewrite_call, wildcard
from tvm.script import relax as R


def rewrite_attention(use_flash_mqa=False):
    @tvm.ir.transform.module_pass(opt_level=0, name="mlc_llm.transform.rewrite_attention")
    def ir_module_transform(mod: tvm.IRModule, context) -> tvm.IRModule:
        Q = wildcard()
        K = wildcard()
        V = wildcard()

        Q_BNSH = is_op("relax.permute_dims")(Q)

        if use_flash_mqa:
            K_BNSH = is_op("relax.permute_dims")(is_op("relax.repeat")(K))
            V_BNSH = is_op("relax.permute_dims")(is_op("relax.repeat")(V))
        else:
            K_BNSH = is_op("relax.permute_dims")(K)
            V_BNSH = is_op("relax.permute_dims")(V)

        K_BNSH_T = is_op("relax.permute_dims")(K_BNSH)

        matmul1 = is_op("relax.matmul")(Q_BNSH, K_BNSH_T)
        divide = is_op("relax.divide")(matmul1, is_const())
        max = is_op("relax.maximum")(divide, is_const())
        min = is_op("relax.minimum")(max, wildcard())
        softmax = is_op("relax.nn.softmax")(is_op("relax.astype")(min))
        matmul2 = is_op("relax.matmul")(is_op("relax.astype")(softmax), V_BNSH)

        pattern = is_op("relax.permute_dims")(matmul2)

        def callback(_, matchings):
            return R.nn.attention(
                matchings[Q], matchings[K], matchings[V], causal_mask="BottomRight"
            )

        new_module = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, tvm.relax.Function):
                func = rewrite_call(pattern, callback, func)
            new_module[gvar] = func

        return tvm.IRModule(new_module, mod.type_definitions, mod.attrs, mod.global_infos)

    return ir_module_transform
