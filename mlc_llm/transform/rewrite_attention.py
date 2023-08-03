from tvm.relax.dpl import PatternContext, is_const, is_op, rewrite_call, wildcard
from tvm.script import relax as R


def rewrite_attention(f):
    Q = wildcard()
    K = wildcard()
    V = wildcard()

    Q_BNSH = is_op("relax.permute_dims")(Q)
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

    return rewrite_call(pattern, callback, f)
