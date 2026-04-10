"""Utilities shared across model definitions."""

from tvm import te
from tvm.relax.frontend.nn import Tensor, op


def index_last_token(x: Tensor) -> Tensor:
    """Select the last token while preserving the historical `index` TE op shape/name."""

    def _index(x: te.Tensor):
        b, s, d = x.shape
        return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

    return op.tensor_expr_op(_index, name_hint="index", args=[x])
