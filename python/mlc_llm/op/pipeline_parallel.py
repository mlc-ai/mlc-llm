"""Operators for pipeline parallelism."""

from typing import List

from tvm import relax
from tvm.relax.frontend.nn import Tensor, op


def pipeline_stage_boundary(*tensors: Tensor) -> List[Tensor]:
    """Pipeline parallelism stage boundary mark operator in MLC.

    Parameters
    ----------
    tensors : Tensor
        The tensors to be passed to the next stage.

    Returns
    -------
    tensors : List[Tensor]
        The list of input tensors passed to the next stage.
    """
    # pylint: disable=protected-access
    return op.wrap_nested(
        relax.call_pure_packed(
            "mlc.pipeline_parallel_stage_boundary",
            *[tensor._expr for tensor in tensors],
            sinfo_args=(
                tensors[0]._expr.struct_info
                if len(tensors) == 1
                else relax.TupleStructInfo([tensor._expr.struct_info for tensor in tensors])
            )
        ),
        name="pipeline_stage_boundary",
    )
    # pylint: enable=protected-access
