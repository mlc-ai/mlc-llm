# pylint: disable=missing-docstring
from typing import Callable

from tvm import tir

from .sch_inline import auto_inline_consumers, auto_inline_producers


def sch_norm(
    name_norm: str,
    len_tx: int = 256,
    unroll_depth: int = 256,
):
    def sch_func(sch: tir.Schedule) -> None:  # pylint: disable=too-many-locals
        b_reduce = sch.get_block(name_norm)
        (b_spatial,) = sch.get_consumers(b_reduce)
        loops = sch.get_loops(b_spatial)
        bx = sch.fuse(*loops[:-1])
        _, tx = sch.split(loops[-1], [None, len_tx])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for i, _ in enumerate(sch.get(b_reduce).writes):
            sch.set_scope(b_reduce, buffer_index=i, storage_scope="shared")
        sch.compute_at(b_reduce, bx, preserve_unit_loops=True)
        _, s, *_, tx = sch.get_loops(b_reduce)
        _, tx = sch.split(tx, [None, len_tx])
        sch.bind(tx, "threadIdx.x")
        auto_inline_consumers(sch, b_spatial)
        sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)

    return sch_func
