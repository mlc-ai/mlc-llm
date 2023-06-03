# pylint: disable=missing-docstring
from typing import Callable

from tvm import tir

from .sch_inline import auto_inline_consumers, auto_inline_producers


def sch_softmax(
    name_reduce_0: str = "T_softmax_maxelem",
    name_reduce_1: str = "T_softmax_expsum",
    name_spatial: str = "T_softmax_norm",
    len_tx: int = 256,
    unroll_depth: int = 256,
):
    def sch_func(sch: tir.Schedule) -> None:  # pylint: disable=too-many-locals
        b_reduce_0 = sch.get_block(name_reduce_0)
        b_reduce_1 = sch.get_block(name_reduce_1)
        b_spatial = sch.get_block(name_spatial)

        sch.compute_inline(sch.get_producers(b_reduce_1)[0])

        loops = sch.get_loops(b_spatial)
        bx = sch.fuse(*loops[:-1])
        _, tx = sch.split(loops[-1], [None, len_tx])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        sch.set_scope(b_reduce_1, buffer_index=0, storage_scope="shared")
        sch.compute_at(b_reduce_1, bx, preserve_unit_loops=True)
        _, s, *_, tx = sch.get_loops(b_reduce_1)
        _, tx = sch.split(tx, [None, len_tx])
        sch.bind(tx, "threadIdx.x")
        # _, tx = sch.split(
        #     sch.fuse(*sch.get_loops(sch.decompose_reduction(b_reduce_1, s))[1:]),
        #     [None, len_tx],
        # )
        sch.bind(tx, "threadIdx.x")

        sch.set_scope(b_reduce_0, buffer_index=0, storage_scope="shared")
        sch.compute_at(b_reduce_0, bx, preserve_unit_loops=True)
        _, s, *_, tx = sch.get_loops(b_reduce_0)
        _, tx = sch.split(tx, [None, len_tx])
        sch.bind(tx, "threadIdx.x")
        # _, tx = sch.split(
        #     sch.fuse(*sch.get_loops(sch.decompose_reduction(b_reduce_0, s))[1:]),
        #     [None, len_tx],
        # )
        sch.bind(tx, "threadIdx.x")
        auto_inline_consumers(sch, b_spatial)
        sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)

        # sch.mod.show(black_format=False)

    return sch_func
