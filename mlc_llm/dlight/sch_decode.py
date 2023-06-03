# pylint: disable=missing-docstring
from typing import Callable

from tvm import tir

from .sch_inline import auto_inline_consumers, auto_inline_producers


def sch_decode(
    name_decode: str = "decode",
    len_tx: int = 8,  # <= 16
    len_ty: int = 8,
    len_yi: int = 1,  # <= 4
    len_yc: int = 8,
):
    def sch_func(sch: tir.Schedule) -> None:  # pylint: disable=too-many-locals
        decode = sch.get_block(name_decode)
        # Step 1. Tile the decoding
        i, j = sch.get_loops(decode)
        by, ty, yi, yc = sch.split(  # pylint: disable=invalid-name
            i, factors=[None, len_ty, len_yi, len_yc]
        )
        bx, tx = sch.split(j, factors=[None, len_tx])  # pylint: disable=invalid-name
        sch.reorder(by, bx, ty, tx, yi, yc)
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.unroll(yc)
        # Step 2. Cache results in shared memory
        rb = sch.cache_write(decode, 0, "shared")  # pylint: disable=invalid-name
        consumers = sch.get_consumers(rb)
        if consumers:
            (consumer,) = consumers
            auto_inline_consumers(sch, consumer)
            sch.compute_inline(rb)
            rb = consumer
        # Step 3. Schedule the shared memory write back
        sch.reverse_compute_at(rb, bx, preserve_unit_loops=True)
        loop = sch.fuse(*sch.get_loops(rb)[-2:])
        _, ty, tx, vec = sch.split(  # pylint: disable=invalid-name
            loop, factors=[None, len_ty, len_tx, 4]
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        sch.storage_align(decode, buffer_index=0, axis=0, factor=32, offset=1)

    return sch_func
