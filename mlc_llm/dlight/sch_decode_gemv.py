# pylint: disable=missing-docstring
from typing import Callable

from tvm import tir

from .sch_inline import auto_inline_consumers


def sch_decode_gemv(  # pylint: disable=too-many-statements,too-many-arguments
    name_decode: str = "decode",
    name_gemv: str = "matmul",
    len_vx: int = 2,
    len_tx: int = 64,
    len_km: int = 2,
    len_ki: int = 1 * 8,
) -> Callable[[tir.Schedule], None]:
    def sch_func(
        sch: tir.Schedule,
    ):  # pylint: disable=too-many-branches,too-many-locals
        gemv = sch.get_block(name_gemv)
        decode = sch.get_block(name_decode)
        # Step 1. Schedule GEMV
        # [b=1, i=1, j, k]
        # split j           => [b=1, i=1, (bx, vx, tx), k]
        # fuse (b, i, bx)   => [bx, vx, tx, (k)]
        # split k           => [bx, vx, tx, (ko, k_m, ki * 8)]
        rb = sch.cache_write(gemv, 0, "local")  # pylint: disable=invalid-name
        b, i, j, k = sch.get_loops(gemv)  # pylint: disable=invalid-name
        assert sch.get(b).extent.value == 1
        assert sch.get(i).extent.value == 1
        bx, vx, tx = sch.split(  # pylint: disable=invalid-name
            j, [None, len_vx, len_tx]
        )
        bx = sch.fuse(b, i, bx)  # pylint: disable=invalid-name
        k_o, k_m, k_i = sch.split(k, [None, len_km, len_ki])
        sch.bind(bx, thread_axis="blockIdx.x")
        sch.bind(vx, thread_axis="vthread.x")
        sch.bind(tx, thread_axis="threadIdx.x")
        sch.reorder(bx, vx, tx, k_o, k_m, k_i)
        sch.unroll(k_i)
        # Step 2. Schedule decode: move to under threadIdx.x and fetch separately for each thread
        sch.compute_at(decode, k_m, preserve_unit_loops=True)
        sch.set_scope(decode, 0, "local")
        _, unroll = sch.split(sch.get_loops(decode)[-2], [None, 8])
        sch.unroll(unroll)

        # Step 3. Cooperative fetch GEMV
        def cooperative_fetch(block, tx):  # pylint: disable=invalid-name
            block = sch.cache_read(block, 0, "shared")
            sch.compute_at(block, tx, preserve_unit_loops=True)
            loop = sch.fuse(*sch.get_loops(block)[-2:])
            len_vector = sch.sample_categorical(
                [1, 2, 3, 4],
                probs=[0.25, 0.25, 0.25, 0.25],
            )
            _, tx, vec = sch.split(loop, [None, len_tx, len_vector])
            sch.bind(tx, thread_axis="threadIdx.x")
            sch.vectorize(vec)
            sch.storage_align(block, buffer_index=0, axis=-2, factor=32, offset=8)

        cooperative_fetch(gemv, k_o)
        # Step 4. Schedule epilogue
        auto_inline_consumers(sch, rb)
        sch.reverse_compute_at(rb, tx, preserve_unit_loops=True)
        # Step 5. Postprocess: decompose reduction
        sch.decompose_reduction(gemv, k_o)

    return sch_func
