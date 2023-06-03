# pylint: disable=missing-docstring
from typing import List

import numpy as np
import tvm
from tvm import tir

from mlc_llm.dlight import sch_decode_gemv as S
from mlc_llm.dlight.mod_decode_gemv import Module

TARGET = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
DEVICE = tvm.cuda(0)


def build_and_measure(idx: int, func: tir.PrimFunc, sch: tir.Schedule):
    args: List[np.ndarray] = []
    shapes: List[List[int]] = []
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        shapes.append(shape)
        np_array = np.random.uniform(size=shape).astype(buffer.dtype)
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)

    rt_mod = tvm.build(sch.mod, target=TARGET)
    rt_mod(*args)

    DEVICE.sync()
    time_eval = rt_mod.time_evaluator(rt_mod.entry_name, DEVICE, number=10)
    DEVICE.sync()
    time = time_eval(*args).mean
    DEVICE.sync()

    time = time * 1e3

    print(
        idx,
        f"Time (ms): {time:.3f}",
        "",
        f'shape: {", ".join(map(str, shapes))}',
        sep="\t",
    )


def main():
    failed = []
    for i in range(1, 7):
        # print("######### Working on func", i, "#########")
        func = Module[f"func{i}"].with_attr("global_symbol", "main")
        try:
            sch = tir.Schedule(func)
            S.sch_decode_gemv()(sch)
            # sch.mod.show(black_format=False)
            build_and_measure(
                i,
                func,
                sch,
            )
        except Exception as e:
            print(f"Failed to schedule func{i}")
            print(e)
            failed.append(i)
            raise
    print(f"Total {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
