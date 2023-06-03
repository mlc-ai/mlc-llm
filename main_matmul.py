# pylint: disable=missing-docstring
from typing import Callable, List

import numpy as np
import tvm
from tvm import tir

from mlc_llm.dlight import sch_matmul as S
from mlc_llm.dlight.mod_matmul import Module

TARGET = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
DEVICE = tvm.cuda(0)


def build_and_measure(idx: int, func: tir.PrimFunc, sch_funcs: List[Callable]):
    args: List[np.ndarray] = []
    analyzer = tvm.arith.Analyzer()
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tir.Var):
                shape.append(128)
                analyzer.bind(dim, 128)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        np_array = np.random.uniform(size=shape).astype(buffer.dtype)
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)

    all_times = []
    all_flops = []
    for sch_func in sch_funcs:
        sch = tir.Schedule(func)
        sch_func(sch)

        rt_mod = tvm.build(sch.mod, target=TARGET)
        rt_mod(*args)

        DEVICE.sync()
        time_eval = rt_mod.time_evaluator(rt_mod.entry_name, DEVICE, number=10)
        DEVICE.sync()
        time = time_eval(*args).mean
        DEVICE.sync()

        num_b = S.B.value if hasattr(S.B, "value") else S.B  # type: ignore
        num_x = S.X.value if hasattr(S.X, "value") else S.X  # type: ignore
        num_y = S.Y.value if hasattr(S.Y, "value") else S.Y  # type: ignore
        num_k = S.K.value if hasattr(S.K, "value") else S.K  # type: ignore

        b = analyzer.simplify(S.B).value
        x = analyzer.simplify(S.X).value
        y = analyzer.simplify(S.Y).value
        k = analyzer.simplify(S.K).value
        if b > 1:
            return
        flop = b * x * y * k
        flops = flop / time / 1e9
        all_times.append(time * 1e3)
        all_flops.append(flops)

    print(
        idx,
        f"Time (ms): {min(all_times):.3f}",
        f"GFLOPS: {max(all_flops):.3f}",
        "",
        f"b, x, y, k: {num_b}, {num_x}, {num_y}, {num_k}",
        sep="\t",
    )


def main():
    failed = []
    for i in range(1, 53):
        # print("######### Working on func", i, "#########")
        func = Module[f"func{i}"].with_attr("global_symbol", "main")
        if 'with T.block("matmul"):' in func.script():
            matmul = "matmul"
        elif 'with T.block("NT_matmul"):' in func.script():
            matmul = "NT_matmul"
        else:
            print(f"Weird workload. Skipping func{i}")
        sch_0 = S.sch_matmul(matmul, smem_transpose_a=False, smem_transpose_b=False)
        sch_1 = S.sch_matmul(matmul, smem_transpose_a=False, smem_transpose_b=True)
        sch_2 = S.sch_matmul(matmul, smem_transpose_a=True, smem_transpose_b=False)
        sch_3 = S.sch_matmul(matmul, smem_transpose_a=True, smem_transpose_b=True)
        try:
            sch = tir.Schedule(func)
            S.sch_matmul(matmul)(sch)
            # sch.mod.show(black_format=False)
            build_and_measure(
                i,
                func,
                [sch_0, sch_1, sch_2, sch_3],
            )
        except Exception as e:
            print(f"Failed to schedule func{i}")
            print(e)
            failed.append(i)
            raise
    print(f"Total {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
