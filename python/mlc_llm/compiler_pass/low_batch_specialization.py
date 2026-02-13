"""A compiler pass that dispatch low-batch-gemm to gemv schedule."""

import tvm
from tvm import tir
from tvm.ir.module import IRModule
from tvm.s_tir import dlight as dl

# pylint: disable=too-many-locals,not-callable


@tvm.transform.module_pass(opt_level=0, name="LowBatchGemvSpecialize")
class LowBatchGemvSpecialize:  # pylint: disable=too-few-public-methods
    """A compiler pass that dispatch low-batch-gemm to gemv schedule."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                low_batch_range = [2, 8]
                buckets = [2, 4]
                low_batch_funcs = []
                for bucket in buckets:
                    low_batch_mod = IRModule({})
                    low_batch_mod["main"] = func
                    low_batch_mod = dl.ApplyDefaultSchedule(
                        dl.gpu.LowBatchGEMV(bucket),
                    )(low_batch_mod)
                    low_batch_funcs.append(low_batch_mod["main"])
                if any(
                    tvm.ir.structural_equal(low_batch_func, func)
                    for low_batch_func in low_batch_funcs
                ):
                    continue
                buffers = func.buffer_map.values()
                shapes = [buffer.shape for buffer in buffers]
                symbolic_vars = set(
                    expr for shape in shapes for expr in shape if isinstance(expr, tir.Var)
                )
                if len(symbolic_vars) != 1:
                    continue
                gemm_mod = IRModule({})
                gemm_mod["main"] = func
                gemm_mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                )(gemm_mod)
                gemm_func = gemm_mod["main"]
                sym_var = list(symbolic_vars)[0]
                body = gemm_func.body
                for i, range_limit in reversed(list(enumerate(low_batch_range))):
                    body = tir.IfThenElse(
                        tir.op.tvm_thread_invariant(sym_var <= range_limit),
                        low_batch_funcs[i].body,
                        body,
                    )
                body = tir.SBlock([], [], [], "root", body)
                body = tir.SBlockRealize([], True, body)
                new_func = func.with_body(body)
                new_func = new_func.with_attr("tir.is_scheduled", 1)
                new_func = new_func.with_attr("tir.HoistIfThenElseExprWithBlock", 1)
                mod.update_func(g_var, new_func)
        return mod
