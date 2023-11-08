"""Memory usage estimation analysis function for Relax functions."""
from typing import Dict

import tvm
from tvm import relax
from tvm.ir import IRModule, Op
from tvm.relax.expr_functor import PyExprVisitor, visitor


@tvm.transform.module_pass(opt_level=0, name="EstimateMemoryUsage")
class EstimateMemoryUsage:  # pylint: disable=too-few-public-methods
    """A pass that attaches the memory usage information as an IRModule attribute.

    This pass relies on static analysis on each TVM Relax function in the specific IRModule.
    It simply accumulates all memory allocation calls in a function, and does not consider
    more dynamic runtime features like control flo "if" or function calls.
    """

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entry point of the pass."""
        lowered_mod = tvm.transform.Sequential(
            [
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
            ],
            name="relax.lower",
        )(mod)
        usage = _MemoryEstimator().run(lowered_mod)
        return mod.with_attr("mlc_llm.memory_usage", usage)


@visitor
class _MemoryEstimator(PyExprVisitor):
    """The IR visitor which estimates the memory usage of each Relax function."""

    def __init__(self) -> None:
        self.planned_alloc_mem = 0
        self.planned_mem_num = 0
        self._op_alloc_tensor = Op.get("relax.builtin.alloc_tensor")
        self._op_alloc_storage = Op.get("relax.memory.alloc_storage")

    def run(self, mod: IRModule) -> Dict[str, int]:
        """Entry point of the visitor."""
        result: Dict[str, int] = {}
        for global_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.planned_alloc_mem = 0
                self.planned_mem_num = 0
                self.visit_expr(func)
                result[global_var.name_hint] = self.planned_alloc_mem
        return result

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        if call.op == self._op_alloc_tensor:
            self._builtin_tensor_alloc(shape=call.args[0], dtype_str=call.args[1].value)
        elif call.op == self._op_alloc_storage:
            self._storage_alloc(size=call.args[0])
        super().visit_call_(call)

    def _builtin_tensor_alloc(self, shape: relax.Expr, dtype_str: str) -> None:
        assert isinstance(shape, relax.ShapeExpr)
        size = 1
        for dim_len in shape.values:
            if not isinstance(dim_len, tvm.tir.IntImm):
                return
            size *= dim_len.value
        dtype = tvm.DataType(dtype_str)
        self.planned_mem_num += 1
        self.planned_alloc_mem += size * ((dtype.bits + 7) // 8) * dtype.lanes

    def _storage_alloc(self, size: relax.Expr) -> None:
        assert isinstance(size, relax.ShapeExpr)
        self.planned_mem_num += 1
        self.planned_alloc_mem += size.values[0].value
