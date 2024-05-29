"""Memory usage estimation analysis function for Relax functions."""

import json
from typing import Any, Dict

import tvm
from tvm import relax, tir
from tvm.ir import IRModule, Op
from tvm.relax.expr_functor import PyExprVisitor, visitor

from mlc_llm.support import logging

logger = logging.getLogger(__name__)


@tvm.transform.module_pass(opt_level=0, name="AttachMetadata")
class AttachMetadataWithMemoryUsage:  # pylint: disable=too-few-public-methods
    """Attach a Relax function that returns metadata in a JSON string"""

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""

        func_name = "_metadata"

        def _emit_metadata(metadata):
            bb = relax.BlockBuilder()  # pylint: disable=invalid-name
            with bb.function(func_name, params=[]):
                bb.emit_func_output(relax.StringImm(json.dumps(metadata)))
            return bb.finalize()[func_name]

        self.metadata["memory_usage"] = _MemoryEstimator().run(mod)
        mod[func_name] = _emit_metadata(self.metadata)
        return mod


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
                logger.info(
                    "[Memory usage] Function `%s`: %.2f MB",
                    global_var.name_hint,
                    self.planned_alloc_mem / 1024 / 1024,
                )
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
        if isinstance(size.values[0], tir.IntImm):
            self.planned_mem_num += 1
            self.planned_alloc_mem += size.values[0].value
