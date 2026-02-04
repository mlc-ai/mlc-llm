"""A compiler pass that scatters TupleGetItem for lazy TupleGetItems."""

from typing import Dict

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr import Expr, Var
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="ScatterTupleGetItem")
class ScatterTupleGetItem:  # pylint: disable=too-few-public-methods
    """A compiler pass that scatters TupleGetItem for lazy TupleGetItems."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        return _Scatter(mod).transform()


@mutator
class _Scatter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod = mod
        self.var_map: Dict[Var, Expr] = {}

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                updated_func = remove_all_unused(updated_func)
                self.builder_.update_func(g_var, updated_func)
        return self.builder_.get()

    def visit_var_binding_(self, binding: relax.VarBinding):
        super().visit_var_binding_(binding)
        if isinstance(binding.value, relax.TupleGetItem):
            self.var_map[binding.var] = binding.value

    def visit_dataflow_var_(  # pylint: disable=arguments-renamed
        self, var: relax.DataflowVar
    ) -> Expr:
        if var in self.var_map:
            new_var = self.builder_.emit(self.var_map[var], name_hint=var.name_hint)
            self.set_var_remap(var.vid, new_var)
            self.var_map.pop(var)
            return new_var
        return var
