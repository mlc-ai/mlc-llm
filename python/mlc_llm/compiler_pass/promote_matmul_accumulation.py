"""Use FP32 accumulation for FP16 matrix products while preserving their output ABI."""

import tvm
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class _PromoteMatmul(PyExprMutator):
    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        if call.ty.dtype != "float16" or any(arg.ty.dtype != "float16" for arg in call.args):
            return call
        product = self.builder_.normalize(
            relax.op.matmul(call.args[0], call.args[1], out_dtype="float32")
        )
        return relax.op.astype(product, "float16")


@tvm.transform.module_pass(opt_level=0, name="PromoteMatmulAccumulation")
class PromoteMatmulAccumulation:
    """Promote only FP16 matmuls; weights, intermediate storage and public types stay FP16.

    This is an explicit arithmetic policy for matched native/WebGPU controls.
    Already-FP32 products, integer products and other operators are untouched.
    Apply before transpose/matmul fusion and TIR legalization.
    """

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        rewriter = _PromoteMatmul(mod)
        for global_var, function in mod.functions_items():
            if isinstance(function, relax.Function):
                rewriter.builder_.update_func(global_var, rewriter.visit_expr(function))
        return rewriter.builder_.get()
