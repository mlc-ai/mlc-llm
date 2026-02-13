"""LoRA injection pass for Relax."""

from __future__ import annotations

import tvm
from tvm import IRModule, ir, relax
from tvm.relax.expr_functor import mutator

# pylint: disable=abstract-method,arguments-renamed,no-member


@mutator
class _LoraInjectMutator(relax.PyExprMutator):
    """Inject `get_lora_delta` into each dense/linear weight with `param_name` attr."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.mod = mod

    def transform(self) -> IRModule:
        """Entry point."""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                new_func = self.visit_expr(func)
                self.builder_.update_func(g_var, new_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # type: ignore[override]
        new_call = super().visit_call_(call)
        if (
            not isinstance(new_call, relax.Call)
            or new_call.attrs is None
            or not hasattr(new_call.attrs, "param_name")
        ):
            return new_call

        param_name = new_call.attrs.param_name
        if param_name is None or len(new_call.args) < 2:
            return new_call

        weight = new_call.args[1]
        # Keep delta tensor shape/type aligned with the original weight tensor.
        delta = relax.call_pure_packed(
            "mlc.get_lora_delta",
            param_name,
            sinfo_args=weight.struct_info,  # type: ignore[arg-type]
        )
        new_args = list(new_call.args)
        new_args[1] = relax.add(weight, delta)
        return relax.Call(
            new_call.op,
            new_args,
            new_call.attrs,
            new_call.type_args,
            new_call.span,
        )


@tvm.transform.module_pass(opt_level=0, name="InjectLoRADelta")
class _InjectLoRADelta:  # pylint: disable=too-few-public-methods
    """Module pass wrapper for LoRA delta injection."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Apply LoRA injection when enabled, otherwise return module unchanged."""
        if not self.enabled:
            return mod
        return _LoraInjectMutator(mod).transform()


def make_lora_inject_pass(enabled: bool) -> ir.transform.Pass:
    """Return a pass that injects LoRA deltas when *enabled* is True."""
    return _InjectLoRADelta(enabled)
