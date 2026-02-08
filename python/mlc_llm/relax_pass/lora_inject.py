"""LoRA injection pass for Relax."""

from __future__ import annotations

from tvm import ir, relax

# pylint: disable=abstract-method,arguments-renamed,no-member


class _LoraInjectMutator(relax.PyExprMutator):
    """Inject `get_lora_delta` into every dense/linear weight that has param_name attr."""

    def visit_call_(self, call: relax.Call):  # type: ignore[override]
        new_call = super().visit_call_(call)
        if not isinstance(new_call, relax.Call):
            return new_call

        param_name = new_call.attrs.get("param_name", None) if new_call.attrs else None
        if param_name is None:
            return new_call

        # Only process matmul/dense style ops where the weight is the second arg.
        if len(new_call.args) < 2:
            return new_call

        weight = new_call.args[1]
        delta = relax.call_packed("mlc.get_lora_delta", param_name)
        new_weight = relax.add(weight, delta)
        new_args = list(new_call.args)
        new_args[1] = new_weight
        return relax.Call(new_call.op, new_args, new_call.attrs, new_call.type_args, new_call.span)


def make_lora_inject_pass(enabled: bool) -> ir.transform.Pass:
    """Return a FunctionPass that injects LoRA deltas when *enabled* is True."""

    if not enabled:
        # Create a no-op pass if Identity doesn't exist
        try:
            return relax.transform.Identity()
        except AttributeError:
            # Fallback: create a pass that does nothing
            def _identity_transform(func: relax.Function, _mod: ir.IRModule, _ctx):
                return func

            return relax.transform.FunctionPass(
                _identity_transform,
                opt_level=0,
                name="IdentityLoRAPass",
            )

    def _transform(
        func: relax.Function, _mod: ir.IRModule, _ctx
    ):  # pylint: disable=unused-argument
        return _LoraInjectMutator().visit_expr(func)  # type: ignore[arg-type]

    return relax.transform.FunctionPass(
        _transform,
        opt_level=0,
        name="InjectLoRADelta",
    )
