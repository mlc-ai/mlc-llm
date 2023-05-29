# pylint: disable=missing-docstring
import tvm
from tvm import IRModule


@tvm.transform.module_pass(opt_level=0, name="DispatchTIROperator")
class DispatchTIROperator:  # pylint: disable=too-few-public-methods
    def __init__(self, model: str):
        # pylint: disable=import-outside-toplevel
        if model == "llama":
            from .llama import lookup

        elif model == "gpt_neox":
            from .gpt_neox import lookup

        else:
            raise ValueError(f"Model {model} not supported")
        self.lookup = lookup

    # pylint: enable=import-outside-toplevel

    def transform_module(
        self,
        mod: IRModule,
        ctx: tvm.transform.PassContext,
    ) -> IRModule:
        for gv in mod.functions:
            scheduled_func = self.lookup(mod[gv])
            if scheduled_func is not None:
                mod[gv] = scheduled_func
                print("- Dispatch to pre-scheduled op:", gv.name_hint)

        return mod
