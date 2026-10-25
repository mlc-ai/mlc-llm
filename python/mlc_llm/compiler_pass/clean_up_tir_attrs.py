"""A compiler pass that cleans up undesired TIR attrs."""

from typing import List  # noqa: UP035

import tvm_tirx
from tvm_tirx.ir.module import IRModule


@tvm_tirx.transform.module_pass(opt_level=0, name="CleanUpTIRAttrs")
class CleanUpTIRAttrs:
    """A compiler pass that cleans up undesired TIR attrs."""

    def __init__(self, attrs: List[str]):  # noqa: UP006
        self.attrs = attrs

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm_tirx.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        for g_var, func in mod.functions_items():
            changed = False
            for attr in self.attrs:
                if func.attrs is not None and attr in func.attrs:
                    func = func.without_attr(attr)
                    changed = True
                    break
            if changed:
                mod[g_var] = func
        return mod
