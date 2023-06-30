"""Clean up TIR attributes that may affect dispatching"""

import tvm
from tvm.ir.module import IRModule


@tvm.transform.module_pass(opt_level=0, name="CleanUpTIRAttrs")
class CleanUpTIRAttrs:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        undesired_attrs = ["op_pattern"]

        for gv in list(mod.functions):
            func = mod[gv]
            changed = False
            for attr in undesired_attrs:
                if func.attrs is not None and attr in func.attrs:
                    func = func.without_attr(attr)
                    changed = True
                    break

            if changed:
                mod[gv] = func
        return mod
