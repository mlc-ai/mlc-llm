"""A compiler pass that dispatches patterns to CUBLAS."""
import tvm
import tvm.relax.backend.contrib.cublas as _cublas
from tvm import IRModule, relax
from tvm.relax.backend import get_patterns_with_prefix


@tvm.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:  # pylint: disable=too-few-public-methods,broad-exception-raised
    """A compiler pass that dispatches patterns to CUBLAS."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        has_cublas = tvm.get_global_func("relax.ext.cublas", True)
        if not has_cublas:
            raise Exception("CUBLAS is not enabled.")

        patterns = get_patterns_with_prefix("cublas")

        model_names = [
            gv.name_hint for gv, func in mod.functions.items() if isinstance(func, relax.Function)
        ]
        mod = tvm.transform.Sequential(
            [
                relax.transform.FuseOpsByPattern(
                    patterns, bind_constants=False, annotate_codegen=True
                ),
                relax.transform.RunCodegen({}, entry_functions=model_names),
            ]
        )(mod)
        return mod
