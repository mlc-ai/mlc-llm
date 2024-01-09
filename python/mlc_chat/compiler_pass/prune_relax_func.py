"""A pass that removes undesired Relax function from IRModule based on target."""
import tvm
from tvm import IRModule


@tvm.transform.module_pass(opt_level=0, name="PruneRelaxFunc")
class PruneRelaxFunc:  # pylint: disable=too-few-public-methods
    """Removes undesired Relax function from IRModule based on target."""

    def __init__(self, flashinfer: bool) -> None:
        """Initializer.

        Parameters
        ----------
        flashinfer : bool
            A boolean indicating if flashinfer is enabled.
        """
        self.flashinfer = flashinfer

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        func_dict = {}
        for g_var, func in mod.functions_items():
            # Remove "create_flashinfer_paged_kv_cache" for unsupported target
            if g_var.name_hint == "create_flashinfer_paged_kv_cache" and not self.flashinfer:
                continue
            func_dict[g_var] = func
        ret_mod = IRModule(func_dict)
        if mod.attrs is not None:
            ret_mod = ret_mod.with_attrs(mod.attrs)
        return ret_mod
