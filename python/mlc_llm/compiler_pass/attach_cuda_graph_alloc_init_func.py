"""The pass that attaches an empty function for initialization."""

import tvm
from tvm import IRModule, relax


@tvm.transform.module_pass(opt_level=0, name="AttachCUDAGraphAllocInitFunc")
class AttachCUDAGraphAllocInitFunc:  # pylint: disable=too-few-public-methods
    """Attach an empty function for initialization."""

    def __init__(self):
        pass

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        bb = relax.BlockBuilder(mod)
        alloc_func_gv = None
        for gv, _ in mod.functions_items():
            if gv.name_hint.startswith("cuda_graph_alloc"):
                assert alloc_func_gv is None
                alloc_func_gv = gv
        if alloc_func_gv is None:
            return mod

        with bb.function("cuda_graph_alloc_init", []):
            bb.emit_func_output(
                relax.op.call_builtin_with_ctx(
                    "vm.builtin.cuda_graph.get_cached_alloc",
                    args=[alloc_func_gv, relax.PrimValue(0)],
                    sinfo_args=relax.ObjectStructInfo(),
                )
            )
        return bb.finalize()
