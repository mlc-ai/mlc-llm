"""A couple of passes that simply supportive information onto the IRModule."""

from typing import Dict, List

import tvm
from tvm import IRModule, relax, tir


@tvm.transform.module_pass(opt_level=0, name="AttachVariableBounds")
class AttachVariableBounds:  # pylint: disable=too-few-public-methods
    """Attach variable bounds to each Relax function, which primarily helps with memory planning."""

    def __init__(self, variable_bounds: Dict[str, int]):
        # Specifically for RWKV workloads, which contains -1 max_seq_len
        self.variable_bounds = {k: v for k, v in variable_bounds.items() if v > 0}
        self.non_negative_var = ["vocab_size"]

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                mod[g_var] = func.with_attr("tir_var_upper_bound", self.variable_bounds).with_attr(
                    "tir_non_negative_var", self.non_negative_var
                )
        return mod


@tvm.transform.module_pass(opt_level=0, name="AttachAdditionalPrimFuncs")
class AttachAdditionalPrimFuncs:  # pylint: disable=too-few-public-methods
    """Attach extra TIR PrimFuncs to the IRModule"""

    def __init__(self, functions: Dict[str, tir.PrimFunc]):
        self.functions = functions

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        for func_name, func in self.functions.items():
            mod[func_name] = func.with_attr("global_symbol", func_name)
        return mod


@tvm.transform.module_pass(opt_level=0, name="AttachMemoryPlanAttr")
class AttachMemoryPlanAttr:  # pylint: disable=too-few-public-methods
    """Attach memory planning attribute for dynamic function output planning to Relax functions."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                mod[g_var] = func.with_attr("relax.memory_plan_dynamic_func_output", True)
        return mod


@tvm.transform.module_pass(opt_level=0, name="AttachCUDAGraphCaptureHints")
class AttachCUDAGraphSymbolicCaptureHints:  # pylint: disable=too-few-public-methods
    """Attach CUDA graph capture hints to the IRModule"""

    def __init__(self, hints: Dict[str, List[str]]):
        self.hints = hints

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        for g_var, func in mod.functions_items():
            func_name = g_var.name_hint
            if isinstance(func, relax.Function):
                if func_name in self.hints:
                    mod[g_var] = func.with_attr(
                        "relax.rewrite_cuda_graph.capture_symbolic_vars", self.hints[func_name]
                    )

        return mod


@tvm.transform.module_pass(opt_level=0, name="AttachPipelineParallelStages")
class AttachPipelineParallelStages:  # pylint: disable=too-few-public-methods
    """Attach number of pipeline stages to relax functions."""

    def __init__(self, pipeline_parallel_shards: int):
        self.pipeline_parallel_shards = pipeline_parallel_shards

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        for g_var, func in mod.functions_items():
            func_name = g_var.name_hint
            if not isinstance(func, relax.Function) or func_name not in [
                "prefill",
                "decode",
                "prefill_to_last_hidden_states",
                "decode_to_last_hidden_states",
                "batch_prefill",
                "batch_decode",
                "batch_verify",
                "batch_prefill_to_last_hidden_states",
                "batch_decode_to_last_hidden_states",
                "batch_verify_to_last_hidden_states",
            ]:
                continue
            mod[g_var] = func.with_attr("pipeline_parallel_stages", self.pipeline_parallel_shards)

        return mod
