"""The compilation pipeline for LLM applications."""
import logging

import tvm
from tvm import IRModule
from tvm import dlight as dl
from tvm.relax import register_pipeline  # pylint: disable=no-name-in-module

from .clean_up_tir_attrs import CleanUpTIRAttrs
from .estimate_memory_usage import EstimateMemoryUsage
from .fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
from .fuse_dequantize_take import FuseDequantizeTake
from .fuse_dequantize_transpose import FuseDequantizeTranspose
from .fuse_transpose_matmul import FuseTransposeMatmul
from .lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc

logger = logging.getLogger(__name__)


@tvm.transform.module_pass(opt_level=0, name="_LogProgress")
class _LogProgress:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging."""

    def __init__(self, *args):
        self.args = args

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation"""
        logger.info(*self.args)
        return mod


@register_pipeline("mlc_llm")
def _mlc_llm_pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 1. Passes on high-level operator graph
                _LogProgress("Running TVM Relax graph-level optimizations"),
                FuseDequantizeTranspose(skip_gemm=False),
                FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                _LogProgress("Lowering to TVM TIR kernels"),
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                _LogProgress("Running TVM TIR-level optimizations"),
                FuseDequantizeMatmulEwise(),
                FuseDequantizeTake(),
                tvm.relax.transform.DeadCodeElimination(),
                CleanUpTIRAttrs(["op_pattern"]),
                # Phase 4. Low-level Optimizations
                _LogProgress("Running TVM Dlight low-level optimizations"),
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                _LogProgress("Running memory optimizations"),
                LiftTIRGlobalBufferAlloc(),
                tvm.tir.transform.ForceNarrowIndexToInt32(),
                EstimateMemoryUsage(),
            ]
        )
        mod = seq(mod._move())  # pylint: disable=protected-access
        return mod

    return _pipeline
