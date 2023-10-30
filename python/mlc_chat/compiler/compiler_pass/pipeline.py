"""The compilation pipeline for LLM applications."""
import tvm
from tvm import dlight as dl
from tvm.relax import register_pipeline  # pylint: disable=no-name-in-module

from .clean_up_tir_attrs import CleanUpTIRAttrs
from .fuse_decode_matmul_ewise import FuseDecodeMatmulEwise
from .fuse_decode_take import FuseDecodeTake
from .fuse_decode_transpose import FuseDecodeTranspose
from .fuse_transpose_matmul import FuseTransposeMatmul
from .lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc


@register_pipeline("mlc_llm")
def _mlc_llm_pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 1. Passes on high-level operator graph
                FuseDecodeTranspose(skip_gemm=False),
                FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                FuseDecodeMatmulEwise(),
                FuseDecodeTake(),
                tvm.relax.transform.DeadCodeElimination(),
                CleanUpTIRAttrs(["op_pattern"]),
                # Phase 4. Low-level Optimizations
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                LiftTIRGlobalBufferAlloc(),
                tvm.tir.transform.ForceNarrowIndexToInt32(),
            ]
        )
        mod = seq(mod._move())  # pylint: disable=protected-access
        return mod

    return _pipeline
