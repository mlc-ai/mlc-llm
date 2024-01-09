"""The compilation pipeline for LLM applications."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import tvm
from tvm import IRModule
from tvm import dlight as dl
from tvm.relax import register_pipeline  # pylint: disable=no-name-in-module
from tvm.relax.frontend import nn

from mlc_chat.support import logging

from .attach_to_ir_module import AttachAdditionalPrimFuncs, AttachVariableBounds
from .clean_up_tir_attrs import CleanUpTIRAttrs
from .cublas_dispatch import CublasDispatch
from .estimate_memory_usage import AttachMetadataWithMemoryUsage
from .fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
from .fuse_dequantize_take import FuseDequantizeTake
from .fuse_dequantize_transpose import FuseDequantizeTranspose
from .fuse_ft_dequantize_matmul_epilogue import FuseFTDequantizeEpilogue
from .fuse_transpose_matmul import FuseTransposeMatmul
from .lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc
from .prune_relax_func import PruneRelaxFunc

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


@tvm.transform.module_pass(opt_level=0, name="DebugDump")
class _DebugDump:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging.
    Only enabled when debug_dump is not None"""

    def __init__(self, file_name: str, file_path: Optional[Path], show_meta: bool = False):
        self.file_name = file_name
        self.file_path = file_path
        self.show_meta = show_meta

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation that dumps the module to file"""
        if self.file_path is not None:
            # NOTE: We use debug level here to avoid spamming the console
            logger.debug("Dumping IR to %s", self.file_path / self.file_name)
            with open(self.file_path / self.file_name, "w", encoding="utf-8") as f:
                f.write(mod.script(show_meta=self.show_meta))
        return mod


@register_pipeline("mlc_llm")
def _mlc_llm_pipeline(  # pylint: disable=too-many-arguments
    flashinfer: bool = False,
    cublas_gemm: bool = False,
    variable_bounds: Dict[str, int] = None,
    additional_tirs: Dict[str, tvm.tir.PrimFunc] = None,
    metadata: Dict[str, Any] = None,
    ext_mods: List[nn.ExternModule] = None,
    debug_dump: Optional[Path] = None,
):
    variable_bounds = variable_bounds or {}
    additional_tirs = additional_tirs or {}
    metadata = metadata or {}
    ext_mods = ext_mods or []

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 0. Add additional information for compilation and remove unused Relax func
                PruneRelaxFunc(flashinfer=flashinfer),
                AttachVariableBounds(variable_bounds),
                AttachAdditionalPrimFuncs(additional_tirs),
                _DebugDump("debug-phase0.py", debug_dump, show_meta=False),
                # Phase 1. Passes on high-level operator graph
                _LogProgress("Running TVM Relax graph-level optimizations"),
                FuseFTDequantizeEpilogue(),
                FuseDequantizeTranspose(),
                CublasDispatch() if cublas_gemm else tvm.transform.Sequential([]),
                FuseTransposeMatmul(),
                _DebugDump("debug-phase1.py", debug_dump, show_meta=False),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                _LogProgress("Lowering to TVM TIR kernels"),
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                _DebugDump("debug-phase2.py", debug_dump, show_meta=False),
                # Phase 3. Passes on TIR
                _LogProgress("Running TVM TIR-level optimizations"),
                FuseDequantizeMatmulEwise(),
                FuseDequantizeTake(),
                tvm.relax.transform.DeadCodeElimination(),
                CleanUpTIRAttrs(["op_pattern"]),
                _DebugDump("debug-phase3.py", debug_dump, show_meta=False),
                # Phase 4. Low-level Optimizations
                _LogProgress("Running TVM Dlight low-level optimizations"),
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                _DebugDump("debug-phase4.py", debug_dump, show_meta=False),
                _LogProgress("Lowering to VM bytecode"),
                LiftTIRGlobalBufferAlloc(),
                tvm.tir.transform.ForceNarrowIndexToInt32(),
                tvm.relax.transform.RewriteDataflowReshape(),
                tvm.relax.transform.ToNonDataflow(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                tvm.relax.transform.StaticPlanBlockMemory(),
                AttachMetadataWithMemoryUsage(metadata),
                tvm.relax.transform.RewriteCUDAGraph(),
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.VMBuiltinLower(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
                _DebugDump("debug-final.py", debug_dump, show_meta=False),
                _LogProgress("Compiling external modules"),
                tvm.relax.transform.AttachExternModules(ext_mods),
                _LogProgress("Compilation complete! Exporting to disk"),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline
