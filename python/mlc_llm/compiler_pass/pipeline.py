"""The compilation pipeline for LLM applications."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import tvm
from tvm import IRModule
from tvm.relax import register_pipeline  # pylint: disable=no-name-in-module
from tvm.relax.frontend import nn
from tvm.s_tir import dlight as dl

from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType
from mlc_llm.support import logging

from .attach_cuda_graph_alloc_init_func import AttachCUDAGraphAllocInitFunc
from .attach_embedding_allocator import AttachAllocEmbeddingTensorFunc
from .attach_logit_processor import AttachLogitProcessFunc
from .attach_sampler import AttachGPUSamplingFunc
from .attach_softmax_with_temperature import AttachSoftmaxWithTemperature
from .attach_spec_decode_aux_funcs import AttachSpecDecodeAuxFuncs
from .attach_support_info import (
    AttachAdditionalPrimFuncs,
    AttachCUDAGraphSymbolicCaptureHints,
    AttachMemoryPlanAttr,
    AttachPipelineParallelStages,
    AttachSequenceLengthPaddingFactor,
    AttachVariableBounds,
)
from .blas_dispatch import BLASDispatch
from .clean_up_tir_attrs import CleanUpTIRAttrs
from .dispatch_kv_cache_creation import DispatchKVCacheCreation
from .dispatch_triton_kernel import DispatchTritonKernel
from .estimate_memory_usage import AttachMetadataWithMemoryUsage
from .fuse_add_norm import FuseAddRMSNorm
from .fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
from .fuse_dequantize_take import FuseDequantizeTake
from .fuse_dequantize_transpose import FuseDequantizeTranspose
from .fuse_ft_dequantize_matmul_epilogue import FuseFTDequantizeEpilogue
from .fuse_transpose_matmul import FuseTransposeMatmul
from .lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc
from .low_batch_specialization import LowBatchGemvSpecialize
from .pipeline_parallel_rewrite import PipelineParallelRewrite
from .scatter_tuple_get_item import ScatterTupleGetItem

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
    target: tvm.target.Target,
    flashinfer: bool = False,
    cublas_gemm: bool = False,
    faster_transformer: bool = False,  # pylint: disable=unused-argument
    allreduce_strategy: IPCAllReduceStrategyType = IPCAllReduceStrategyType.NONE,
    variable_bounds: Dict[str, int] = None,
    cuda_graph_symbolic_capture_hints: Dict[str, List[str]] = None,
    additional_tirs: Dict[str, tvm.tir.PrimFunc] = None,
    metadata: Dict[str, Any] = None,
    ext_mods: List[nn.ExternModule] = None,
    debug_dump: Optional[Path] = None,
):
    variable_bounds = variable_bounds or {}
    cuda_graph_symbolic_capture_hints = cuda_graph_symbolic_capture_hints or {}
    additional_tirs = additional_tirs or {}
    metadata = metadata or {}
    ext_mods = ext_mods or []
    tensor_parallel_shards = metadata.get("tensor_parallel_shards", 1)

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 0. Add additional information for compilation and remove unused Relax func
                DispatchKVCacheCreation(target, flashinfer, metadata),
                AttachSoftmaxWithTemperature(target, metadata),
                AttachVariableBounds(variable_bounds),
                AttachCUDAGraphSymbolicCaptureHints(cuda_graph_symbolic_capture_hints),
                AttachPipelineParallelStages(metadata["pipeline_parallel_stages"]),
                AttachLogitProcessFunc(target),
                AttachAdditionalPrimFuncs(additional_tirs),
                AttachAllocEmbeddingTensorFunc(metadata),
                AttachGPUSamplingFunc(target, variable_bounds),
                AttachSpecDecodeAuxFuncs(tensor_parallel_shards),
                AttachMemoryPlanAttr(),
                AttachSequenceLengthPaddingFactor(target, metadata),
                tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
                _DebugDump("debug-phase0.py", debug_dump, show_meta=False),
                # Phase 1. Passes on high-level operator graph
                _LogProgress("Running TVM Relax graph-level optimizations"),
                DispatchTritonKernel(target),
                FuseFTDequantizeEpilogue(),
                FuseDequantizeTranspose(),
                BLASDispatch(target) if cublas_gemm else tvm.transform.Sequential([]),
                (
                    FuseAddRMSNorm(target=target)
                    if target.kind.name != "llvm"
                    else tvm.transform.Sequential([])
                ),
                FuseTransposeMatmul(),
                _DebugDump("debug-phase1.py", debug_dump, show_meta=False),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                _LogProgress("Lowering to TVM TIR kernels"),
                tvm.relax.backend.DispatchSampling(),
                tvm.relax.backend.DispatchSortScan(),
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
                LowBatchGemvSpecialize(),
                (
                    dl.ApplyDefaultSchedule(
                        dl.gpu.Matmul(),
                        dl.gpu.GEMV(),
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    )
                    if target.kind.name != "llvm"
                    else dl.ApplyDefaultSchedule(
                        dl.cpu.GEMV(),
                    )
                ),
                _DebugDump("debug-phase4.py", debug_dump, show_meta=False),
                _LogProgress("Lowering to VM bytecode"),
                (
                    LiftTIRGlobalBufferAlloc()
                    if target.kind.name != "llvm"
                    else tvm.transform.Sequential([])
                ),
                (
                    tvm.tir.transform.ForceNarrowIndexToInt32()
                    if target.kind.name != "cuda"
                    else tvm.transform.Sequential([])
                ),
                ScatterTupleGetItem(),
                PipelineParallelRewrite(),
                tvm.relax.transform.RewriteDataflowReshape(),
                tvm.relax.transform.ToNonDataflow(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                (
                    tvm.relax.transform.IPCAllReduceRewrite(allreduce_strategy)
                    if allreduce_strategy != IPCAllReduceStrategyType.NONE
                    else tvm.transform.Sequential([])
                ),
                tvm.relax.transform.StaticPlanBlockMemory(),
                AttachMetadataWithMemoryUsage(metadata),
                _DebugDump("debug-phase5.py", debug_dump, show_meta=False),
                tvm.relax.transform.RewriteCUDAGraph(),
                AttachCUDAGraphAllocInitFunc(),
                tvm.relax.transform.LowerGPUIPCAllocStorage(),
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.LowerRuntimeBuiltin(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
                _LogProgress("Compiling external modules"),
                tvm.relax.transform.AttachExternModules(ext_mods),
                _LogProgress("Compilation complete! Exporting to disk"),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline
