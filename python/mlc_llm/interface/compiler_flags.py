"""Flags for overriding model config."""

import dataclasses
import enum
from io import StringIO
from typing import Optional

from mlc_llm.support import argparse, logging
from mlc_llm.support.config import ConfigOverrideBase

logger = logging.getLogger(__name__)


class IPCAllReduceStrategyType(enum.IntEnum):
    """The all-reduce strategy."""

    NONE = 0
    ONESHOT = 1
    TWOSHOT = 2
    AUTO = 3


@dataclasses.dataclass
class OptimizationFlags:
    """Optimization flags"""

    flashinfer: bool = False
    cublas_gemm: bool = False
    faster_transformer: bool = False
    cudagraph: bool = False
    cutlass: bool = False
    ipc_allreduce_strategy: IPCAllReduceStrategyType = IPCAllReduceStrategyType.NONE

    def __repr__(self) -> str:
        out = StringIO()
        print(f"flashinfer={int(self.flashinfer)}", file=out, end="")
        print(f";cublas_gemm={int(self.cublas_gemm)}", file=out, end="")
        print(f";faster_transformer={int(self.faster_transformer)}", file=out, end="")
        print(f";cudagraph={int(self.cudagraph)}", file=out, end="")
        print(f";cutlass={int(self.cutlass)}", file=out, end="")
        print(f";ipc_allreduce_strategy={self.ipc_allreduce_strategy.name}", file=out, end="")
        return out.getvalue().rstrip()

    @staticmethod
    def from_str(source: str) -> "OptimizationFlags":
        """Parse optimization flags from a string."""

        if source in OPT_FLAG_PRESET:
            return OPT_FLAG_PRESET[source]

        def boolean(value: str) -> bool:
            if value == "0":
                return False
            if value == "1":
                return True
            raise ValueError(f"Invalid boolean value: {value}")

        parser = argparse.ArgumentParser(description="optimization flags")
        parser.add_argument("--flashinfer", type=boolean, default=True)
        parser.add_argument("--cublas_gemm", type=boolean, default=False)
        parser.add_argument("--faster_transformer", type=boolean, default=False)
        parser.add_argument("--cudagraph", type=boolean, default=False)
        parser.add_argument("--cutlass", type=boolean, default=False)
        parser.add_argument(
            "--ipc_allreduce_strategy",
            type=str,
            choices=["NONE", "ONESHOT", "TWOSHOT", "AUTO"],
            default="NONE",
        )
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return OptimizationFlags(
            flashinfer=results.flashinfer,
            cublas_gemm=results.cublas_gemm,
            faster_transformer=results.faster_transformer,
            cudagraph=results.cudagraph,
            cutlass=results.cutlass,
            ipc_allreduce_strategy=IPCAllReduceStrategyType[results.ipc_allreduce_strategy],
        )

    def update(self, target, quantization) -> None:
        """Update optimization flags based on additional information."""

        def _flashinfer(target) -> bool:
            from mlc_llm.support.auto_target import (  # pylint: disable=import-outside-toplevel
                detect_cuda_arch_list,
            )

            if not self.flashinfer:
                return False
            if target.kind.name != "cuda":
                return False
            arch_list = detect_cuda_arch_list(target)
            for arch in arch_list:
                if arch < 80:
                    logger.warning("flashinfer is not supported on CUDA arch < 80")
                    return False
            return True

        def _cublas_gemm(target, quantization) -> bool:
            """correct cublas_gemm flag"""
            if not target.kind.name in ["cuda", "rocm"]:
                return False
            if not (
                quantization.name in ["q0f16", "q0bf16", "q0f32"]
                or "e4m3" in quantization.name
                or "e5m2" in quantization.name
            ):
                return False
            return self.cublas_gemm

        def _faster_transformer(target) -> bool:
            """correct faster_transformer flag"""
            if not target.kind.name == "cuda":
                return False
            return self.faster_transformer

        def _cutlass(target) -> bool:
            """correct cutlass flag"""
            if not target.kind.name == "cuda":
                return False
            return self.cutlass

        def _cudagraph(target) -> bool:
            """correct cudagraph flag"""
            if not target.kind.name == "cuda":
                return False
            return self.cudagraph

        self.flashinfer = _flashinfer(target)
        self.cublas_gemm = _cublas_gemm(target, quantization)
        self.faster_transformer = _faster_transformer(target)
        self.cutlass = _cutlass(target)
        self.cudagraph = _cudagraph(target)


@dataclasses.dataclass
class ModelConfigOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None
    disaggregation: Optional[bool] = None

    def __repr__(self) -> str:
        out = StringIO()
        print(f"context_window_size={self.context_window_size}", file=out, end="")
        print(f";sliding_window_size={self.sliding_window_size}", file=out, end="")
        print(f";prefill_chunk_size={self.prefill_chunk_size}", file=out, end="")
        print(f";attention_sink_size={self.attention_sink_size}", file=out, end="")
        print(f";max_batch_size={self.max_batch_size}", file=out, end="")
        print(f";tensor_parallel_shards={self.tensor_parallel_shards}", file=out, end="")
        print(f";pipeline_parallel_stages={self.pipeline_parallel_stages}", file=out, end="")
        print(f";disaggregation={self.disaggregation}", file=out, end="")
        return out.getvalue().rstrip()

    @staticmethod
    def from_str(source: str) -> "ModelConfigOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="model config override values")
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--sliding_window_size", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--attention_sink_size", type=int, default=None)
        parser.add_argument("--max_batch_size", type=int, default=None)
        parser.add_argument("--tensor_parallel_shards", type=int, default=None)
        parser.add_argument("--pipeline_parallel_stages", type=int, default=None)
        parser.add_argument(
            "--disaggregation",
            type=lambda x: (str(x).lower() in ["true", "1", "yes", "True"]),
            default=None,
        )
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ModelConfigOverride(
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            prefill_chunk_size=results.prefill_chunk_size,
            attention_sink_size=results.attention_sink_size,
            max_batch_size=results.max_batch_size,
            tensor_parallel_shards=results.tensor_parallel_shards,
            pipeline_parallel_stages=results.pipeline_parallel_stages,
            disaggregation=results.disaggregation,
        )


OPT_FLAG_PRESET = {
    "O0": OptimizationFlags(
        flashinfer=False,
        cublas_gemm=False,
        cudagraph=False,
    ),
    "O1": OptimizationFlags(
        flashinfer=False,
        cublas_gemm=True,
        faster_transformer=True,
        cudagraph=False,
        cutlass=True,
    ),
    "O2": OptimizationFlags(
        flashinfer=True,
        cublas_gemm=True,
        faster_transformer=False,
        cudagraph=True,
        cutlass=True,
        ipc_allreduce_strategy=IPCAllReduceStrategyType.NONE,
    ),
    "O3": OptimizationFlags(
        flashinfer=True,
        cublas_gemm=True,
        faster_transformer=True,
        cudagraph=True,
        cutlass=True,
        ipc_allreduce_strategy=IPCAllReduceStrategyType.AUTO,
    ),
}
