"""Flags for overriding model config."""
import dataclasses
from io import StringIO
from typing import Any, Optional

from mlc_chat.support import argparse, logging
from mlc_chat.support.style import bold, red

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OptimizationFlags:
    """Optimization flags"""

    flashinfer: bool = False
    cublas_gemm: bool = False
    cudagraph: bool = False

    def __repr__(self) -> str:
        out = StringIO()
        print(f"flashinfer={int(self.flashinfer)}", file=out, end="")
        print(f";cublas_gemm={int(self.cublas_gemm)}", file=out, end="")
        print(f";cudagraph={int(self.cudagraph)}", file=out, end="")
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
        parser.add_argument("--cudagraph", type=boolean, default=False)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return OptimizationFlags(
            flashinfer=results.flashinfer,
            cublas_gemm=results.cublas_gemm,
            cudagraph=results.cudagraph,
        )

    def update(self, target, quantization) -> None:
        """Update optimization flags based on additional information."""

        def _flashinfer(target) -> bool:
            from mlc_chat.support.auto_target import (  # pylint: disable=import-outside-toplevel
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
            if not (target.kind.name == "cuda" and quantization.name in ["q0f16", "q0f32"]):
                return False
            return self.cublas_gemm

        self.flashinfer = _flashinfer(target)
        self.cublas_gemm = _cublas_gemm(target, quantization)


@dataclasses.dataclass
class ModelConfigOverride:
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None

    def __repr__(self) -> str:
        out = StringIO()
        print(f"context_window_size={self.context_window_size}", file=out, end="")
        print(f";sliding_window_size={self.sliding_window_size}", file=out, end="")
        print(f";prefill_chunk_size={self.prefill_chunk_size}", file=out, end="")
        print(f";attention_sink_size={self.attention_sink_size}", file=out, end="")
        print(f";max_batch_size={self.max_batch_size}", file=out, end="")
        print(f";tensor_parallel_shards={self.tensor_parallel_shards}", file=out, end="")
        return out.getvalue().rstrip()

    def __post_init__(self):
        # If `sliding_window_size` is set
        # - 1) Disable `context_window_size`
        # - 2) Require `prefill_chunk_size` to present
        # - 3) Set `attention_sink_size` to default (4)
        if self.sliding_window_size is not None:
            self.context_window_size = -1
            logger.info(
                "Setting %s to -1 (disabled), because %s is already set",
                bold("context_window_size"),
                bold("sliding_window_size"),
            )
            if self.prefill_chunk_size is None:
                logger.info(
                    "Default %s to %s (%d) because it is not provided",
                    bold("prefill_chunk_size"),
                    bold("sliding_window_size"),
                    self.sliding_window_size,
                )
                self.prefill_chunk_size = self.sliding_window_size
            if self.attention_sink_size is None:
                logger.info(
                    "Default %s to %d because it is not provided",
                    bold("attention_sink_size"),
                    4,
                )
                self.attention_sink_size = 4
        elif self.context_window_size is not None:
            if self.prefill_chunk_size is None:
                logger.info(
                    "Default %s to %s (%d) because it is not provided",
                    bold("prefill_chunk_size"),
                    bold("context_window_size"),
                    self.context_window_size,
                )
                self.prefill_chunk_size = self.context_window_size

    def apply(self, model_config):
        """Apply the overrides to the given model config."""
        if self.context_window_size is not None:
            _model_config_override(model_config, "context_window_size", self.context_window_size)
        if self.sliding_window_size is not None:
            _model_config_override(model_config, "sliding_window_size", self.sliding_window_size)
        if self.prefill_chunk_size is not None:
            _model_config_override(model_config, "prefill_chunk_size", self.prefill_chunk_size)
        if self.attention_sink_size is not None:
            _model_config_override(model_config, "attention_sink_size", self.attention_sink_size)
        if self.max_batch_size is not None:
            _model_config_override(model_config, "max_batch_size", self.max_batch_size)
        if self.tensor_parallel_shards is not None:
            _model_config_override(
                model_config, "tensor_parallel_shards", self.tensor_parallel_shards
            )

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
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ModelConfigOverride(
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            prefill_chunk_size=results.prefill_chunk_size,
            attention_sink_size=results.attention_sink_size,
            max_batch_size=results.max_batch_size,
            tensor_parallel_shards=results.tensor_parallel_shards,
        )


def _model_config_override(model_config, field: str, value: Any) -> None:
    if hasattr(model_config, field):
        logger.info(
            "Overriding %s from %d to %d",
            bold(field),
            getattr(model_config, field),
            value,
        )
        setattr(model_config, field, value)
    else:
        logger.warning(
            "%s: %s does not have %s",
            red("Warning"),
            bold(type(model_config).__name__),
            bold(field),
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
        cudagraph=False,
    ),
    "O2": OptimizationFlags(
        flashinfer=False,
        cublas_gemm=True,
        cudagraph=False,
    ),
    "O3": OptimizationFlags(
        flashinfer=True,
        cublas_gemm=True,
        cudagraph=True,
    ),
}
