"""Optimization flags"""
import argparse
import dataclasses
from io import StringIO

from tvm.target import Target

from mlc_chat.support.logging import getLogger

logger = getLogger(__name__)


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

    def update(self, target: Target) -> None:
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

        self.flashinfer = _flashinfer(target)


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
