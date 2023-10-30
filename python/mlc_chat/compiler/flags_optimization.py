"""Optimization flags"""
import argparse
import dataclasses
from io import StringIO


@dataclasses.dataclass
class OptimizationFlags:
    """Optiization flags"""

    cutlass_attn: bool = True
    cutlass_norm: bool = True
    cublas_gemm: bool = False
    cudagraph: bool = False

    def __repr__(self) -> str:
        out = StringIO()
        print(f"cutlass_attn={int(self.cutlass_attn)}", file=out, end="")
        print(f";cutlass_norm={int(self.cutlass_norm)}", file=out, end="")
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
        parser.add_argument("--cutlass_attn", type=boolean, default=True)
        parser.add_argument("--cutlass_norm", type=boolean, default=True)
        parser.add_argument("--cublas_gemm", type=boolean, default=False)
        parser.add_argument("--cudagraph", type=boolean, default=False)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return OptimizationFlags(
            cutlass_attn=results.cutlass_attn,
            cutlass_norm=results.cutlass_norm,
            cublas_gemm=results.cublas_gemm,
            cudagraph=results.cudagraph,
        )


OPT_FLAG_PRESET = {
    "O0": OptimizationFlags(
        cutlass_attn=False,
        cutlass_norm=False,
        cublas_gemm=False,
        cudagraph=False,
    ),
    "O1": OptimizationFlags(
        cutlass_attn=False,
        cutlass_norm=True,
        cublas_gemm=False,
        cudagraph=False,
    ),
    "O2": OptimizationFlags(
        cutlass_attn=True,
        cutlass_norm=True,
        cublas_gemm=False,
        cudagraph=False,
    ),
    "O3": OptimizationFlags(
        cutlass_attn=True,
        cutlass_norm=True,
        cublas_gemm=False,
        cudagraph=True,
    ),
}
