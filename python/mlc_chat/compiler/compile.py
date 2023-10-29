"""Python entrypoint of compilation."""
import argparse
import dataclasses
import logging
from io import StringIO
from pathlib import Path
from typing import Callable

from mlc_chat.compiler.model import Model
from tvm import IRModule  # pylint: disable=wrong-import-order
from tvm.target import Target  # pylint: disable=wrong-import-order

from ..support.style import bold

logger = logging.getLogger(__name__)


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


@dataclasses.dataclass
class CompileArgs:  # pylint: disable=too-many-instance-attributes
    """Arguments to MLC LLM's compiler."""

    config: Path
    quantization: str
    model: Model
    target: Target
    opt: OptimizationFlags
    build_func: Callable[[IRModule, "CompileArgs"], None]
    prefix_symbols: str
    output: Path


def _echo_args(args: CompileArgs) -> None:
    out = StringIO()
    print(f"{bold('Compiling with arguments:')}", file=out)
    print(f"  {bold('--config'):<25} {args.config}", file=out)
    print(f"  {bold('--quantization'):<25} {args.quantization}", file=out)
    print(f"  {bold('--model-type'):<25} {args.model.name}", file=out)
    print(f"  {bold('--target'):<25} {args.target.export()}", file=out)
    print(f"  {bold('--opt'):<25} {args.opt}", file=out)
    print(f"  {bold('--output'):<25} {args.output}", file=out)
    print(out.getvalue().rstrip())


def compile(  # pylint: disable=too-many-arguments,redefined-builtin
    config: Path,
    quantization,
    model_type: Model,
    target: Target,
    opt: OptimizationFlags,
    build_func: Callable[[IRModule, CompileArgs], None],
    prefix_symbols: str,
    output: Path,
):
    """Compile a model given its configuration and quantization format to a specific target."""
    args = CompileArgs(
        config, quantization, model_type, target, opt, build_func, prefix_symbols, output
    )
    _echo_args(args)
    model_config = args.model.config.from_file(args.config)
    model = args.model.model(model_config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    mod.show(black_format=False)
    for name, param in named_params:
        print(f"{name}: {param.shape} {param.dtype}")


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
