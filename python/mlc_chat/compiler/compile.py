"""Python entrypoint of compilation."""
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
class CompileArgs:
    """Arguments to MLC LLM's compiler."""

    config: Path
    quantization: str
    model_type: Model
    target: Target
    opt: str
    build_func: Callable[[IRModule, "CompileArgs"], None]
    output: Path


def _echo_args(args: CompileArgs) -> None:
    out = StringIO()
    print(f"{bold('Compiling with arguments:')}", file=out)
    print(f"  {bold('--config'):<25} {args.config}", file=out)
    print(f"  {bold('--quantization'):<25} {args.quantization}", file=out)
    print(f"  {bold('--model-type'):<25} {args.model_type.name}", file=out)
    print(f"  {bold('--target'):<25} {args.target.export()}", file=out)
    print(f"  {bold('--opt'):<25} {args.opt}", file=out)
    print(f"  {bold('--output'):<25} {args.output}", file=out)
    print(out.getvalue().rstrip())


def compile(  # pylint: disable=too-many-arguments,redefined-builtin
    config: Path,
    quantization,
    model_type: Model,
    target: Target,
    opt,
    build_func: Callable[[IRModule, CompileArgs], None],
    output: Path,
):
    """Compile a model given its configuration and quantization format to a specific target."""
    args = CompileArgs(config, quantization, model_type, target, opt, build_func, output)
    _echo_args(args)
