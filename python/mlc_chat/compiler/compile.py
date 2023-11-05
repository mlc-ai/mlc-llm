"""Python entrypoint of compilation."""
import dataclasses
import logging
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

from tvm import IRModule, relax
from tvm.target import Target

from ..support.style import bold
from .flags_model_config_override import ModelConfigOverride
from .flags_optimization import OptimizationFlags
from .model import Model
from .quantization import Quantization

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CompileArgs:  # pylint: disable=too-many-instance-attributes
    """Arguments to MLC LLM's compiler."""

    config: Path
    quantization: Quantization
    model: Model
    target: Target
    opt: OptimizationFlags
    build_func: Callable[[IRModule, "CompileArgs"], None]
    prefix_symbols: str
    output: Path
    overrides: ModelConfigOverride

    def display(self) -> None:
        """Display the arguments to stdout."""
        out = StringIO()
        print(f"{bold('Compiling with arguments:')}", file=out)
        print(f"  {bold('--config'):<25} {self.config}", file=out)
        print(f"  {bold('--quantization'):<25} {self.quantization}", file=out)
        print(f"  {bold('--model-type'):<25} {self.model.name}", file=out)
        print(f"  {bold('--target'):<25} {self.target.export()}", file=out)
        print(f"  {bold('--opt'):<25} {self.opt}", file=out)
        print(f"  {bold('--prefix-symbols'):<25} \"{self.prefix_symbols}\"", file=out)
        print(f"  {bold('--output'):<25} {self.output}", file=out)
        print(f"  {bold('--overrides'):<25} {dataclasses.asdict(self.overrides)}", file=out)
        print(out.getvalue().rstrip())


def _compile(args: CompileArgs):
    logger.info("Creating model from: %s", args.config)
    model_config = args.model.config.from_file(args.config)
    args.overrides.apply(model_config)
    model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
    logger.info("Exporting the model to TVM Unity compiler")
    mod, _named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    logger.info("Running optimizations using TVM Unity")
    with args.target:
        mod = relax.get_pipeline("mlc_llm")(mod)
    logger.info("Generating code using TVM Unity")
    args.build_func(mod, args)
    logger.info("Code dumped to: %s", bold(str(args.output)))


def compile(  # pylint: disable=too-many-arguments,redefined-builtin
    config: Path,
    quantization: Quantization,
    model_type: Model,
    target: Target,
    opt: OptimizationFlags,
    build_func: Callable[[IRModule, CompileArgs], None],
    prefix_symbols: str,
    output: Path,
    max_sequence_length: Optional[int],
):
    """Compile a model given its configuration and quantization format to a specific target."""
    args = CompileArgs(
        config,
        quantization,
        model_type,
        target,
        opt,
        build_func,
        prefix_symbols,
        output,
        ModelConfigOverride(max_sequence_length=max_sequence_length),
    )
    args.display()
    _compile(args)
