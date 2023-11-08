"""Python entrypoint of compilation."""
import dataclasses
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from tvm import IRModule, relax
from tvm.relax.frontend import nn
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


def _attach_auxiliary_methods(
    mod: IRModule,
    named_params: List[Tuple[str, nn.Parameter]],
    args: CompileArgs,
    model_config,
) -> None:
    def _metadata():
        metadata = {
            "quantization": args.quantization.name,
            "model_type": args.model.name,
            "memory_usage": {str(k): int(v) for k, v in mod.attrs["mlc_llm.memory_usage"].items()},
            "params": [
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": param.dtype,
                }
                for name, param in named_params
            ],
        }
        print(json.dumps(metadata, indent=2))
        bb = relax.BlockBuilder()  # pylint: disable=invalid-name
        with bb.function("main", params=[]):
            bb.emit_func_output(relax.StringImm(json.dumps(metadata)))
        return bb.get()["main"]

    def _attach_variable_bounds():
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                mod[g_var] = func.with_attr(
                    "tir_var_upper_bound",
                    {
                        "seq_len": model_config.max_sequence_length,
                        "total_seq_len": model_config.max_sequence_length,
                    },
                )

    mod["_metadata"] = _metadata()
    _attach_variable_bounds()


def _compile(args: CompileArgs):
    logger.info("Creating model from: %s", args.config)
    model_config = args.model.config.from_file(args.config)
    args.overrides.apply(model_config)
    model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
    logger.info("Exporting the model to TVM Unity compiler")
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    logger.info("Running optimizations using TVM Unity")
    with args.target:
        mod = relax.get_pipeline("mlc_llm")(mod)
    _attach_auxiliary_methods(mod, named_params, args, model_config)
    logger.info("Generating code using TVM Unity")
    args.build_func(mod, args)
    logger.info("Generated: %s", bold(str(args.output)))


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
