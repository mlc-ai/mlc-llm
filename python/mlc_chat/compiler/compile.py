"""Python entrypoint of compilation."""
import dataclasses
import json
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tvm import IRModule, relax
from tvm.relax.frontend import nn
from tvm.target import Target

from ..support import logging
from ..support.config import ConfigBase
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
    system_lib_prefix: str
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
        print(f"  {bold('--system-lib-prefix'):<25} \"{self.system_lib_prefix}\"", file=out)
        print(f"  {bold('--output'):<25} {self.output}", file=out)
        print(f"  {bold('--overrides'):<25} {self.overrides}", file=out)
        print(out.getvalue().rstrip())


def _attach_auxiliary_methods(
    mod: IRModule,
    named_params: List[Tuple[str, nn.Parameter]],
    args: CompileArgs,
    model_config,
) -> None:
    def _get_memory_usage():
        return {str(k): int(v) for k, v in mod.attrs["mlc_llm.memory_usage"].items()}

    def _get_param_info():
        return [
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": param.dtype,
            }
            for name, param in named_params
        ]

    def _emit_metadata(metadata):
        bb = relax.BlockBuilder()  # pylint: disable=invalid-name
        with bb.function("main", params=[]):
            bb.emit_func_output(relax.StringImm(json.dumps(metadata)))
        return bb.finalize()["main"]

    metadata = {
        "quantization": args.quantization.name,
        "model_type": args.model.name,
        "memory_usage": _get_memory_usage(),
        "params": _get_param_info(),
        "context_window_size": model_config.context_window_size,
        "prefill_chunk_size": model_config.prefill_chunk_size,
        "sliding_window": getattr(model_config, "sliding_window", -1),
    }
    mod["_metadata"] = _emit_metadata(metadata)


def _attach_variable_bounds(mod, model_config):
    if hasattr(model_config, "sliding_window"):
        tir_bound_map = {
            "seq_len": model_config.prefill_chunk_size,
            "rolling_cache_len": model_config.sliding_window,
            "kv_seq_len": model_config.sliding_window + model_config.prefill_chunk_size,
        }
    else:
        tir_bound_map = {
            "seq_len": model_config.prefill_chunk_size,
            "total_seq_len": model_config.context_window_size,
        }
    for g_var, func in mod.functions_items():
        if isinstance(func, relax.Function):
            mod[g_var] = func.with_attr("tir_var_upper_bound", tir_bound_map)


def _compile(args: CompileArgs, model_config: ConfigBase):
    logger.info("Creating model from: %s", args.config)
    args.overrides.apply(model_config)
    model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
    logger.info("Exporting the model to TVM Unity compiler")
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    logger.info("Running optimizations using TVM Unity")
    _attach_variable_bounds(mod, model_config)
    with args.target:
        mod = relax.get_pipeline("mlc_llm")(mod)
    _attach_auxiliary_methods(mod, named_params, args, model_config)
    logger.info("Generating code using TVM Unity")
    args.build_func(mod, args)
    logger.info("Generated: %s", bold(str(args.output)))


def compile(  # pylint: disable=too-many-arguments,redefined-builtin
    config: Dict[str, Any],
    quantization: str,
    model_type: Model,
    target: Target,
    opt: OptimizationFlags,
    build_func: Callable[[IRModule, CompileArgs], None],
    system_lib_prefix: str,
    output: Path,
    overrides: ModelConfigOverride,
):
    """Compile a model given its configuration and quantization format to a specific target."""
    model_config = model_type.config.from_dict(config)
    args = CompileArgs(
        model_config,
        quantization,
        model_type,
        target,
        opt,
        build_func,
        system_lib_prefix,
        output,
        overrides,
    )
    args.display()
    _compile(args, model_config)
