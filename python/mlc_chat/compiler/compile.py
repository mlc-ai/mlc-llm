"""Python entrypoint of compilation."""
import dataclasses
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tvm import IRModule, relax, tir
from tvm.ir.transform import Pass
from tvm.relax.frontend import nn
from tvm.target import Target

from ..support import logging
from ..support.config import ConfigBase
from ..support.style import bold
from . import extern
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
    build_func: Callable[[IRModule, "CompileArgs", Pass], None]
    system_lib_prefix: str
    output: Path
    overrides: ModelConfigOverride

    def __post_init__(self) -> None:
        self.opt.update(self.target)

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


def _apply_preproc_to_params(
    named_params: List[Tuple[str, nn.Parameter]],
    model_config,
) -> Dict[str, tir.PrimFunc]:
    extra_tirs: Dict[str, tir.PrimFunc] = {}
    for _, param in named_params:
        preprocs = param.attrs.get("preprocs", [])
        shard_strategy = param.attrs.get("shard_strategy", None)
        if shard_strategy is not None and model_config.tensor_parallel_shards > 1:
            preprocs.append(
                shard_strategy.gen_shard_info(
                    shards=model_config.tensor_parallel_shards,
                    weight=param,
                )
            )
            if shard_strategy.name not in extra_tirs:
                extra_tirs[shard_strategy.name] = shard_strategy.gen_tir(
                    shards=model_config.tensor_parallel_shards,
                    weight=param,
                )
        param.attrs["preprocs"] = preprocs
    return extra_tirs


def _compile(args: CompileArgs, model_config: ConfigBase):
    def _get_variable_bounds(model_config) -> Dict[str, int]:
        if hasattr(model_config, "sliding_window_size"):
            return {
                "seq_len": model_config.prefill_chunk_size,
                "rolling_cache_len": model_config.sliding_window_size,
                "kv_seq_len": model_config.sliding_window_size + model_config.prefill_chunk_size,
            }
        return {
            "seq_len": model_config.prefill_chunk_size,
            "total_seq_len": model_config.context_window_size,
        }

    def _get_param_metadata(name: str, param: nn.Parameter) -> Dict[str, Any]:
        return {
            "name": name,
            "shape": list(param.shape),
            "dtype": param.dtype,
            "preprocs": param.attrs["preprocs"],
        }

    args.overrides.apply(model_config)
    with args.target:
        extern.enable(
            target=args.target,
            flashinfer=args.opt.flashinfer,
        )
        # Step 1. Create the quantized model
        logger.info("Creating model from: %s", args.config)
        model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
        # Step 2. Exporting the model to TVM Unity
        logger.info("Exporting the model to TVM Unity compiler")
        mod, named_params, ext_mods = model.export_tvm(
            spec=model.get_default_spec(),  # type: ignore
            allow_extern=True,
        )
        # Step 3. Running relax compilation pipeline
        logger.info("Running optimizations using TVM Unity")
        additional_tirs = _apply_preproc_to_params(named_params, model_config)
        variable_bounds = _get_variable_bounds(model_config)
        args.build_func(
            mod,
            args,
            pipeline=relax.get_pipeline(
                "mlc_llm",
                variable_bounds=variable_bounds,
                additional_tirs=additional_tirs,
                ext_mods=ext_mods,
                metadata={
                    "model_type": args.model.name,
                    "quantization": args.quantization.name,
                    "params": [_get_param_metadata(name, param) for name, param in named_params],
                    "context_window_size": model_config.context_window_size,  # type: ignore
                    "prefill_chunk_size": model_config.prefill_chunk_size,  # type: ignore
                    "sliding_window_size": getattr(model_config, "sliding_window_size", -1),
                    "tensor_parallel_shards": model_config.tensor_parallel_shards,  # type: ignore
                },
            ),
        )
    logger.info("Generated: %s", bold(str(args.output)))


def compile(  # pylint: disable=too-many-arguments,redefined-builtin
    config: Dict[str, Any],
    quantization: Quantization,
    model_type: Model,
    target: Target,
    opt: OptimizationFlags,
    build_func: Callable[[IRModule, CompileArgs, Pass], None],
    system_lib_prefix: str,
    output: Path,
    overrides: ModelConfigOverride,
):
    """Compile a model given its configuration and quantization format to a specific target."""
    if "model_config" in config:
        model_config = model_type.config.from_dict({**config["model_config"], **config})
    else:
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
