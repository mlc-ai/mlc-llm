"""Python entrypoint of compilation."""
import dataclasses
import math
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tvm import IRModule, relax, tir
from tvm.ir.transform import Pass, PassContext
from tvm.relax.frontend import nn
from tvm.target import Target

from mlc_chat import compiler_pass as _
from mlc_chat import op as op_ext
from mlc_chat.cli.model_metadata import _report_memory_usage
from mlc_chat.model import Model
from mlc_chat.quantization import Quantization
from mlc_chat.support import logging
from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

from .compiler_flags import ModelConfigOverride, OptimizationFlags

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
    debug_dump: Optional[Path]

    def __post_init__(self) -> None:
        self.opt.update(self.target, self.quantization)

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
        # As it's debug only, no need to display
        # print(f"  {bold('--debug-dump'):<25} {self.debug_dump}", file=out)
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
                "rolling_cache_len": model_config.sliding_window_size,
                "kv_seq_len": model_config.sliding_window_size + model_config.prefill_chunk_size,
                "seq_len": model_config.prefill_chunk_size,
                "batch_size": getattr(model_config, "max_batch_size", 1),
            }
        return {
            "total_seq_len": model_config.context_window_size,
            "seq_len": model_config.prefill_chunk_size,
            "batch_size": getattr(model_config, "max_batch_size", 1),
        }

    def _get_param_metadata(name: str, param: nn.Parameter) -> Dict[str, Any]:
        return {
            "name": name,
            # Record dynamic shape as -1 (e.g. vocab_size)
            "shape": [s if isinstance(s, int) else s.name for s in param.shape],
            "dtype": param.dtype,
            "preprocs": param.attrs["preprocs"],
        }

    def _find_kv_cache_bytes(model: nn.Module, model_config) -> int:
        all_kv_cache = nn.core._attribute_finder(  # pylint: disable=protected-access
            model,
            prefix="",
            condition_yield=lambda x: isinstance(x, nn.KVCache),
        )
        result = 0
        for _, kv_cache in all_kv_cache:
            result += math.prod(kv_cache.unit_shape) * np.dtype(kv_cache.dtype).itemsize
        if getattr(model_config, "sliding_window_size", -1) > 0:
            window_size = model_config.sliding_window_size
        elif getattr(model_config, "context_window_size", -1) > 0:
            window_size = model_config.context_window_size
        else:
            window_size = 0
        return result * window_size

    model_config = args.overrides.apply(model_config)
    with args.target:
        op_ext.enable(
            target=args.target,
            flashinfer=args.opt.flashinfer,
            faster_transformer=args.opt.faster_transformer,
        )
        # Step 1. Create the quantized model
        logger.info("Creating model from: %s", args.config)
        if (
            args.quantization.kind == "ft-quant"
            and hasattr(model_config, "tensor_parallel_shards")
            and model_config.tensor_parallel_shards > 1
        ):
            raise NotImplementedError
        if (
            hasattr(args.quantization, "linear_weight_layout")
            and args.quantization.linear_weight_layout == "KN"
            and hasattr(model_config, "tensor_parallel_shards")
            and model_config.tensor_parallel_shards > 1
        ):
            raise NotImplementedError(
                "KN layout (q3f16_0 and q4f16_0) is not supported for tensor parallelism"
            )
        model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
        kv_cache_bytes = _find_kv_cache_bytes(model, model_config)
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
        metadata = {
            "model_type": args.model.name,
            "quantization": args.quantization.name,
            "context_window_size": getattr(model_config, "context_window_size", -1),
            "sliding_window_size": getattr(model_config, "sliding_window_size", -1),
            "attention_sink_size": getattr(model_config, "attention_sink_size", -1),
            "prefill_chunk_size": model_config.prefill_chunk_size,  # type: ignore
            "tensor_parallel_shards": model_config.tensor_parallel_shards,  # type: ignore
            "kv_cache_bytes": kv_cache_bytes,
        }
        logger.info("Registering metadata: %s", metadata)
        metadata["params"] = [_get_param_metadata(name, param) for name, param in named_params]
        with PassContext(config={"relax.backend.use_cuda_graph": args.opt.cudagraph}):
            args.build_func(
                mod,
                args,
                pipeline=relax.get_pipeline(  # type: ignore
                    "mlc_llm",
                    target=args.target,
                    flashinfer=args.opt.flashinfer,
                    cublas_gemm=args.opt.cublas_gemm,
                    faster_transformer=args.opt.faster_transformer,
                    variable_bounds=variable_bounds,
                    additional_tirs=additional_tirs,
                    ext_mods=ext_mods,
                    metadata=metadata,
                    debug_dump=args.debug_dump,
                ),
            )
        _report_memory_usage(metadata=metadata, config=model_config)
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
    debug_dump: Optional[Path] = None,
):
    """Compile a model given its configuration and quantization format to a specific target."""
    if "model_config" in config:
        model_config = config.pop("model_config")
        model_config.update(config)
        model_config = model_type.config.from_dict(model_config)
    else:
        model_config = model_type.config.from_dict(config)
    model_config.kwargs = {}
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
        debug_dump,
    )
    args.display()
    _compile(args, model_config)
