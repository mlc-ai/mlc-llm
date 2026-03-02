"""Python entrypoint of weight conversion."""

import contextlib
import dataclasses
import math
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from tvm import tir
from tvm.contrib import tvmjs
from tvm.runtime import DataType, Device, Tensor
from tvm.runtime import cpu as cpu_device
from tvm.target import Target

from mlc_llm.loader import LOADER
from mlc_llm.model import Model
from mlc_llm.quantization import Quantization
from mlc_llm.support import logging, tqdm
from mlc_llm.support.auto_weight import detect_weight
from mlc_llm.support.preshard import apply_preshard
from mlc_llm.support.style import bold, green

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConversionArgs:  # pylint: disable=too-many-instance-attributes
    """Arguments to MLC LLM's weight conversation and quantization flow."""

    config: Path
    quantization: Quantization
    model: Model
    device: Device
    source: Path
    source_format: str
    output: Path
    lora_adapter: Optional[Path] = None

    def display(self) -> None:
        """Display the arguments to stdout."""

        def _device_to_str(device: Device) -> str:
            return f"{Device._DEVICE_TYPE_TO_NAME[device.dlpack_device_type()]}:{device.index}"  # pylint: disable=protected-access, line-too-long

        out = StringIO()
        print(f"{bold('Weight conversion with arguments:')}", file=out)
        print(f"  {bold('--config'):<25} {self.config}", file=out)
        print(f"  {bold('--quantization'):<25} {self.quantization}", file=out)
        print(f"  {bold('--model-type'):<25} {self.model.name}", file=out)
        print(f"  {bold('--device'):<25} {_device_to_str(self.device)}", file=out)
        print(f"  {bold('--source'):<25} {self.source}", file=out)
        print(f"  {bold('--source-format'):<25} {self.source_format}", file=out)
        print(f"  {bold('--output'):<25} {self.output}", file=out)
        if self.lora_adapter is not None:
            print(f"  {bold('--lora-adapter'):<25} {self.lora_adapter}", file=out)
        print(out.getvalue().rstrip())


def _resolve_base_model_dir(source: Path) -> Path:
    return source if source.is_dir() else source.parent


@contextlib.contextmanager
def _merge_lora_adapter_with_base_model(base_source: Path, lora_adapter: Path) -> Iterator[Path]:
    base_model_dir = _resolve_base_model_dir(base_source)
    if not base_model_dir.exists():
        raise ValueError(f"Base model directory does not exist: {base_model_dir}")
    if not lora_adapter.exists() or not lora_adapter.is_dir():
        raise ValueError(f"LoRA adapter directory does not exist: {lora_adapter}")

    try:
        # pylint: disable=import-outside-toplevel
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        # pylint: enable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "`--lora-adapter` requires `peft` and `transformers` to be installed."
        ) from err

    with tempfile.TemporaryDirectory() as temp_dir:
        merged_model_dir = Path(temp_dir) / "merged_model"
        logger.info("Merging LoRA adapter %s into base model %s", lora_adapter, base_model_dir)

        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_dir),
            torch_dtype="auto",
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        merged_model = PeftModel.from_pretrained(
            base_model, str(lora_adapter), is_trainable=False
        ).merge_and_unload()
        merged_model.save_pretrained(str(merged_model_dir), safe_serialization=True)
        yield merged_model_dir


def _convert_args(args: ConversionArgs) -> None:  # pylint: disable=too-many-locals
    pre_shards_num = os.getenv("MLC_INTERNAL_PRESHARD_NUM")
    # model config & quantization config
    model_config = args.model.config.from_file(args.config)
    if (
        args.quantization.kind == "ft-quant"
        and hasattr(model_config, "tensor_parallel_shards")
        and model_config.tensor_parallel_shards > 1
    ):
        raise NotImplementedError
    if pre_shards_num is not None:
        model_config.tensor_parallel_shards = int(pre_shards_num)
    model, quantize_map = args.model.quantize[args.quantization.kind](
        model_config, args.quantization
    )
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),  # type: ignore[attr-defined]
        allow_extern=True,
    )
    named_params = dict(_named_params)

    if pre_shards_num is not None:
        named_params, preshard_funcs = apply_preshard(named_params, int(pre_shards_num), args)
    else:
        preshard_funcs = None

    def _check_param(name: str, param: Tensor):
        nonlocal named_params
        if name not in named_params:
            raise ValueError(f"Parameter not found in model: {name}")
        if name in param_names:
            raise ValueError(f"Duplication: Parameter {name} already computed")

        # Check shape (possibly dynamic)
        def _check_shape(actual: tuple, expect: tuple):  # expect can have tir.Var
            if len(actual) != len(expect):
                return False
            for actual_i, expect_i in zip(actual, expect):
                assert isinstance(expect_i, (int, tir.Var))
                if isinstance(expect_i, int) and actual_i != expect_i:
                    return False
            return True

        expect_shape = named_params[name].shape
        actual_shape = param.shape
        if not _check_shape(actual_shape, expect_shape):
            raise ValueError(
                f"Parameter {name} has shape {param.shape}, but expected {expect_shape}"
            )
        # Check dtype
        actual_dtype = param.dtype
        expect_dtype = named_params[name].dtype
        if actual_dtype != expect_dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {expect_dtype}"
            )
        del named_params[name]

    # load and quantize
    param_names = set()
    total_bytes = 0.0
    total_params: int

    def _param_generator() -> Iterator[Tuple[str, Tensor]]:
        nonlocal total_params, total_bytes
        with Target.from_device(args.device), tqdm.redirect():
            loader = LOADER[args.source_format](
                path=args.source,
                extern_param_map=args.model.source[args.source_format](
                    model_config, args.quantization
                ),
                quantize_param_map=quantize_map,
            )
            for name, param in loader.load(device=args.device, preshard_funcs=preshard_funcs):
                _check_param(name, param)
                param_names.add(name)
                param = param.copyto(cpu_device())
                total_bytes += math.prod(param.shape) * DataType(param.dtype).itemsize
                yield name, param
        total_params = loader.stats.total_param_num

    def _metadata_callback() -> Dict[str, Any]:
        return {
            "ParamSize": len(param_names),
            "ParamBytes": total_bytes,
            "BitsPerParam": total_bytes * 8.0 / total_params,
        }

    # dump to output directory
    tvmjs.dump_tensor_cache(
        _param_generator(),
        str(args.output),
        meta_data=_metadata_callback,
        encode_format="f32-to-bf16",
        show_progress=False,
    )
    if named_params:
        raise ValueError(f"Parameter not found in source: {', '.join(named_params.keys())}")
    # Log necessary statistics
    logger.info(
        "%s after quantization: %.3f GB",
        green("Parameter size"),
        total_bytes / (1024**3),
    )
    logger.info(f"%s: {total_params:,}", green("Total parameters"))
    logger.info(
        "%s: %.3f",
        green("Bits per parameter"),
        total_bytes * 8.0 / total_params,
    )
    logger.info("Saved to directory: %s", bold(str(args.output)))


def convert_weight(  # pylint: disable=too-many-arguments
    config: Path,
    quantization: Quantization,
    model: Model,
    device: Device,
    source: Path,
    source_format: str,
    output: Path,
    lora_adapter: Optional[Path] = None,
):
    """MLC LLM's weight conversation and quantization flow."""
    args = ConversionArgs(
        config, quantization, model, device, source, source_format, output, lora_adapter
    )

    allowed_lora_source_formats = {"huggingface-safetensor", "huggingface-torch"}
    if lora_adapter is not None and source_format not in allowed_lora_source_formats:
        raise ValueError(
            "`--lora-adapter` only supports source formats: "
            f"{sorted(allowed_lora_source_formats)}"
        )

    if lora_adapter is not None:
        with _merge_lora_adapter_with_base_model(source, lora_adapter) as merged_model_dir:
            merged_source, merged_source_format = detect_weight(
                weight_path=merged_model_dir,
                config_json_path=config,
                weight_format="auto",
            )
            merged_args = dataclasses.replace(
                args, source=merged_source, source_format=merged_source_format
            )
            merged_args.display()
            _convert_args(merged_args)
            return

    args.display()
    _convert_args(args)
