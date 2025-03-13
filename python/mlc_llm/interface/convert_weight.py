"""Python entrypoint of weight conversion."""

import dataclasses
import math
import os
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Optional
import re
import copy

import tvm
from tvm import tir
from tvm.contrib import tvmjs
from tvm.runtime import DataType, Device, NDArray
from tvm.runtime import cpu as cpu_device
from tvm.target import Target

from mlc_llm.loader import LOADER
from mlc_llm.model import Model
from mlc_llm.quantization import Quantization
from mlc_llm.support import logging, tqdm
from mlc_llm.support.preshard import apply_preshard
from mlc_llm.support.style import bold, green
from mlc_llm.lora.lora_config import LoRAConfig
from mlc_llm.lora.lora import set_lora
from mlc_llm.loader import ExternMapping

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
    lora_paths: Optional[Dict[str, Path]]
    lora_only: bool
    output: Path

    def display(self) -> None:
        """Display the arguments to stdout."""

        def _device_to_str(device: Device) -> str:
            return f"{Device.MASK2STR[device.device_type]}:{device.device_id}"

        out = StringIO()
        print(f"{bold('Weight conversion with arguments:')}", file=out)
        print(f"  {bold('--config'):<25} {self.config}", file=out)
        print(f"  {bold('--quantization'):<25} {self.quantization}", file=out)
        print(f"  {bold('--model-type'):<25} {self.model.name}", file=out)
        print(f"  {bold('--device'):<25} {_device_to_str(self.device)}", file=out)
        print(f"  {bold('--source'):<25} {self.source}", file=out)
        print(f"  {bold('--source-format'):<25} {self.source_format}", file=out)
        print(f"  {bold('--lora-paths'):<25} {self.lora_paths}", file=out)
        print(f"  {bold('--lora-only'):<25} {self.lora_only}", file=out)
        print(f"  {bold('--output'):<25} {self.output}", file=out)
        print(out.getvalue().rstrip())


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
    if args.lora_paths:
        set_lora(model, model_config, args.lora_paths, args.quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),  # type: ignore[attr-defined]
        allow_extern=True,
    )
    base_named_params = {name:param for name, param in _named_params if "lora_" not in name}
    lora_named_params = {name:param for name, param in _named_params if "lora_" in name}

    if pre_shards_num is not None:
        base_named_params, preshard_funcs = apply_preshard(base_named_params, int(pre_shards_num), args)
    else:
        preshard_funcs = None

    def _check_param(named_params, name: str, param: NDArray):
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

        # lora params is stacked and skip check first dimension max_loras_per_batch
        expect_shape = named_params[name].shape if "lora_" not in name else named_params[name].shape[1:]
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

    # Use base model extern map as lora extern map
    def get_lora_extern_map(lora_config: LoRAConfig):
        base_model_param_map = args.model.source[args.source_format](model_config, args.quantization)
        hf_name_params = lora_config.get_weights()
        param_map = {}
        map_func = {}
        attn_mlp_pattern = re.compile(
            r'(.*\.layers\.([0-9]+)\.(self_attn|mlp)\.([a-z_]+))\.(?:lora_(?:(A|B)\.weight|(magnitude)_vector)|weight_(m_wdecomp)\.weight).*'
        )
        for mlc_name, hf_names in base_model_param_map.param_map.items():
            for key in hf_name_params.keys():
                attn_nlp_m = attn_mlp_pattern.match(key)
                if attn_nlp_m:
                    layer_index = attn_nlp_m.group(2)
                    hf_module = attn_nlp_m.group(4)
                    A_or_B = attn_nlp_m.group(5)
                    for hf_name in hf_names:
                        if hf_module in hf_name and f'layers.{layer_index}.' in mlc_name:
                            new_mlc_name = mlc_name.replace("weight", f'lora_{A_or_B}.weight')
                            if new_mlc_name in param_map:
                                param_map[new_mlc_name].append(key)
                            else:
                                param_map[new_mlc_name] = [key]
                            map_func[new_mlc_name] = base_model_param_map.map_func[mlc_name]
                            break
                else:
                    raise ValueError(f'Unsupported lora_param: {key}')

        extern_map = ExternMapping()
        extern_map.param_map = param_map
        extern_map.map_func = map_func
        return extern_map

    def _param_generator(source_format, source, extern_param_map, quantize_param_map, named_params, lora_config) -> Iterator[Tuple[str, NDArray]]:
        nonlocal total_params, total_bytes
        total_bytes = 0
        with Target.from_device(args.device), tqdm.redirect():
            loader = LOADER[source_format](
                path=source,
                extern_param_map=extern_param_map,
                quantize_param_map=quantize_param_map,
            )
            for name, param in loader.load(device=args.device, preshard_funcs=preshard_funcs):
                _check_param(named_params, name, param)
                param_names.add(name)
                param = param.copyto(cpu_device())
                total_bytes += math.prod(param.shape) * DataType(param.dtype).itemsize()
                if lora_config:
                    # Optimize the LoRA by merging the scaling into lora b
                    scaled_param = param.numpy() * lora_config.scaling if ".lora_B" in name else param.numpy()
                    param = tvm.nd.empty(param.shape, param.dtype, device=cpu_device())
                    param.copyfrom(scaled_param)
                yield name, param
        total_params = loader.stats.total_param_num

    def _metadata_callback() -> Dict[str, Any]:
        return {
            "ParamSize": len(param_names),
            "ParamBytes": total_bytes,
            "BitsPerParam": total_bytes * 8.0 / total_params,
        }

    if args.lora_paths:
        lora_r = None
        for lora_name, path in args.lora_paths.items():
            param_names.clear()
            lora_config = LoRAConfig(path)
            # Only support lora with the same rank
            if lora_r:
                assert lora_r == lora_config.r
            else:
                lora_r = lora_config.r
            lora_named_params_copy = copy.deepcopy(lora_named_params)
            tvmjs.dump_ndarray_cache(
                _param_generator(lora_config.source_format, lora_config.source, get_lora_extern_map(lora_config), None, lora_named_params_copy, lora_config),
                str(args.output),
                meta_data=_metadata_callback,
                encode_format="f32-to-bf16",
                show_progress=False,
                file_name_prefix=lora_name + "-",
            )
            if lora_named_params_copy:
                raise ValueError(f"Parameter not found in source: {', '.join(lora_named_params_copy.keys())}")
    
    if args.lora_only is False:
        # dump to output directory
        tvmjs.dump_ndarray_cache(
            _param_generator(args.source_format, args.source, args.model.source[args.source_format](model_config, args.quantization), quantize_map, base_named_params, None),
            str(args.output),
            meta_data=_metadata_callback,
            encode_format="f32-to-bf16",
            show_progress=False,
        )
        if base_named_params:
            raise ValueError(f"Parameter not found in source: {', '.join(base_named_params.keys())}")

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
    lora_paths: Optional[Dict[str, Path]],
    lora_only: bool,
    output: Path,
):
    """MLC LLM's weight conversation and quantization flow."""
    args = ConversionArgs(config, quantization, model, device, source, source_format, lora_paths, lora_only, output)
    args.display()
    _convert_args(args)
