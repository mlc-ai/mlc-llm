"""Python entrypoint of weight conversion."""
import dataclasses
import logging
import math
from io import StringIO
from pathlib import Path

import numpy as np
from tvm.contrib import tvmjs
from tvm.runtime import Device, NDArray
from tvm.runtime import cpu as cpu_device
from tvm.target import Target

from mlc_chat.support import tqdm

from ..support.style import bold, green
from .loader import LOADER
from .model import Model
from .quantization import Quantization

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


def _echo_args(args: ConversionArgs) -> None:
    def _device_to_str(device: Device) -> str:
        return f"{Device.MASK2STR[device.device_type]}:{device.device_id}"

    out = StringIO()
    print(f"{bold('Weight conversion with arguments:')}", file=out)
    print(f"  {bold('--config'):<25} {args.config}", file=out)
    print(f"  {bold('--quantization'):<25} {args.quantization}", file=out)
    print(f"  {bold('--model-type'):<25} {args.model.name}", file=out)
    print(f"  {bold('--device'):<25} {_device_to_str(args.device)}", file=out)
    print(f"  {bold('--source'):<25} {args.source}", file=out)
    print(f"  {bold('--source-format'):<25} {args.source_format}", file=out)
    print(f"  {bold('--output'):<25} {args.output}", file=out)
    print(out.getvalue().rstrip())


def _convert_args(args: ConversionArgs) -> None:  # pylint: disable=too-many-locals
    # model config & quantization config
    model_config = args.model.config.from_file(args.config)
    model, quantize_map = args.model.quantize[args.quantization.kind](
        model_config, args.quantization
    )
    _, _named_params = model.export_tvm(spec=model.get_default_spec())  # type: ignore[attr-defined]
    named_params = dict(_named_params)

    def _check_param(name: str, param: NDArray):
        nonlocal named_params
        if name not in named_params:
            raise ValueError(f"Parameter not found in model: {name}")
        if name in param_dict:
            raise ValueError(f"Duplication: Parameter {name} already computed")
        expect_shape = tuple(int(x) for x in named_params[name].shape)
        expect_dtype = named_params[name].dtype
        actual_shape = tuple(int(x) for x in param.shape)
        actual_dtype = param.dtype
        if actual_shape != expect_shape:
            raise ValueError(
                f"Parameter {name} has shape {param.shape}, but expected {expect_shape}"
            )
        if actual_dtype != expect_dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {expect_dtype}"
            )
        del named_params[name]

    # load and quantize
    param_dict = {}
    total_bytes = 0.0
    total_params = 0
    with Target.from_device(args.device), tqdm.redirect():
        for name, param in LOADER[args.source_format](
            path=args.source,
            extern_param_map=args.model.source[args.source_format](model_config, args.quantization),
            quantize_param_map=quantize_map,
        ).load(device=args.device):
            _check_param(name, param)
            param = param.copyto(cpu_device())
            param_dict[name] = param
            total_bytes += math.prod(param.shape) * np.dtype(param.dtype).itemsize
            total_params += math.prod(param.shape)
    if named_params:
        raise ValueError(f"Parameter not found in source: {', '.join(named_params.keys())}")
    # dump to output directory
    tvmjs.dump_ndarray_cache(
        param_dict,
        str(args.output),
        meta_data={"ParamSize": len(param_dict)},
        encode_format="raw",
    )
    logger.info("%s to %s", green("Saved"), bold(str(args.output)))
    logger.info("%s: %.3f GB", green("Total parameter size"), total_bytes / (1024**3))
    logger.info("%s: %d", green("Total number of parameter tensors"), len(param_dict))
    logger.info(f"%s: {total_params:,}", green("Total number of parameters"))


def convert_weight(  # pylint: disable=too-many-arguments
    config: Path,
    quantization: Quantization,
    model: Model,
    device: Device,
    source: Path,
    source_format: str,
    output: Path,
):
    """MLC LLM's weight conversation and quantization flow."""
    args = ConversionArgs(config, quantization, model, device, source, source_format, output)
    _echo_args(args)
    _convert_args(args)
