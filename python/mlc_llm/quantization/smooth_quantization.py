"""The SmoothQuant config"""

from dataclasses import dataclass
from typing import List, Literal, Union, Dict, Any, Iterator, Tuple
from collections import OrderedDict
import numpy as np
import os

from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm import nd
from tvm import relax

from ..loader import QuantizeMapping
from mlc_llm.support.preshard import _sharded_param_name, _compile_shard_funcs

@dataclass
class SmoothQuantize:  # pylint: disable=too-many-instance-attributes
    name: str
    kind: str
    activation_dtype: Literal["int8", "e4m3_float8", "e5m2_float8"]
    weight_dtype: Literal["int8", "e4m3_float8", "e5m2_float8"]
    zero_point_dtype: Literal["int8", "float16", "float16"]
    accumulator_dtype: Literal["int32", "float32", "float32"]
    model_dtype: Literal["float16"]

    def __post_init__(self):
        return

    def quantize_weight(self, weight: NDArray) -> List[NDArray]:
        return []

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with using smooth quantization
        currently all conversions are performed using compilation passes.
        ToDo: apply final model conversion using this pass.

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """
        return model


def _split_array(arr, num: int):
    return np.split(arr.numpy(), num) if arr is not None else [None] * num


def _duplicate_array(arr, num: int):
    return [np.copy(arr.numpy()) for _ in range(num)] if arr is not None else [None] * num


def load_file(path):
    import json
    with open(path, 'r') as f:
        loaded_dict = json.load(f)
    return loaded_dict


def shard_smoothquant_params(tensor_parallel_shards, args) -> Iterator[Tuple[str, NDArray]]:
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)

    pth = args.statistics_path
    param_to_smooth_factor = load_file(path=os.path.join(pth, "smooth_scale2param.json"))
    param_to_scale = load_file(path=os.path.join(pth, "quantize_scale2param.json"))
    import tvm
    from tvm.contrib import tvmjs
    smoothing_factors_dict, _ = tvmjs.load_ndarray_cache(f"{pth}/smooth/", tvm.cpu())
    scales_dict, _ = tvmjs.load_ndarray_cache(f"{pth}/quantize/", tvm.cpu())

    smooth_0_quants = ["smq_q8i8f16_0", "smq_e4m3_float8_0", "smq_e5m2_float8_0"]
    for name, param in model.state_dict().items():
        smooth_factor_names = param_to_smooth_factor["prefill"].pop(name, None)
        scale_names = param_to_scale["prefill"].pop(name, None)
        shard_strategy = param.attrs.get("shard_strategy", None)
        if smooth_factor_names is not None and scale_names is not None:
            a_factor, w_factor = smooth_factor_names
            a_scale, w_scale, a_zp, w_zp = scale_names
            if shard_strategy is not None:
                if shard_strategy.dim == 0:
                    a_factors = _duplicate_array(smoothing_factors_dict[a_factor], tensor_parallel_shards)
                    w_factors = _duplicate_array(smoothing_factors_dict[w_factor], tensor_parallel_shards)
                    a_scales =  _duplicate_array(scales_dict[a_scale], tensor_parallel_shards)
                    w_scales =  _split_array(scales_dict[w_scale], tensor_parallel_shards)
                    a_zps =     _duplicate_array(scales_dict[a_zp], tensor_parallel_shards)
                    w_zps =     _split_array(scales_dict[w_zp], tensor_parallel_shards)
                else:
                    assert shard_strategy.dim == 1, f"Not supported shard.dim={shard_strategy.dim}"
                    a_factors = _split_array(smoothing_factors_dict[a_factor], tensor_parallel_shards)
                    w_factors = _split_array(smoothing_factors_dict[w_factor], tensor_parallel_shards)
                    a_scales =  _split_array(scales_dict[a_scale], tensor_parallel_shards)
                    w_scales =  _duplicate_array(scales_dict[w_scale], tensor_parallel_shards)
                    a_zps =     _split_array(scales_dict[a_zp], tensor_parallel_shards)
                    w_zps =     _duplicate_array(scales_dict[w_zp], tensor_parallel_shards)
                for shard_idx in range(tensor_parallel_shards):
                    yield _sharded_param_name(a_factor, shard_idx), a_factors[shard_idx]
                    yield _sharded_param_name(w_factor, shard_idx), w_factors[shard_idx]
                    if not args.quantization.name in smooth_0_quants:
                        yield _sharded_param_name(w_scale, shard_idx), w_scales[shard_idx]
                        yield _sharded_param_name(w_zp, shard_idx), w_zps[shard_idx]
            else:
                yield a_factor, smoothing_factors_dict[a_factor]
                yield w_factor, smoothing_factors_dict[w_factor]
                if not args.quantization.name in smooth_0_quants:
                    yield w_scale, scales_dict[w_scale]
                    yield w_zp, scales_dict[w_zp]


def _create_smoothquant_func(
    bb: relax.BlockBuilder, param: nn.Parameter, param_name: str, idx: int, tensor_parallel_shards: int, **smq_params
):
    def _create_func(
        func_name: str,
        bb: relax.BlockBuilder,
        param: nn.Parameter,
        smoothing_factor: Union[np.ndarray, nd.NDArray],
        scale: Union[np.ndarray, nd.NDArray],
        zp: Union[np.ndarray, nd.NDArray],
        dtype: str
    ):
        weight_var = relax.Var("weight", relax.TensorStructInfo(param.shape, param.dtype))
        with bb.function(name=func_name, params=[weight_var]):
            with bb.dataflow():
                if smoothing_factor is not None:
                    weight_var = bb.emit(relax.op.multiply(weight_var, relax.const(smoothing_factor)))
                if scale is not None and zp is not None:
                    weight_var = bb.emit(relax.op.quantize(weight_var, relax.const(scale), relax.const(zp), axis=-2, out_dtype=dtype))
                gv = bb.emit_output(weight_var)
            bb.emit_func_output(gv)

    func_names = []
    shard_strategy = param.attrs.get("shard_strategy", None)
    factor_param = smq_params.get("smoothing_factor")
    scale_param = smq_params.get("scale")
    zp_param = smq_params.get("zp")
    if tensor_parallel_shards == 1 or shard_strategy is None:
        func_name = f"convert_param_{idx}"
        func_names.append((param_name, func_name))
        _create_func(func_name, bb, param, factor_param, scale_param, zp_param, smq_params.get("quant_config").weight_dtype)
    else:
        if shard_strategy.dim == 0:
            factors = _duplicate_array(factor_param, tensor_parallel_shards) if factor_param else None
            scales =  _split_array(scale_param, tensor_parallel_shards)
            zps =     _split_array(zp_param, tensor_parallel_shards)
        else:
            assert shard_strategy.dim == 1, f"Not supported shard.dim={shard_strategy.dim}"
            factors = _split_array(factor_param, tensor_parallel_shards) if factor_param else None
            scales =  _duplicate_array(scale_param, tensor_parallel_shards)
            zps =     _duplicate_array(zp_param, tensor_parallel_shards)
        for shard_idx in range(tensor_parallel_shards):
            func_name = f"convert_param_{idx}_shard_{shard_idx}"
            func_names.append((_sharded_param_name(param_name, shard_idx), func_name))
            _create_func(func_name, bb, param, factors[shard_idx], scales[shard_idx], zps[shard_idx], dtype = smq_params.get("quant_config").weight_dtype)
    return func_names

def gen_smoothquant(named_params: Dict[str, nn.Parameter], tensor_parallel_shards: int, args: Any):
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)

    # verification if smooth parameters exist to determine if smoothing was used
    smooth_scales_file = f"{args.statistics_path}/smooth_scale2param.json"
    if os.path.isfile(smooth_scales_file):
        param_to_smooth_factor = load_file(path=smooth_scales_file)
    else:
        param_to_smooth_factor = None

    param_to_scale = load_file(path=f"{args.statistics_path}/quantize_scale2param.json")
    import tvm
    from tvm.contrib import tvmjs
    if param_to_smooth_factor:
        smoothing_factors_dict, _ = tvmjs.load_ndarray_cache(f"{args.statistics_path}/smooth/", tvm.cpu())
    else:
        smoothing_factors_dict = None
    scales_dict, _ = tvmjs.load_ndarray_cache(f"{args.statistics_path}/quantize/", tvm.cpu())

    bb = relax.BlockBuilder()
    param_to_smoothquant_func = {}
    for idx, (name, param) in enumerate(model.state_dict().items()):
        if  param_to_smooth_factor:
            smooth_factor_names = param_to_smooth_factor["prefill"].pop(name, None)
        scale_names = param_to_scale["prefill"].pop(name, None)
        if (param_to_smooth_factor is None or smooth_factor_names is not None) and scale_names is not None:
            if  param_to_smooth_factor:
                _, smooth_factor_name = smooth_factor_names
            _, scale_name, _, zp_name = scale_names
            func_names = _create_smoothquant_func(
                bb,
                param,
                name,
                idx,
                tensor_parallel_shards,
                smoothing_factor=smoothing_factors_dict[smooth_factor_name] if param_to_smooth_factor else None,
                scale=scales_dict[scale_name],
                zp=scales_dict[zp_name],
                quant_config = args.quantization,
                dtype = args.quantization.model_dtype
            )
            for sharded_param_name, func_name in func_names:
                param_to_smoothquant_func[sharded_param_name] = func_name

                named_params[sharded_param_name].to(args.quantization.weight_dtype)  # Update dtype for checker

    assert not param_to_scale["prefill"], "detected not processed scales/zero_points"

    mod = bb.finalize()
    vm = _compile_shard_funcs(mod, args.device)

    for name in param_to_smoothquant_func:
        param_to_smoothquant_func[name] = vm[param_to_smoothquant_func[name]]
    return param_to_smoothquant_func
