from typing import Dict, List, Callable

import tvm
from tvm.runtime import NDArray

from .llama_config import LlamaConfig
from .llama_model import LlamaForCasualLM
from ..parameter import QuantizeMapping
from ..quantization import QuantizeConfig

from ..quantization.group_quantizer import te_quantize as te_group_quantize


def huggingface_group_quantize(
    model_config: LlamaConfig, quantize_config: QuantizeConfig
) -> QuantizeMapping:
    """Returns a parameter mapping that maps a parameter in MLC LLM's model
    definition to its eventual names and values after quantization.

    Parameters
    ----------
    model_config : LlamaConfig
        The configuration of the Llama model.
    quantize_config : GroupQuantizeConfig
        The configuration of the group quantization.

    Returns
    -------
    quantize_map : QuantizeMapping
        The parameter mapping from a parameter in MLC LLM's model definition to
        its eventual names and values after quantization.
    """

    def group_quantize(param: NDArray, config: QuantizeConfig):
        param_tensor = tvm.te.placeholder(param.shape, dtype=param.dtype, name="param")
        weight_compute, scale_compute = te_group_quantize(param_tensor, config)
        f_quantize = tvm.build(
            tvm.te.create_schedule([weight_compute.op, scale_compute.op]),
            [param_tensor, weight_compute, scale_compute],
            name="group_quantize",
        )
        weight = tvm.nd.empty(weight_compute.shape, weight_compute.dtype)
        scale = tvm.nd.empty(scale_compute.shape, scale_compute.dtype)
        f_quantize(param, weight, scale)
        return weight, scale

    # Param check
    assert (
        quantize_config.kind == "group_quantize"
    ), f"Invalid quantization config: group quantization expected but got {quantize_config.kind}"
    assert (
        quantize_config.name == "q4f16_1"
    ), """Only support q4f16_1 quantization scheme for now."""

    # Fetch model parameter & names
    model = LlamaForCasualLM(model_config)
    _, named_params = model.export_tvm(spec=model.get_default_spec())
    parameter_names = {name for name, _ in named_params}

    # Init mappings
    param_map: Dict[str, List[str]] = {}
    map_func: Dict[str, Callable] = {}

    # Dispatch quantization scheme
    # Also see https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/quantization/__init__.py
    for name in parameter_names:
        if "norm.weight" not in name:
            param_map[name] = [f"{name}_quantized", f"{name}_scale"]
            map_func[name] = lambda x: group_quantize(x, quantize_config)
        else:
            # skip these parameters
            param_map[name] = [name]
            map_func[name] = lambda x: [x]

    return QuantizeMapping(param_map, map_func)
