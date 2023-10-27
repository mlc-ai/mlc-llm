"""
Quantization specs for Llama2 architecture.
TODO: add docstring
"""
from typing import Callable, Dict, List, Optional

import tvm
from tvm.runtime import NDArray

from ..parameter import QuantizeMapping
from ..quantization import QuantizeConfig
from ..quantization.group_quantizer import te_quantize as te_group_quantize
from .llama_config import LlamaConfig
from .llama_model import LlamaForCasualLM


def huggingface_group_quantize(
    model_config: LlamaConfig,
    quantize_config: QuantizeConfig,
    target: Optional[tvm.target.Target] = None,
) -> QuantizeMapping:
    """Returns a parameter mapping that maps a parameter in MLC LLM's model
    definition to its eventual names and values after quantization.

    Parameters
    ----------
    model_config : LlamaConfig
        The configuration of the Llama model.
    quantize_config : GroupQuantizeConfig
        The configuration of the group quantization.
    target : Optional[tvm.target.Target]
        The target device to run the quantization on, by default None, which
        means the quantization will be run on CPU.

    Returns
    -------
    quantize_map : QuantizeMapping
        The parameter mapping from a parameter in MLC LLM's model definition to
        its eventual names and values after quantization.
    """

    def group_quantize(
        param: NDArray, config: QuantizeConfig, target: Optional[tvm.target.Target] = None
    ):
        if target is None or target.kind.name == "llvm":
            target = tvm.target.Target("llvm")
            device = tvm.cpu()
        elif target.kind.name == "cuda":
            device = tvm.cuda()
        else:
            raise ValueError(f"Invalid target device: {target}")
        param_tensor = tvm.te.placeholder(param.shape, dtype=param.dtype, name="param")
        weight_compute, scale_compute, other_computes = te_group_quantize(  # type: ignore
            param_tensor, config
        )
        s = tvm.te.create_schedule(
            [compute.op for compute in [weight_compute, scale_compute] + other_computes]
        )
        if target.kind.name == "cuda":
            # thread_binding for cuda
            for compute in [weight_compute, scale_compute] + other_computes:
                xo, xi = s[compute].split(compute.op.axis[0], 256)
                s[compute].bind(xo, tvm.te.thread_axis("blockIdx.x"))
                s[compute].bind(xi, tvm.te.thread_axis("threadIdx.x"))
        f_quantize = tvm.build(
            s, [param_tensor, weight_compute, scale_compute], name="group_quantize", target=target
        )
        weight = tvm.nd.empty(weight_compute.shape, weight_compute.dtype, device=device)
        scale = tvm.nd.empty(scale_compute.shape, scale_compute.dtype, device=device)
        f_quantize(param.copyto(device), weight, scale)
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
        if "norm.weight" not in name and "embed" not in name:
            param_map[name] = [f"{name}_quantized", f"{name}_scale"]
            map_func[name] = lambda x: group_quantize(x, quantize_config, target=target)
        else:
            # skip these parameters
            param_map[name] = [name]
            map_func[name] = lambda x: [x]

    return QuantizeMapping(param_map, map_func)
