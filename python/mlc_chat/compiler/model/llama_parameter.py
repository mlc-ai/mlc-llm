"""
This file specifies how MLC's Llama parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
from typing import Callable, Tuple, Dict, List

import numpy as np

import tvm
from tvm.runtime import NDArray

from ..parameter import ExternMapping, QuantizeMapping
from .llama_config import LlamaConfig
from .llama_model import LlamaForCasualLM


def huggingface(model_config: LlamaConfig, _) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : LlamaConfig
        The configuration of the Llama model.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = LlamaForCasualLM(model_config)
    _, named_params = model.export_tvm(spec=model.get_default_spec())
    parameter_names = {name for name, _ in named_params}

    param_map: Dict[str, List[str]] = {}
    map_func: Dict[str, Callable] = {}
    unused_params = set()

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        assert f"{attn}.qkv_proj.weight" in parameter_names
        map_func[f"{attn}.qkv_proj.weight"] = lambda q, k, v: np.concatenate([q, k, v], axis=0)
        param_map[f"{attn}.qkv_proj.weight"] = [
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
        ]
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        assert f"{mlp}.gate_up_proj.weight" in parameter_names
        map_func[f"{mlp}.gate_up_proj.weight"] = lambda gate, up: np.concatenate([gate, up], axis=0)
        param_map[f"{mlp}.gate_up_proj.weight"] = [
            f"{mlp}.gate_proj.weight",
            f"{mlp}.up_proj.weight",
        ]
        # inv_freq is not used in the model
        unused_params.add(f"{attn}.rotary_emb.inv_freq")

    for name in parameter_names:
        if name not in map_func:
            map_func[name] = lambda x: x
            param_map[name] = [name]
    return ExternMapping(param_map, map_func, unused_params)


def hf_torch_group_quantize(model_config: LlamaConfig, mode: str = "q4f16_1") -> QuantizeMapping:
    """Returns a parameter mapping that maps a parameter in MLC LLM's model
    definition to its eventual names and values after quantization.

    Parameters
    ----------
    model_config : LlamaConfig
        The configuration of the Llama model.

    Returns
    -------
    quantize_map : QuantizeMapping
        The parameter mapping from a parameter in MLC LLM's model definition to
        its eventual names and values after quantization.
    """
    bits = {
        "int4": 4,
        "int8": 8,
        "fp16": 16,
        "fp32": 32,
        "int32": 32,
        "uint32": 32,
    }

    def group_quantize(
        w: NDArray,
        quantize_dtype: str = "int4",
        storage_dtype: str = "uint32",
        group_size: int = 32,
        # symmetric: bool = True,
        # transpose: bool = True,
    ) -> Tuple[NDArray, NDArray]:
        # pylint: disable=too-many-locals
        def _pad_axis_by_factor(tensor: np.ndarray, axis: int, factor: int) -> np.ndarray:
            dim = int(tensor.shape[axis])
            if dim % factor == 0:
                return tensor
            pad_width = ((0, 0) for i in tensor.shape)
            pad_width[axis][1] = factor - (dim % factor)
            return np.pad(tensor, pad_width, mode="constant", constant_values=0)

        def _clip(
            x: np.ndarray,
            x_min: int,
            x_max: int,
            dtype: str,
        ) -> np.ndarray:
            return np.clip(x, a_min=x_min, a_max=x_max).astype(dtype)

        num_elem_per_storage = bits[storage_dtype] // bits[quantize_dtype]
        assert group_size % num_elem_per_storage == 0
        num_storage_units = (group_size + num_elem_per_storage - 1) // num_elem_per_storage

        # using numpy for now
        w = w.numpy()

        # Step 1. Tile `w`: [n, k'] -> [n, k, group_size]
        w = _pad_axis_by_factor(w, axis=1, factor=group_size)
        n, k = [int(v) for v in w.shape]  # pylint: disable=invalid-name
        assert k % group_size == 0, "Padding is not working properly"
        k = k // group_size
        w = w.reshape([n, k, group_size])

        # Step 2. Calculate
        if quantize_dtype.startswith("int"):
            max_int_value = (2 ** (bits[quantize_dtype] - 1)) - 1
            # 1) `scale`: [n, k, group_size] -> [n, k]
            scale = np.maximum(np.amax(w, axis=-1), 1e-4) / max_int_value
            # 2) `w`: w / scale

            w = _clip(
                np.round(w / scale[:, :, np.newaxis]).astype("int") + max_int_value,
                x_min=0,
                x_max=max_int_value * 2,
                dtype=storage_dtype,
            )
        else:
            raise NotImplementedError

        # Step 3. Compress `w` to every `num_elem_per_storage` elements
        res = np.zeros((n, k, num_storage_units), dtype=np.uint32)
        for i in range(n):
            for j in range(k):
                for m in range(num_storage_units):
                    for k in range(num_elem_per_storage):
                        res[i, j, m] += w[i, j, m * num_elem_per_storage + k] * 2**k
        return tvm.nd.array(res), tvm.nd.array(scale)
        # pylint: enable=too-many-locals

    assert mode == "q4f16_1", "Other mode not supported yet"
    model = LlamaForCasualLM(model_config)
    _, named_params = model.export_tvm(spec=model.get_default_spec())
    parameter_names = {name for name, _ in named_params}

    param_map: Dict[str, List[str]] = {}
    map_func: Dict[str, Callable] = {}
    for name in parameter_names:
        if "norm.weight" not in name:
            # skip these
            param_map[name] = [f"{name}_quantized", f"{name}_scale"]
            map_func[name] = lambda x: group_quantize(x, quantize_dtype="int4")
        else:
            param_map[name] = [name]
            map_func[name] = lambda x: [x]

    return QuantizeMapping(param_map, map_func)
