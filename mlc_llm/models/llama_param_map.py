"""
This file specifies how MLC's Llama parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import numpy as np

from mlc_llm.param_loader import ParameterMapping

from .llama import LlamaConfig, LlamaForCasualLM


def hf_torch(model_config: LlamaConfig) -> ParameterMapping:
    """
    Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : LlamaConfig
        The configuration of the Llama model.

    Returns
    -------
    param_map : ParameterMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = LlamaForCasualLM(model_config)
    _, named_params = model.export_tvm(spec=model.get_default_spec())
    parameter_names = {name for name, _ in named_params}

    name_map = {}
    map_func = {}
    unused_params = set()

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        assert f"{attn}.qkv_proj.weight" in parameter_names
        map_func[f"{attn}.qkv_proj.weight"] = lambda q, k, v: np.concatenate([q, k, v], axis=0)
        name_map[f"{attn}.qkv_proj.weight"] = (
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
        )
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        assert f"{mlp}.gate_up_proj.weight" in parameter_names
        map_func[f"{mlp}.gate_up_proj.weight"] = lambda gate, up: np.concatenate([gate, up], axis=0)
        name_map[f"{mlp}.gate_up_proj.weight"] = (
            f"{mlp}.gate_proj.weight",
            f"{mlp}.up_proj.weight",
        )
        # inv_freq is not used in the model
        unused_params.add(f"{attn}.rotary_emb.inv_freq")

    for name in parameter_names:
        if name not in map_func:
            map_func[name] = lambda x: x
            name_map[name] = (name,)
    return ParameterMapping(name_map, map_func, unused_params)
