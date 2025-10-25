"""
This file specifies how MLC's Qwen2 VL parameters map from HuggingFace format.
"""

import functools
import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen2_vl_config import QWen2VLConfig
from .qwen2_vl_model import QWen2VLForCausalLM

def huggingface(model_config: QWen2VLConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping from MLC LLM parameters to HuggingFace parameters.

    Parameters
    ----------
    model_config : QWen2VLConfig
        The configuration of the Qwen2 VL model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = QWen2VLForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # Vision model mappings
    def _add_vision(mlc_name: str, hf_name: str = None):
        if hf_name is None:
            hf_name = mlc_name
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    # Vision transformer layers
    vision_prefix = "vision_model"
    for i in range(model_config.vision_config.num_hidden_layers):
        layer_prefix = f"{vision_prefix}.layers.{i}"
        _add_vision(f"{layer_prefix}.layernorm1.weight")
        _add_vision(f"{layer_prefix}.layernorm1.bias")
        _add_vision(f"{layer_prefix}.layernorm2.weight")
        _add_vision(f"{layer_prefix}.layernorm2.bias")
        
        # Attention weights
        _add_vision(f"{layer_prefix}.attention.q_proj.weight")
        _add_vision(f"{layer_prefix}.attention.q_proj.bias")
        _add_vision(f"{layer_prefix}.attention.k_proj.weight")
        _add_vision(f"{layer_prefix}.attention.k_proj.bias")
        _add_vision(f"{layer_prefix}.attention.v_proj.weight")
        _add_vision(f"{layer_prefix}.attention.v_proj.bias")
        _add_vision(f"{layer_prefix}.attention.o_proj.weight")
        _add_vision(f"{layer_prefix}.attention.o_proj.bias")
        
        # MLP weights
        _add_vision(f"{layer_prefix}.mlp.fc1.weight")
        _add_vision(f"{layer_prefix}.mlp.fc1.bias")
        _add_vision(f"{layer_prefix}.mlp.fc2.weight")
        _add_vision(f"{layer_prefix}.mlp.fc2.bias")

    # Vision embeddings and final layer norm
    _add_vision(f"{vision_prefix}.embeddings.patch_embed.weight")
    _add_vision(f"{vision_prefix}.embeddings.patch_embed.bias")
    _add_vision(f"{vision_prefix}.embeddings.pos_embed")
    _add_vision(f"{vision_prefix}.post_layernorm.weight")
    _add_vision(f"{vision_prefix}.post_layernorm.bias")

    # Vision projection
    _add_vision("vision_projection.linear_1.weight", "visual_proj.0.weight")
    _add_vision("vision_projection.linear_1.bias", "visual_proj.0.bias")
    _add_vision("vision_projection.linear_2.weight", "visual_proj.2.weight")
    _add_vision("vision_projection.linear_2.bias", "visual_proj.2.bias")

    # Language model mappings
    for i in range(model_config.num_hidden_layers):
        # Map attention weights
        attn = f"language_model.layers.{i}.self_attn"
        for weight_type in ["weight", "bias"]:
            mlc_name = f"{attn}.c_attn.{weight_type}"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{attn}.q_proj.{weight_type}",
                    f"{attn}.k_proj.{weight_type}",
                    f"{attn}.v_proj.{weight_type}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # Map MLP weights
        mlp = f"language_model.layers.{i}.mlp"
        mlc_name = f"{mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    # Map remaining parameters directly
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping
