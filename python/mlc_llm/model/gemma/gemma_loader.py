"""
This file specifies how MLC's Gemma parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gemma_model import GemmaConfig, GemmaForCausalLM


def huggingface(model_config: GemmaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GemmaConfig
        The configuration of the Gemma model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = GemmaForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        mlc_name = f"{attn}.qkv_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
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
        # Modify RMS layernorm weights, since Gemma model adds 1 to the weights
        # We add 1 to the weights here for efficiency purpose
        mlc_name = f"model.layers.{i}.input_layernorm.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.post_attention_layernorm.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    mlc_name = "model.norm.weight"
    mlc_param = named_parameters[mlc_name]
    mapping.add_mapping(
        mlc_name,
        [mlc_name],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=named_parameters[mlc_name].dtype,
        ),
    )

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
