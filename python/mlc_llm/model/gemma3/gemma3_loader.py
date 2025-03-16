"""
This file specifies how MLC's Gemma3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gemma3_model import Gemma3Config, Gemma3ForCausalLM


def huggingface(model_config: Gemma3Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Gemma3Config
        The configuration of the Gemma model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Gemma3ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    mlc_prefix = "language_model."
    hf_prefix = "language_model." if not model_config.is_text_model else ""
    for i in range(model_config.text_config.num_hidden_layers):
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{mlc_prefix + mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{hf_prefix + mlp}.gate_proj.weight",
                f"{hf_prefix + mlp}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        # Modify RMS layernorm weights, since Gemma model adds 1 to the weights
        # We add 1 to the weights here for efficiency purpose
        mlc_name = f"model.layers.{i}.input_layernorm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.post_attention_layernorm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.pre_feedforward_layernorm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.post_feedforward_layernorm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.self_attn.k_norm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

        mlc_name = f"model.layers.{i}.self_attn.q_norm.weight"
        mlc_param = named_parameters[mlc_prefix + mlc_name]
        mapping.add_mapping(
            mlc_prefix + mlc_name,
            [hf_prefix + mlc_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[mlc_prefix + mlc_name].dtype,
            ),
        )

    mlc_name = "model.norm.weight"
    mlc_param = named_parameters[mlc_prefix + mlc_name]
    mapping.add_mapping(
        mlc_prefix + mlc_name,
        [hf_prefix + mlc_name],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=named_parameters[mlc_prefix + mlc_name].dtype,
        ),
    )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [hf_prefix + mlc_name[len(mlc_prefix) :]],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping
