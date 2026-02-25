"""
This file specifies how MLC's QWen2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Callable, List, Literal

import numpy as np

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization

from .qwen3_model import Qwen3Config, Qwen3LMHeadModel


def huggingface(
    model_config: Qwen3Config,
    quantization: Quantization,
    hf_prefix: Literal["", "model."] = "model.",
) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Qwen3Config
        The configuration of the Qwen3 model.

    quantization : Quantization
        The quantization configuration.

    hf_prefix : Literal["", "model."]
        Prefix used in HuggingFace weight names. Defaults to "model." for standard
        Qwen3 models. Use "" for Qwen3-Embedding models without prefix.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Qwen3LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    if isinstance(quantization, BlockScaleQuantize):
        model = quantization.quantize_model(model, QuantizeMapping({}, {}), "")
        if model_config.weight_block_size is None:
            raise ValueError(
                "The input Qwen3 model is not fp8 block quantized. "
                "Thus BlockScaleQuantize is not supported."
            )

    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    if (
        not isinstance(quantization, BlockScaleQuantize)
        and model_config.weight_block_size is not None
    ):
        raise ValueError(
            "The input Qwen3 model is fp8 block quantized. "
            "Please use BlockScaleQuantize for the model."
        )

    def to_hf(name: str) -> str:
        if hf_prefix == "model.":
            return name
        return name[6:] if name.startswith("model.") else name

    def add_weight_and_scale_mapping(
        weight_mlc_name: str,
        weight_hf_names: List[str],
        weight_transform_func: Callable,
    ):
        mlc_param = named_parameters[weight_mlc_name]
        hf_names = [to_hf(name) for name in weight_hf_names]
        mapping.add_mapping(
            weight_mlc_name,
            hf_names,
            functools.partial(weight_transform_func, dtype=mlc_param.dtype),
        )

        if isinstance(quantization, BlockScaleQuantize):
            scale_mlc_name = f"{weight_mlc_name}_scale_inv"
            if scale_mlc_name in named_parameters:
                scale_hf_names = [f"{name}_scale_inv" for name in hf_names]
                scale_param = named_parameters[scale_mlc_name]
                mapping.add_mapping(
                    scale_mlc_name,
                    scale_hf_names,
                    functools.partial(weight_transform_func, dtype=scale_param.dtype),
                )

    for i in range(model_config.num_hidden_layers):
        # map attention weight
        attn = f"model.layers.{i}.self_attn"
        add_weight_and_scale_mapping(
            f"{attn}.c_attn.weight",
            [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
            ],
            lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
        )
        if model_config.attention_bias:
            mlc_name = f"{attn}.c_attn.bias"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    to_hf(f"{attn}.q_proj.bias"),
                    to_hf(f"{attn}.k_proj.bias"),
                    to_hf(f"{attn}.v_proj.bias"),
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
        # map mlp weight
        mlp = f"model.layers.{i}.mlp"
        add_weight_and_scale_mapping(
            f"{mlp}.gate_up_proj.weight",
            [
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
            ],
            lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [to_hf(mlc_name)],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping


def huggingface_embedding(model_config: Qwen3Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping for Qwen3-Embedding models (no 'model.' prefix)."""
    return huggingface(model_config, quantization, "")
