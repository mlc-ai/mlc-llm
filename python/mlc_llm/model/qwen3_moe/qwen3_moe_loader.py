"""
This file specifies how MLC's QWen2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Callable, List

import numpy as np

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization

from .qwen3_moe_model import Qwen3MoeConfig, Qwen3MoeForCausalLM


def huggingface(model_config: Qwen3MoeConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Qwen3MoeConfig
        The configuration of the Qwen3Moe model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Qwen3MoeForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    if isinstance(quantization, BlockScaleQuantize):
        # Convert the model to block-scale quantized model before loading parameters
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

    # Helper function to add both weight and scale mappings
    def add_weight_and_scale_mapping(
        weight_mlc_name: str,
        weight_hf_names: List[str],
        weight_transform_func: Callable,
    ):
        mlc_param = named_parameters[weight_mlc_name]
        mapping.add_mapping(
            weight_mlc_name,
            weight_hf_names,
            functools.partial(weight_transform_func, dtype=mlc_param.dtype),
        )

        if isinstance(quantization, BlockScaleQuantize):
            scale_mlc_name = f"{weight_mlc_name}_scale_inv"
            if scale_mlc_name in named_parameters:
                scale_hf_names = [f"{name}_scale_inv" for name in weight_hf_names]
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
                    f"{attn}.q_proj.bias",
                    f"{attn}.k_proj.bias",
                    f"{attn}.v_proj.bias",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
        # map mlp moe gate and up weight
        mlp = f"model.layers.{i}.mlp"

        def combine_expert_gate_up(*hf_params, dtype):
            stack = []
            for i in range(0, len(hf_params), 2):
                stack.append(np.concatenate([hf_params[i], hf_params[i + 1]], axis=0))
            return np.stack(stack, axis=0).astype(dtype)

        add_weight_and_scale_mapping(
            f"{mlp}.moe_gate_up_proj.weight",
            functools.reduce(
                lambda a, b: a + b,
                [
                    [
                        f"{mlp}.experts.{expert_id}.gate_proj.weight",
                        f"{mlp}.experts.{expert_id}.up_proj.weight",
                    ]
                    for expert_id in range(model_config.num_experts)
                ],
            ),
            combine_expert_gate_up,
        )

        # map mlp moe down projection weight
        add_weight_and_scale_mapping(
            f"{mlp}.moe_down_proj.weight",
            [
                f"{mlp}.experts.{expert_id}.down_proj.weight"
                for expert_id in range(model_config.num_experts)
            ],
            lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
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
