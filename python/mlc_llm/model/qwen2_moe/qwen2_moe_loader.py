"""
This file specifies how MLC's QWen2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen2_moe_model import Qwen2MoeConfig, Qwen2MoeForCausalLM


def huggingface(model_config: Qwen2MoeConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Qwen2MoeConfig
        The configuration of the Qwen2Moe model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Qwen2MoeForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # map attention weight
        attn = f"model.layers.{i}.self_attn"
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
        # map mlp shared expert weight
        mlp = f"model.layers.{i}.mlp"
        shared_expert = f"{mlp}.shared_expert"
        mlc_name = f"{shared_expert}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{shared_expert}.gate_proj.weight",
                f"{shared_expert}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        # map mlp moe gate and up weight
        mlc_name = f"{mlp}.moe_gate_up_proj.weight"

        def combine_expert_gate_up(*hf_params, dtype):
            stack = []
            for i in range(0, len(hf_params), 2):
                stack.append(np.concatenate([hf_params[i], hf_params[i + 1]], axis=0))
            return np.stack(stack, axis=0).astype(dtype)

        mapping.add_mapping(
            mlc_name,
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
            functools.partial(
                combine_expert_gate_up,
                dtype=mlc_param.dtype,
            ),
        )

        # map mlp moe gate and up weight
        mlc_name = f"{mlp}.moe_down_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{mlp}.experts.{expert_id}.down_proj.weight"
                for expert_id in range(model_config.num_experts)
            ],
            functools.partial(
                lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
                dtype=mlc_param.dtype,
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
