"""
This file specifies how MLC's GLM-4.5-Air MoE parameters map from HuggingFace formats.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .glm4_moe_model import Glm4MoeConfig, Glm4MoeForCausalLM


def huggingface(model_config: Glm4MoeConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Glm4MoeConfig
        The configuration of the GLM-4.5-Air MoE model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Glm4MoeForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # Map attention weights
        attn = f"model.layers.{i}.self_attn"

        # QKV weight concatenation
        mlc_name = f"{attn}.c_attn.weight"
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

        # QKV bias concatenation (GLM-4.5 has attention bias)
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

        mlp = f"model.layers.{i}.mlp"

        if i < model_config.first_k_dense_replace:
            # Dense MLP layers (layer 0 by default)
            # Combine gate_proj and up_proj into gate_up_proj
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
        else:
            # MoE layers (layers 1+ by default)

            # 1. Router gate weight (direct mapping)
            # model.layers.{i}.mlp.gate.weight -> model.layers.{i}.mlp.gate.weight

            # 2. Router bias correction
            mlc_name = f"{mlp}.e_score_correction_bias"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [f"{mlp}.gate.e_score_correction_bias"],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

            # 3. Routed experts: combine gate_proj and up_proj for each expert, then stack
            mlc_name = f"{mlp}.moe_gate_up_proj.weight"
            mlc_param = named_parameters[mlc_name]

            def combine_expert_gate_up(*hf_params, dtype, num_experts):
                """Combine gate and up projections for all experts into stacked tensor.

                HuggingFace format:
                    experts.{j}.gate_proj.weight: [moe_intermediate_size, hidden_size]
                    experts.{j}.up_proj.weight: [moe_intermediate_size, hidden_size]

                MLC format:
                    moe_gate_up_proj.weight: [num_experts, 2*moe_intermediate_size, hidden_size]
                """
                stack = []
                for j in range(num_experts):
                    gate = hf_params[j]  # gate_proj for expert j
                    up = hf_params[num_experts + j]  # up_proj for expert j
                    combined = np.concatenate([gate, up], axis=0)
                    stack.append(combined)
                return np.stack(stack, axis=0).astype(dtype)

            # Build list of HF parameter names: first all gate_proj, then all up_proj
            hf_gate_names = [
                f"{mlp}.experts.{j}.gate_proj.weight"
                for j in range(model_config.n_routed_experts)
            ]
            hf_up_names = [
                f"{mlp}.experts.{j}.up_proj.weight"
                for j in range(model_config.n_routed_experts)
            ]
            mapping.add_mapping(
                mlc_name,
                hf_gate_names + hf_up_names,
                functools.partial(
                    combine_expert_gate_up,
                    dtype=mlc_param.dtype,
                    num_experts=model_config.n_routed_experts,
                ),
            )

            # 4. Routed experts: stack down_proj for all experts
            mlc_name = f"{mlp}.moe_down_proj.weight"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{mlp}.experts.{j}.down_proj.weight"
                    for j in range(model_config.n_routed_experts)
                ],
                functools.partial(
                    lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

            # 5. Shared expert: combine gate_proj and up_proj
            mlc_name = f"{mlp}.shared_expert_gate_up.weight"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{mlp}.shared_experts.gate_proj.weight",
                    f"{mlp}.shared_experts.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

            # 6. Shared expert: down_proj (direct mapping with renamed path)
            mlc_name = f"{mlp}.shared_expert_down.weight"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [f"{mlp}.shared_experts.down_proj.weight"],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    # Map all remaining parameters that have direct correspondence
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
