"""
This file specifies how MLC's Mixtral parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .mixtral_model import MixtralConfig, MixtralForCasualLM


def huggingface(model_config: MixtralConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : MixtralConfig
        The configuration of the Mixtral model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = MixtralForCasualLM(model_config)
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

        # Add gates in MLP (when MoE is enabled)
        mlp = f"model.layers.{i}.block_sparse_moe"
        mlc_mlp = f"model.layers.{i}.moe"
        mlc_name = f"{mlc_mlp}.e1_e3.weight"
        mlc_param = named_parameters[mlc_name]

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
                        f"{mlp}.experts.{expert_id}.w1.weight",
                        f"{mlp}.experts.{expert_id}.w3.weight",
                    ]
                    for expert_id in range(model_config.num_local_experts)
                ],
            ),
            functools.partial(
                combine_expert_gate_up,
                dtype=mlc_param.dtype,
            ),
        )

        mlc_name = f"{mlc_mlp}.e2.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{mlp}.experts.{expert_id}.w2.weight"
                for expert_id in range(model_config.num_local_experts)
            ],
            functools.partial(
                lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        mlc_name = f"{mlc_mlp}.gate.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.gate.weight"],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

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
