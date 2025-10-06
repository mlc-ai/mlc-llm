"""
This file specifies how MLC's Deepseek-V2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Callable, List

import numpy as np

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization

from .deepseek_v2_model import DeepseekV2Config, DeepseekV2ForCausalLM


def huggingface(  # pylint: disable=too-many-locals,too-many-statements
    model_config: DeepseekV2Config, quantization: Quantization
) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : DeepseekV2Config
        The configuration of the DeepseekV2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = DeepseekV2ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    if isinstance(quantization, BlockScaleQuantize):
        # Convert the model to block-scale quantized model before loading parameters
        model = quantization.quantize_model(model, QuantizeMapping({}, {}), "")
        if model_config.weight_block_size is None:
            raise ValueError(
                "The input DeepSeek model is not fp8 block quantized. "
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
            "The input DeepSeek model is fp8 block quantized. "
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
        if i >= model_config.first_k_dense_replace and i % model_config.moe_layer_freq == 0:
            # map mlp shared expert weight
            mlp = f"model.layers.{i}.mlp"
            shared_expert = f"{mlp}.shared_experts"
            add_weight_and_scale_mapping(
                f"{shared_expert}.gate_up_proj.weight",
                [
                    f"{shared_expert}.gate_proj.weight",
                    f"{shared_expert}.up_proj.weight",
                ],
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
            )

            # map mlp moe gate and up weight
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
                        for expert_id in range(model_config.n_routed_experts)
                    ],
                ),
                combine_expert_gate_up,
            )

            # map mlp moe down projection weight
            add_weight_and_scale_mapping(
                f"{mlp}.moe_down_proj.weight",
                [
                    f"{mlp}.experts.{expert_id}.down_proj.weight"
                    for expert_id in range(model_config.n_routed_experts)
                ],
                lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
            )

            # map moe e_score_correction_bias
            if model_config.topk_method == "noaux_tc":
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
        else:
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

        # map MLA kv_b_proj weight
        attn = f"model.layers.{i}.self_attn"
        mlc_name = f"{attn}.w_uk"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{attn}.kv_b_proj.weight"],
            functools.partial(
                lambda kv_b_proj, dtype: np.split(
                    kv_b_proj.reshape(
                        model_config.num_key_value_heads,
                        model_config.qk_nope_head_dim + model_config.v_head_dim,
                        model_config.kv_lora_rank,
                    ),
                    indices_or_sections=[model_config.qk_nope_head_dim],
                    axis=1,
                )[0]
                .transpose(0, 2, 1)
                .astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        if isinstance(quantization, BlockScaleQuantize):
            scale_mlc_name = f"{attn}.w_uk_scale_inv"
            mlc_param = named_parameters[scale_mlc_name]
            mapping.add_mapping(
                scale_mlc_name,
                [f"{attn}.kv_b_proj.weight_scale_inv"],
                functools.partial(
                    lambda kv_b_proj, dtype: np.split(
                        kv_b_proj.reshape(
                            model_config.num_key_value_heads,
                            (model_config.qk_nope_head_dim + model_config.v_head_dim)
                            // quantization.weight_block_size[0],
                            model_config.kv_lora_rank // quantization.weight_block_size[1],
                        ),
                        indices_or_sections=[
                            model_config.qk_nope_head_dim // quantization.weight_block_size[0]
                        ],
                        axis=1,
                    )[0]
                    .transpose(0, 2, 1)
                    .astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
        mlc_name = f"{attn}.w_uv"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{attn}.kv_b_proj.weight"],
            functools.partial(
                lambda kv_b_proj, dtype: np.split(
                    kv_b_proj.reshape(
                        model_config.num_key_value_heads,
                        model_config.qk_nope_head_dim + model_config.v_head_dim,
                        model_config.kv_lora_rank,
                    ),
                    indices_or_sections=[model_config.qk_nope_head_dim],
                    axis=1,
                )[1].astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        if isinstance(quantization, BlockScaleQuantize):
            scale_mlc_name = f"{attn}.w_uv_scale_inv"
            mlc_param = named_parameters[scale_mlc_name]
            mapping.add_mapping(
                scale_mlc_name,
                [f"{attn}.kv_b_proj.weight_scale_inv"],
                functools.partial(
                    lambda kv_b_proj, dtype: np.split(
                        kv_b_proj.reshape(
                            model_config.num_key_value_heads,
                            (model_config.qk_nope_head_dim + model_config.v_head_dim)
                            // quantization.weight_block_size[0],
                            model_config.kv_lora_rank // quantization.weight_block_size[1],
                        ),
                        indices_or_sections=[
                            model_config.qk_nope_head_dim // quantization.weight_block_size[0]
                        ],
                        axis=1,
                    )[1].astype(dtype),
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
