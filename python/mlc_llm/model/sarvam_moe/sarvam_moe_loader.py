import functools
from typing import Callable, List

import numpy as np

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization

from .sarvam_moe_model import SarvamMoeConfig, SarvamMoeForCausalLM


def huggingface(model_config: SarvamMoeConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping from HuggingFace SarvamMoE weights to MLC SarvamMoE weights."""
    model = SarvamMoeForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    if isinstance(quantization, BlockScaleQuantize):
        model = quantization.quantize_model(
            model,
            QuantizeMapping({}, {}),
            "",
            skip_param_names=lambda name: "expert_bias" in name,
        )
        if model_config.weight_block_size is None:
            raise ValueError(
                "The input Sarvam model is not fp8 block quantized. "
                "Thus BlockScaleQuantize is not supported."
            )
    _, _named_params, _ = model.export_tvm(
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
            "The input Sarvam model is fp8 block quantized. "
            "Please use BlockScaleQuantize for the model."
        )

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

    def identity_mapping(mlc_name: str):
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [mlc_name],
            functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
        )

    def combine_gate_up(gate, up, dtype):
        return np.concatenate([gate, up], axis=0).astype(dtype)

    def combine_expert_gate_up(*hf_params, dtype):
        stack = []
        for idx in range(0, len(hf_params), 2):
            stack.append(np.concatenate([hf_params[idx], hf_params[idx + 1]], axis=0))
        return np.stack(stack, axis=0).astype(dtype)

    # Explicit non-identity mappings
    add_weight_and_scale_mapping(
        "model.embed_tokens.weight",
        ["model.word_embeddings.weight"],
        lambda w, dtype: w.astype(dtype),
    )
    for i in range(model_config.num_hidden_layers):
        attn = f"model.layers.{i}.self_attn"
        mlp = f"model.layers.{i}.mlp"
        # Attention projections
        add_weight_and_scale_mapping(
            f"{attn}.c_attn.weight",
            [f"model.layers.{i}.attention.query_key_value.weight"],
            lambda qkv, dtype: qkv.astype(dtype),
        )
        add_weight_and_scale_mapping(
            f"{attn}.o_proj.weight",
            [f"model.layers.{i}.attention.dense.weight"],
            lambda w, dtype: w.astype(dtype),
        )
        # Q/K norm mapping
        add_weight_and_scale_mapping(
            f"{attn}.q_norm.weight",
            [f"model.layers.{i}.attention.query_layernorm.weight"],
            lambda w, dtype: w.astype(dtype),
        )
        add_weight_and_scale_mapping(
            f"{attn}.k_norm.weight",
            [f"model.layers.{i}.attention.key_layernorm.weight"],
            lambda w, dtype: w.astype(dtype),
        )
        # Dense early layers
        if i < model_config.first_k_dense_replace:
            add_weight_and_scale_mapping(
                f"{mlp}.gate_up_proj.weight",
                [
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                ],
                combine_gate_up,
            )
            add_weight_and_scale_mapping(
                f"{mlp}.down_proj.weight",
                [f"model.layers.{i}.mlp.down_proj.weight"],
                lambda w, dtype: w.astype(dtype),
            )
        else:
            # Router
            add_weight_and_scale_mapping(
                f"{mlp}.gate.weight",
                [f"model.layers.{i}.mlp.gate.weight"],
                lambda w, dtype: w.astype(dtype),
            )
            # gate+up packed together
            add_weight_and_scale_mapping(
                f"{mlp}.moe_gate_up_proj.weight",
                functools.reduce(
                    lambda a, b: a + b,
                    [
                        [
                            f"model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                            f"model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                        ]
                        for expert_id in range(model_config.num_experts)
                    ],
                ),
                combine_expert_gate_up,
            )
            # Routed experts
            add_weight_and_scale_mapping(
                f"{mlp}.moe_down_proj.weight",
                [
                    f"model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight"
                    for expert_id in range(model_config.num_experts)
                ],
                lambda *hf_params, dtype: np.stack(hf_params, axis=0).astype(dtype),
            )
            # Shared expert branch
            if model_config.num_shared_experts > 0:
                add_weight_and_scale_mapping(
                    f"{mlp}.shared_expert.gate_up_proj.weight",
                    [
                        f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                        f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                    ],
                    combine_gate_up,
                )
                add_weight_and_scale_mapping(
                    f"{mlp}.shared_expert.down_proj.weight",
                    [f"model.layers.{i}.mlp.shared_experts.down_proj.weight"],
                    lambda w, dtype: w.astype(dtype),
                )
            # Expert bias
            if model_config.moe_router_enable_expert_bias:
                add_weight_and_scale_mapping(
                    f"{mlp}.expert_bias",
                    [f"model.layers.{i}.mlp.gate.expert_bias"],
                    lambda b, dtype: b.astype(dtype),
                )
    # remaining matching names
    for mlc_name in named_parameters:
        if mlc_name not in mapping.param_map:
            identity_mapping(mlc_name)
    return mapping
