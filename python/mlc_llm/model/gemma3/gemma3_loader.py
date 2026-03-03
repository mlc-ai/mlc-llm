"""
This file specifies how MLC's Gemma3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma3_model import Gemma3Config, Gemma3ForCausalLM


def huggingface(model_config: Gemma3Config, quantization: Quantization) -> ExternMapping:
    model = Gemma3ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    base_loader = make_standard_hf_loader(
        model_cls=Gemma3ForCausalLM,
        include_qkv=False,
        include_gate_up=False,
        num_layers_getter=lambda config: config.text_config.num_hidden_layers,  # type: ignore[attr-defined]
    )
    mapping = base_loader(model_config, quantization)

    mlc_prefix = "language_model."
    hf_prefix = "language_model." if not model_config.is_text_model else ""

    def hf(name: str) -> str:
        return f"{hf_prefix}{name}"

    def add_one(name: str) -> None:
        mlc_param = named_parameters[mlc_prefix + name]
        mapping.add_mapping(
            mlc_prefix + name,
            [hf(name)],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    for i in range(model_config.text_config.num_hidden_layers):
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{mlc_prefix + mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [hf(f"{mlp}.gate_proj.weight"), hf(f"{mlp}.up_proj.weight")],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        add_one(f"model.layers.{i}.input_layernorm.weight")
        add_one(f"model.layers.{i}.post_attention_layernorm.weight")
        add_one(f"model.layers.{i}.pre_feedforward_layernorm.weight")
        add_one(f"model.layers.{i}.post_feedforward_layernorm.weight")
        add_one(f"model.layers.{i}.self_attn.k_norm.weight")
        add_one(f"model.layers.{i}.self_attn.q_norm.weight")

    add_one("model.norm.weight")

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
