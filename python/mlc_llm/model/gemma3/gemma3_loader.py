"""
This file specifies how MLC's Gemma3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma3_model import Gemma3Config, Gemma3ForCausalLM


def huggingface(model_config: Gemma3Config, quantization: Quantization) -> ExternMapping:
    """Create HF weight mapping for Gemma3."""
    model = Gemma3ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    mlc_prefix = "language_model."
    if model_config.is_text_model:
        hf_prefix = ""
    else:
        hf_prefix = "language_model."

    def name_transform(name: str) -> str:
        if name.startswith(mlc_prefix):
            name = name[len(mlc_prefix) :]
        return f"{hf_prefix}{name}"

    def num_layers(config: object) -> int:
        return config.text_config.num_hidden_layers

    base_loader = make_standard_hf_loader(
        model_cls=Gemma3ForCausalLM,
        include_qkv=False,
        include_gate_up=True,
        gate_up_target_name="gate_up_proj",
        num_layers_getter=num_layers,
        layer_prefix=f"{mlc_prefix}model.layers",
        name_transform=name_transform,
    )
    mapping = base_loader(model_config, quantization)

    def add_one(name: str) -> None:
        mlc_param = named_parameters[mlc_prefix + name]
        mapping.add_mapping(
            mlc_prefix + name,
            [name_transform(mlc_prefix + name)],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    for i in range(model_config.text_config.num_hidden_layers):
        add_one(f"model.layers.{i}.input_layernorm.weight")
        add_one(f"model.layers.{i}.post_attention_layernorm.weight")
        add_one(f"model.layers.{i}.pre_feedforward_layernorm.weight")
        add_one(f"model.layers.{i}.post_feedforward_layernorm.weight")
        add_one(f"model.layers.{i}.self_attn.k_norm.weight")
        add_one(f"model.layers.{i}.self_attn.q_norm.weight")

    add_one("model.norm.weight")

    return mapping
