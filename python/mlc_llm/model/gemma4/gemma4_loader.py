"""Hugging Face parameter mapping for Gemma 4 text+audio artifacts."""

from __future__ import annotations

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma4_audio import gemma4_audio_generated_parameters
from .gemma4_config import Gemma4Config
from .gemma4_model import Gemma4ForConditionalGeneration


def huggingface(model_config: Gemma4Config, quantization: Quantization) -> ExternMapping:
    """Map a Gemma 4 conditional-generation checkpoint, excluding its vision tower."""

    def _name_transform(name: str) -> str:
        return f"model.{name}"

    base_loader = make_standard_hf_loader(
        model_cls=Gemma4ForConditionalGeneration,
        layer_prefix="language_model.layers",
        include_qkv=False,
        include_gate_up=True,
        gate_up_target_name="gate_up_proj",
        name_transform=_name_transform,
        num_layers_getter=lambda config: config.text_config.num_hidden_layers,
    )
    mapping = base_loader(model_config, quantization)

    model = Gemma4ForConditionalGeneration(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    named_parameters = dict(named_params)

    for name, value in gemma4_audio_generated_parameters(model_config.audio_config).items():
        dtype = str(named_parameters[name].dtype)
        mapping.add_mapping(
            name,
            [],
            functools.partial(_cast_generated_parameter, value=value, dtype=dtype),
        )

    packed_ple_name = "model.language_model.embed_tokens_per_layer.weight"
    ple_dim = model_config.text_config.hidden_size_per_layer_input
    for layer_idx in range(model_config.text_config.num_hidden_layers):
        mlc_name = f"language_model.embed_tokens_per_layer.{layer_idx}.weight"
        dtype = str(named_parameters[mlc_name].dtype)
        start = layer_idx * ple_dim
        end = start + ple_dim
        mapping.add_mapping(
            mlc_name,
            [packed_ple_name],
            functools.partial(
                lambda weight, start, end, dtype: weight[:, start:end].astype(dtype),
                start=start,
                end=end,
                dtype=dtype,
            ),
        )

    first_shared = model_config.text_config.first_kv_shared_layer
    for layer_idx in range(first_shared, model_config.text_config.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        for suffix in (
            "k_norm.weight",
            "k_proj.weight",
            "v_norm.weight",
            "v_proj.weight",
        ):
            mapping.add_unused(f"{prefix}.{suffix}")

    _mark_vision_weights_unused(mapping, model_config)
    return mapping


def _cast_generated_parameter(*, value, dtype):
    return value.astype(dtype)


def _mark_vision_weights_unused(mapping: ExternMapping, config: Gemma4Config) -> None:
    mapping.add_unused("model.embed_vision.embedding_projection.weight")
    mapping.add_unused("model.vision_tower.patch_embedder.input_proj.weight")
    mapping.add_unused("model.vision_tower.patch_embedder.position_embedding_table")

    vision_config = config.vision_config or {}
    num_layers = int(vision_config.get("num_hidden_layers", 16))
    layer_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    clipped_linears = (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    clipped_suffixes = (
        "input_min",
        "input_max",
        "linear.weight",
        "output_min",
        "output_max",
    )
    for layer_idx in range(num_layers):
        prefix = f"model.vision_tower.encoder.layers.{layer_idx}"
        for suffix in layer_suffixes:
            mapping.add_unused(f"{prefix}.{suffix}")
        for linear in clipped_linears:
            for suffix in clipped_suffixes:
                mapping.add_unused(f"{prefix}.{linear}.{suffix}")

    if vision_config.get("standardize", False):
        mapping.add_unused("model.vision_tower.std_bias")
        mapping.add_unused("model.vision_tower.std_scale")


__all__ = ["huggingface"]
