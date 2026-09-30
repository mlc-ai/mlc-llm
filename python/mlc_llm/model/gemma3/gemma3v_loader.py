"""
This file specifies how MLC's Gemma3V parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma3v_model import Gemma3VConfig, Gemma3VForCausalLM

_MLC_PREFIX = "language_model."
_HF_PREFIX = "language_model."


def _name_transform(name: str) -> str:
    """Map MLC parameter names to HuggingFace names.

    Language model params: strip mlc_prefix, add hf_prefix.
    Vision/projector params: pass through unchanged (1:1 mapping).
    """
    if name.startswith(_MLC_PREFIX):
        return _HF_PREFIX + name[len(_MLC_PREFIX) :]
    return name


def _num_layers(config: object) -> int:
    return config.text_config.num_hidden_layers  # type: ignore[attr-defined]


def huggingface(model_config: Gemma3VConfig, quantization: Quantization) -> ExternMapping:
    """Create HF weight mapping for Gemma3V."""
    model = Gemma3VForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    base_loader = make_standard_hf_loader(
        model_cls=Gemma3VForCausalLM,
        include_qkv=False,
        include_gate_up=True,
        gate_up_target_name="gate_up_proj",
        num_layers_getter=_num_layers,
        layer_prefix=f"{_MLC_PREFIX}model.layers",
        name_transform=_name_transform,
    )
    mapping = base_loader(model_config, quantization)

    # ========== Gemma-specific: RMS norm weights need +1 ==========
    def add_one(name: str) -> None:
        mlc_param = named_parameters[_MLC_PREFIX + name]
        mapping.add_mapping(
            _MLC_PREFIX + name,
            [_name_transform(_MLC_PREFIX + name)],
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

    # ========== Multimodal projector overrides ==========
    # mm_input_projection: HF stores as (vision_hidden, text_hidden), nn.Linear expects
    # (text_hidden, vision_hidden) -> TRANSPOSE required
    proj_weight_mlc = "multi_modal_projector.mm_input_projection.weight"
    mlc_param = named_parameters[proj_weight_mlc]
    mapping.add_mapping(
        proj_weight_mlc,
        ["multi_modal_projector.mm_input_projection_weight"],
        functools.partial(
            lambda x, dtype: x.T.astype(dtype),
            dtype=mlc_param.dtype,
        ),
    )

    # mm_soft_emb_norm: HF uses Gemma3RMSNorm which adds +1 at runtime
    norm_mlc = "multi_modal_projector.mm_soft_emb_norm.weight"
    mlc_param = named_parameters[norm_mlc]
    mapping.add_mapping(
        norm_mlc,
        [norm_mlc],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=mlc_param.dtype,
        ),
    )

    return mapping
