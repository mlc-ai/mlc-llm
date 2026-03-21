"""
This file specifies how MLC's GPTJ parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .gpt_j_model import GPTJForCausalLM


def _name_transform(param_name: str) -> str:
    # model.embed_tokens.* -> transformer.wte.*
    if param_name.startswith("model.embed_tokens."):
        return param_name.replace("model.embed_tokens.", "transformer.wte.", 1)
    # lm_head.* -> lm_head.* (no change)
    if param_name.startswith("lm_head."):
        return param_name
    # model.* -> transformer.* (h, ln_f, etc.)
    if param_name.startswith("model."):
        return param_name.replace("model.", "transformer.", 1)
    return param_name


huggingface = make_standard_hf_loader(
    model_cls=GPTJForCausalLM,
    layer_prefix="model.h",
    qkv_target_name="c_attn",
    include_gate_up=False,
    num_layers_getter=lambda config: config.n_layer,  # type: ignore[attr-defined]
    name_transform=_name_transform,
)
