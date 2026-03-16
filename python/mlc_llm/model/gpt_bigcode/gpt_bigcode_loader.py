"""
This file specifies how MLC's GPTBigCode parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .gpt_bigcode_model import GPTBigCodeForCausalLM


def _name_transform(param_name: str) -> str:
    # model.embed_tokens.* -> transformer.wte.*
    if param_name.startswith("model.embed_tokens."):
        return param_name.replace("model.embed_tokens.", "transformer.wte.", 1)
    # lm_head.* -> lm_head.* (no change)
    if param_name.startswith("lm_head."):
        return param_name
    # model.* -> transformer.* (h, wpe, ln_f, etc.)
    if param_name.startswith("model."):
        return param_name.replace("model.", "transformer.", 1)
    return param_name


huggingface = make_standard_hf_loader(
    model_cls=GPTBigCodeForCausalLM,
    include_qkv=False,
    include_gate_up=False,
    name_transform=_name_transform,
)
