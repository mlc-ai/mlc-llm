"""
This file specifies how MLC's InternLM parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .internlm_model import InternLMForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=InternLMForCausalLM,
    qkv_target_name="wqkv_pack",
    add_qkv_bias=True,
    qkv_bias_optional=True,
    add_unused=["rotary_emb.inv_freq"],
)
