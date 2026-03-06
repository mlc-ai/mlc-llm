"""
This file specifies how MLC's StableLM parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .stablelm_model import StableLmForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=StableLmForCausalLM,
    add_qkv_bias=True,
    qkv_bias_optional=True,
)
