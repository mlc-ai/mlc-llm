"""
This file specifies how MLC's BaichuanLM parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .baichuan_model import BaichuanForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=BaichuanForCausalLM,
    include_qkv=False,
)
