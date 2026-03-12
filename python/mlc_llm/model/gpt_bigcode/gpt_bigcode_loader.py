"""
This file specifies how MLC's GPTBigCode parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .gpt_bigcode_model import GPTBigCodeForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=GPTBigCodeForCausalLM,
    include_qkv=False,
    include_gate_up=False,
)
