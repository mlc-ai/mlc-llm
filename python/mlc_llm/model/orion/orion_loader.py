"""
This file specifies how MLC's Orion parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .orion_model import OrionForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=OrionForCausalLM,
    add_unused=["rotary_emb.inv_freq"],
)
