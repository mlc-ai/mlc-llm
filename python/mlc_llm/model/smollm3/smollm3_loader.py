"""
This file specifies how MLC's SmolLM3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .smollm3_model import SmolLM3ForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=SmolLM3ForCausalLM,
    add_unused=["rotary_emb.inv_freq"],
)
