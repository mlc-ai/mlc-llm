"""
This file specifies how MLC's OLMo2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .olmo2_model import OLMo2ForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=OLMo2ForCausalLM,
    add_unused=["rotary_emb.inv_freq"],
)
