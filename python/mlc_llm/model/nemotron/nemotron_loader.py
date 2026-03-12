"""
This file specifies how MLC's Nemotron parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .nemotron_model import NemotronForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=NemotronForCausalLM,
    add_unused=["rotary_emb.inv_freq"],
    include_gate_up=False,
)
