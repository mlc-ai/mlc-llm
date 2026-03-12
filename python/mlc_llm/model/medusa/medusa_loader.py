"""
This file specifies how MLC's Medusa parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .medusa_model import MedusaModel

huggingface = make_standard_hf_loader(
    model_cls=MedusaModel,
    include_qkv=False,
    include_gate_up=False,
)
