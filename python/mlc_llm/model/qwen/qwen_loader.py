"""
This file specifies how MLC's QWen parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .qwen_model import QWenLMHeadModel

huggingface = make_standard_hf_loader(
    model_cls=QWenLMHeadModel,
    layer_prefix="transformer.h",
    qkv_names=(),
    include_qkv=False,
    gate_up_names=("w1", "w2"),
)
