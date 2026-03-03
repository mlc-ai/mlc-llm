"""
This file specifies how MLC's QWen2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .qwen2_model import QWen2LMHeadModel

huggingface = make_standard_hf_loader(
    model_cls=QWen2LMHeadModel,
    qkv_target_name="c_attn",
    add_qkv_bias=True,
    add_unused=["rotary_emb.inv_freq"],
)
