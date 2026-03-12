"""
This file specifies how MLC's GPTJ parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from .gpt_j_model import GPTJForCausalLM

huggingface = make_standard_hf_loader(
    model_cls=GPTJForCausalLM,
    layer_prefix="transformer.h",
    qkv_target_name="c_attn",
    include_gate_up=False,
    num_layers_getter=lambda config: config.n_layer,  # type: ignore[attr-defined]
)
