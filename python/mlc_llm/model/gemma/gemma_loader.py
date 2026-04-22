"""
This file specifies how MLC's Gemma parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma_model import GemmaConfig, GemmaForCausalLM


def huggingface(model_config: GemmaConfig, quantization: Quantization) -> ExternMapping:
    """Create HF weight mapping for Gemma."""
    model = GemmaForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    base_loader = make_standard_hf_loader(
        model_cls=GemmaForCausalLM,
    )
    mapping = base_loader(model_config, quantization)

    def add_one(name: str) -> None:
        mlc_param = named_parameters[name]
        mapping.add_mapping(
            name,
            [name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    for i in range(model_config.num_hidden_layers):
        add_one(f"model.layers.{i}.input_layernorm.weight")
        add_one(f"model.layers.{i}.post_attention_layernorm.weight")

    add_one("model.norm.weight")
    return mapping
