"""
This file specifies how MLC's Phi parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .phi3_model import Phi3Config, Phi3ForCausalLM


def phi3_huggingface(model_config: Phi3Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of Phi-1/Phi-1.5 HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : PhiConfig
        The configuration of the Phi model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Phi3ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=W0632:unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def _add(mlc_name, hf_name):
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    _add("lm_head.weight", "lm_head.weight")
    _add("transformer.norm.weight", "model.norm.weight")
    _add("transformer.embd.weight", "model.embed_tokens.weight")

    prefix = "transformer.h"
    hf_prefix = "model.layers"
    for i in range(model_config.num_hidden_layers):
        _add(f"{prefix}.{i}.ln.weight", f"{hf_prefix}.{i}.input_layernorm.weight")
        _add(f"{prefix}.{i}.mlp.down_proj.weight", f"{hf_prefix}.{i}.mlp.down_proj.weight")
        _add(f"{prefix}.{i}.mlp.gate_up_proj.weight", f"{hf_prefix}.{i}.mlp.gate_up_proj.weight")
        _add(
            f"{prefix}.{i}.post_attention_layernorm.weight",
            f"{hf_prefix}.{i}.post_attention_layernorm.weight",
        )
        _add(f"{prefix}.{i}.mixer.out_proj.weight", f"{hf_prefix}.{i}.self_attn.o_proj.weight")
        _add(f"{prefix}.{i}.mixer.qkv_proj.weight", f"{hf_prefix}.{i}.self_attn.qkv_proj.weight")
    return mapping
