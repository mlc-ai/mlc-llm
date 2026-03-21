"""
This file specifies how MLC's ChatGLM3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .chatglm3_model import ChatGLMForCausalLM, GLMConfig


def huggingface(model_config: GLMConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GLMConfig
        The configuration of the Baichuan model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = ChatGLMForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def _name_transform(param_name: str) -> str:
        # model.embed_tokens.weight -> transformer.embedding.word_embeddings.weight
        if param_name.startswith("model.embed_tokens."):
            return param_name.replace(
                "model.embed_tokens.", "transformer.embedding.word_embeddings.", 1
            )
        # model.* -> transformer.* (encoder, output_layer, etc.)
        if param_name.startswith("model."):
            return param_name.replace("model.", "transformer.", 1)
        return param_name

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [_name_transform(mlc_name)],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping
