"""
This file specifies how MLC's GPT-2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

import numpy as np

from ...loader import ExternMapping
from ...quantization import Quantization
from .gpt2_model import GPT2Config, GPT2LMHeadModel


def huggingface(model_config: GPT2Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GPT2Config
        The configuration of the GPT-2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = GPT2LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    mapping.add_mapping(
        "lm_head.weight",
        ["wte.weight"],
        functools.partial(
            lambda x, dtype: x.astype(dtype),
            dtype=named_parameters["transformer.wte.weight"].dtype,
        ),
    )

    for i in range(model_config.n_layer):
        mapping.add_unused(f"h.{i}.attn.bias")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            # transformer.h.0.attn.c_attn.weight --> h.0.attn.c_attn.weight
            source_name = mlc_name.split(".", 1)[1]
            need_transpose = False
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in mlc_name:
                    continue
                if not mlc_name.endswith(".weight"):
                    continue
                need_transpose = True

            if need_transpose:
                mapping.add_mapping(
                    mlc_name,
                    [source_name],
                    functools.partial(
                        lambda x, dtype: x.transpose().astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
            else:
                mapping.add_mapping(
                    mlc_name,
                    [source_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )

    return mapping
