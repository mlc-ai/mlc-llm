"""
This file specifies how MLC's BGE parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .bge_model import BGEConfig, BGEModel


def huggingface(model_config: BGEConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : BGEConfig
        The configuration of the BGE model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = BGEModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    prefix = "roberta." if model_config.is_reranker else ""

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        attn = f"encoder.layer.{i}.attention.self"
        mlc_name = f"{attn}.qkv.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{prefix}{attn}.query.weight",
                f"{prefix}{attn}.key.weight",
                f"{prefix}{attn}.value.weight",
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        mlc_name = f"{attn}.qkv.bias"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{prefix}{attn}.query.bias",
                f"{prefix}{attn}.key.bias",
                f"{prefix}{attn}.value.bias",
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    if model_config.is_reranker:
        mapping.add_mapping(
            "classifier.dense.bias",
            ["classifier.dense.bias"],
            functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
        )
        mapping.add_mapping(
            "classifier.dense.weight",
            ["classifier.dense.weight"],
            functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
        )
        mapping.add_mapping(
            "classifier.out_proj.bias",
            ["classifier.out_proj.bias"],
            functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
        )
        mapping.add_mapping(
            "classifier.out_proj.weight",
            ["classifier.out_proj.weight"],
            functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [prefix + mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping
