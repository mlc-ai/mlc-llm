"""
This file specifies how MLC's BERT parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Literal

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .bert_model import BertConfig, BertModel


def huggingface(
    model_config: BertConfig,
    quantization: Quantization,
    hf_prefix: Literal["", "bert."] = "",
) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : BertConfig
        The configuration of the BERT model.

    quantization : Quantization
        The quantization configuration.

    hf_prefix : Literal["", "bert."]
        Prefix used in HuggingFace weight names. Defaults to "" for standard
        BERT models. Use "bert." for BGE models whose weights are prefixed.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = BertModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def to_hf(name: str) -> str:
        return f"{hf_prefix}{name}" if hf_prefix else name

    for i in range(model_config.num_hidden_layers):
        attn = f"encoder.layer.{i}.attention.self"
        mlc_name = f"{attn}.qkv.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                to_hf(f"{attn}.query.weight"),
                to_hf(f"{attn}.key.weight"),
                to_hf(f"{attn}.value.weight"),
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
                to_hf(f"{attn}.query.bias"),
                to_hf(f"{attn}.key.bias"),
                to_hf(f"{attn}.value.bias"),
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [to_hf(mlc_name)],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    # Mark unused weights that exist in HF but not in MLC
    if hf_prefix:
        mapping.add_unused(f"{hf_prefix}pooler.dense.weight")
        mapping.add_unused(f"{hf_prefix}pooler.dense.bias")

    return mapping


def huggingface_bge(model_config: BertConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping for BGE models.

    BGE weights have no prefix but include extra unused weights:
    pooler.dense.weight, pooler.dense.bias, embeddings.position_ids
    """
    mapping = huggingface(model_config, quantization, "")
    mapping.add_unused("pooler.dense.weight")
    mapping.add_unused("pooler.dense.bias")
    mapping.add_unused("embeddings.position_ids")
    return mapping
