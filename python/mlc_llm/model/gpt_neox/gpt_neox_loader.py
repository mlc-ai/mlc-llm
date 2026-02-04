"""
This file specifies how MLC's GPTNeoX parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gpt_neox_model import GPTNeoXConfig, GPTNeoXForCausalLM


def huggingface(model_config: GPTNeoXConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GPTNeoXConfig
        The configuration of the GPTNeoX model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = GPTNeoXForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # inv_freq/masked_bias/bias is not used in the model
        attn = f"gpt_neox.layers.{i}.attention"
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")
        mapping.add_unused(f"{attn}.masked_bias")
        mapping.add_unused(f"{attn}.bias")

        # change the layout of query_key_value
        def transform_qkv_layout(w, dtype):  # pylint: disable=invalid-name
            num_attention_heads = model_config.num_attention_heads
            head_dim = model_config.head_dim

            org_shape = w.shape
            w = np.reshape(w, [num_attention_heads, 3 * head_dim, -1])
            qkv = np.split(w, indices_or_sections=3, axis=1)
            w = np.concatenate(qkv, axis=0)
            w = np.reshape(w, org_shape)
            return w.astype(dtype)

        qkv_proj = f"{attn}.query_key_value"
        for param_name in ["weight", "bias"]:
            mlc_name = f"{qkv_proj}.{param_name}"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    transform_qkv_layout,
                    dtype=mlc_param.dtype,
                ),
            )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            if ".dense_h_to_4h.bias" in mlc_name or ".dense_4h_to_h.bias" in mlc_name:
                param_dtype = model_config.ffn_out_dtype
            else:
                param_dtype = mlc_param.dtype
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=param_dtype,
                ),
            )
    return mapping
