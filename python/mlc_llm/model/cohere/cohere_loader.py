"""
This file specifies how MLC's Cohere parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization, make_awq_quant

from .cohere_model import CohereConfig, CohereForCausalLM

awq_quant = make_awq_quant(CohereForCausalLM)


def _cohere_name_transform(name: str) -> str:
    if "out_proj." in name:
        return name.replace("out_proj.", "o_proj.")
    return name


huggingface = make_standard_hf_loader(
    model_cls=CohereForCausalLM,
    include_gate_up=False,
    name_transform=_cohere_name_transform,
)


# https://huggingface.co/alijawad07/aya-23-8B-AWQ-GEMM/tree/main
def awq(model_config: CohereConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : CohereConfig
        The configuration of the Cohere model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),  # type: ignore[attr-defined]
        allow_extern=True,
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

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            mlc_name = f"{attn}.qkv_proj.{quantize_suffix}"
            assert mlc_name in named_parameters
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{attn}.q_proj.{quantize_suffix}",
                    f"{attn}.k_proj.{quantize_suffix}",
                    f"{attn}.v_proj.{quantize_suffix}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate(
                        [q, k, v],
                        axis=1,  # AWQ GEMM would transpose the weight
                    ).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
            _add(f"{attn}.out_proj.{quantize_suffix}", f"{attn}.o_proj.{quantize_suffix}")

        # Concat gate and up in MLP
        mlp = f"model.layers.{i}.mlp"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            _add(f"{mlp}.up_proj.{quantize_suffix}", f"{mlp}.up_proj.{quantize_suffix}")
            _add(
                f"{mlp}.gate_proj.{quantize_suffix}",
                f"{mlp}.gate_proj.{quantize_suffix}",
            )
            _add(
                f"{mlp}.down_proj.{quantize_suffix}",
                f"{mlp}.down_proj.{quantize_suffix}",
            )

        # inv_freq is not used in the model
        # mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
            )
    return mapping
