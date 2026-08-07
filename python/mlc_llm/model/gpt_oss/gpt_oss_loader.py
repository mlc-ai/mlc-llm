"""
This file specifies how MLC's GptOss parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np
import torch
from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gpt_oss_model import GptOssConfig, GptOssForCausalLM


def huggingface(model_config: GptOssConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.
    Parameters
    ----------
    model_config : GptOssConfig
        The configuration of the GptOss model.
    quantization : Quantization
        The quantization configuration.
    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = GptOssForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # -------------------------------------------------------------------------
    # Transformation helpers
    # -------------------------------------------------------------------------

    def trans_blocks(*hf_params, dtype):
        """Transpose MXFP4 packed-uint32 weight blocks from HF layout to MLC KN layout.
        HF layout : (num_experts, out_features, in_features//8)  [NK packed]
        MLC layout: (num_experts, in_features//8, out_features)  [KN packed]
        """
        blk = hf_params[0].view(dtype=dtype)
        # reshape to (num_experts, out_features, in_features//8) then transpose
        # last two dims to get (num_experts, in_features//8, out_features)
        param = np.transpose(
            np.reshape(blk, (blk.shape[0], blk.shape[1], -1)), axes=(0, 2, 1)
        ).astype(dtype)
        return param

    def trans_scales(*hf_params, dtype):
        """Convert and transpose MXFP4 scale factors from HF layout to MLC KN layout.
        HF layout : (num_experts, out_features, in_features//group_size)  [NK]
        MLC layout: (num_experts, in_features//group_size, out_features)  [KN]
        Scale values are stored as uint8 exponents in the checkpoint; convert
        to float16 via  scale = 2^(uint8_val - 127).
        """
        blk = hf_params[0].astype("float16") - 127.0
        blk = np.power(2.0, blk)
        param = np.transpose(
            np.reshape(blk, (blk.shape[0], blk.shape[1], -1)), axes=(0, 2, 1)
        ).astype("float16")
        return param

    # -------------------------------------------------------------------------
    # Per-layer mappings
    # -------------------------------------------------------------------------
    for layer_idx in range(model_config.num_hidden_layers):
        attn = f"model.layers.{layer_idx}.self_attn"
        mlp = f"model.layers.{layer_idx}.mlp"

        # --- Attention: fuse Q, K, V projections into c_attn ---
        for weight_type in ["weight", "bias"]:
            mlc_name = f"{attn}.c_attn.{weight_type}"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{attn}.q_proj.{weight_type}",
                    f"{attn}.k_proj.{weight_type}",
                    f"{attn}.v_proj.{weight_type}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # --- MoE gate-up projection: moe_gate_up_proj (GptOssMxfp4Experts) ---
        #
        # HF name                                  MLC name
        # mlp.experts.gate_up_proj_blocks  ->  mlp.moe_gate_up_proj.weight_blocks
        # mlp.experts.gate_up_proj_scales  ->  mlp.moe_gate_up_proj.weight_scales
        # mlp.experts.gate_up_proj_bias    ->  mlp.moe_gate_up_proj.bias

        mlc_name = f"{mlp}.moe_gate_up_proj.weight_blocks"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.gate_up_proj_blocks"],
            functools.partial(trans_blocks, dtype=mlc_param.dtype),
        )

        mlc_name = f"{mlp}.moe_gate_up_proj.weight_scales"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.gate_up_proj_scales"],
            functools.partial(trans_scales, dtype=mlc_param.dtype),
        )

        mlc_name = f"{mlp}.moe_gate_up_proj.bias"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.gate_up_proj_bias"],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        # --- MoE down projection: moe_down_proj (GptOssMxfp4Experts) ---
        #
        # HF name                                MLC name
        # mlp.experts.down_proj_blocks  ->  mlp.moe_down_proj.weight_blocks
        # mlp.experts.down_proj_scales  ->  mlp.moe_down_proj.weight_scales
        # mlp.experts.down_proj_bias    ->  mlp.moe_down_proj.bias

        mlc_name = f"{mlp}.moe_down_proj.weight_blocks"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.down_proj_blocks"],
            functools.partial(trans_blocks, dtype=mlc_param.dtype),
        )

        mlc_name = f"{mlp}.moe_down_proj.weight_scales"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.down_proj_scales"],
            functools.partial(trans_scales, dtype=mlc_param.dtype),
        )

        mlc_name = f"{mlp}.moe_down_proj.bias"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [f"{mlp}.experts.down_proj_bias"],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        # --- MoE router: renamed from `router` to `gate` ---
        #
        # HF name                    MLC name
        # mlp.router.weight  ->  mlp.gate.weight
        # mlp.router.bias    ->  mlp.gate.bias
        for weight_type in ["weight", "bias"]:
            mlc_name = f"{mlp}.gate.{weight_type}"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [f"{mlp}.router.{weight_type}"],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    # -------------------------------------------------------------------------
    # Fallback: all remaining parameters map 1-to-1 by name
    # -------------------------------------------------------------------------
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping