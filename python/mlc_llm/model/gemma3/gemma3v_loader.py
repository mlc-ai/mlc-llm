"""
This file specifies how MLC's Gemma3V parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gemma3v_model import Gemma3VConfig, Gemma3VForCausalLM


def huggingface(  # pylint: disable=too-many-locals
    model_config: Gemma3VConfig, quantization: Quantization
) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Gemma3VConfig
        The configuration of the Gemma3V model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Gemma3VForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # ========== Language model weights ==========
    # Same pattern as gemma3_loader.py but with fixed hf_prefix = "language_model."
    mlc_prefix = "language_model."
    hf_prefix = "language_model."

    for i in range(model_config.text_config.num_hidden_layers):
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{mlc_prefix + mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{hf_prefix + mlp}.gate_proj.weight",
                f"{hf_prefix + mlp}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

        # Modify RMS layernorm weights (Gemma adds 1 to the weights)
        for norm_name in [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.pre_feedforward_layernorm.weight",
            f"model.layers.{i}.post_feedforward_layernorm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
            f"model.layers.{i}.self_attn.q_norm.weight",
        ]:
            mlc_param = named_parameters[mlc_prefix + norm_name]
            mapping.add_mapping(
                mlc_prefix + norm_name,
                [hf_prefix + norm_name],
                functools.partial(
                    lambda x, dtype: (x + 1).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    # Final norm +1
    mlc_name = "model.norm.weight"
    mlc_param = named_parameters[mlc_prefix + mlc_name]
    mapping.add_mapping(
        mlc_prefix + mlc_name,
        [hf_prefix + mlc_name],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=mlc_param.dtype,
        ),
    )

    # ========== Multimodal projector weights ==========
    # mm_input_projection: HF stores as (vision_hidden, text_hidden), nn.Linear expects
    # (text_hidden, vision_hidden) -> TRANSPOSE required
    proj_weight_mlc = "multi_modal_projector.mm_input_projection.weight"
    proj_weight_hf = "multi_modal_projector.mm_input_projection_weight"
    mlc_param = named_parameters[proj_weight_mlc]
    mapping.add_mapping(
        proj_weight_mlc,
        [proj_weight_hf],
        functools.partial(
            lambda x, dtype: x.T.astype(dtype),
            dtype=mlc_param.dtype,
        ),
    )

    # mm_soft_emb_norm: HF uses Gemma3RMSNorm which adds +1 at runtime, so we need +1 fusion
    norm_mlc = "multi_modal_projector.mm_soft_emb_norm.weight"
    mlc_param = named_parameters[norm_mlc]
    mapping.add_mapping(
        norm_mlc,
        [norm_mlc],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=mlc_param.dtype,
        ),
    )

    # ========== Remaining weights (vision tower + language model residuals) ==========
    # All vision_tower.* params and remaining language_model.* params map 1:1
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            # For language_model.* params, strip prefix and add hf_prefix
            if mlc_name.startswith(mlc_prefix):
                hf_name = hf_prefix + mlc_name[len(mlc_prefix) :]
            else:
                # vision_tower.* and other params map directly
                hf_name = mlc_name
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping
