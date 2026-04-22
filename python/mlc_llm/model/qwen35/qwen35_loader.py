"""
HuggingFace parameter mapping for Qwen3.5 GatedDeltaNet.

Qwen3.5 is a VLM — HF weights are nested under `model.language_model.`.
Our MLC model uses `model.` prefix. The mapping must translate between them.

HF weight layout (under model.language_model.):
  Linear attention layers:
    model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight
    model.language_model.layers.{i}.linear_attn.in_proj_z.weight
    model.language_model.layers.{i}.linear_attn.in_proj_a.weight
    model.language_model.layers.{i}.linear_attn.in_proj_b.weight
    model.language_model.layers.{i}.linear_attn.out_proj.weight
    model.language_model.layers.{i}.linear_attn.conv1d.weight
    model.language_model.layers.{i}.linear_attn.norm.weight
    model.language_model.layers.{i}.linear_attn.A_log        (NO .weight suffix)
    model.language_model.layers.{i}.linear_attn.dt_bias      (NO .weight suffix)

  Full attention layers:
    model.language_model.layers.{i}.self_attn.q_proj.weight
    ...

  Vision/MTP weights are ignored (text backbone only).
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen35_model import Qwen35Config, Qwen35LMHeadModel


def huggingface(model_config: Qwen35Config, quantization: Quantization) -> ExternMapping:
    model = Qwen35LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # HF prefix: Qwen3.5 is a VLM, text weights nested under model.language_model.
    # MLC model uses model. prefix directly.
    hf = "model.language_model"

    layer_types = model_config.layer_types()
    for i in range(model_config.num_hidden_layers):
        if layer_types[i] == "full_attention":
            # Standard attention: fuse Q/K/V into c_attn
            mlc_attn = f"model.layers.{i}.self_attn"
            hf_attn = f"{hf}.layers.{i}.self_attn"
            mlc_name = f"{mlc_attn}.c_attn.weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [
                        f"{hf_attn}.q_proj.weight",
                        f"{hf_attn}.k_proj.weight",
                        f"{hf_attn}.v_proj.weight",
                    ],
                    functools.partial(
                        lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
        else:
            # Linear attention layer
            mlc_lin = f"model.layers.{i}.linear_attn"
            hf_lin = f"{hf}.layers.{i}.linear_attn"

            # in_proj_qkv — maps directly (already fused in HF)
            mlc_name = f"{mlc_lin}.in_proj_qkv.weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_lin}.in_proj_qkv.weight"],
                    functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                )

            # A_log and dt_bias — no .weight suffix in HF
            for param_name in ["A_log", "dt_bias"]:
                mlc_name = f"{mlc_lin}.{param_name}"
                if mlc_name in named_parameters:
                    mlc_param = named_parameters[mlc_name]
                    mapping.add_mapping(
                        mlc_name,
                        [f"{hf_lin}.{param_name}"],
                        functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                    )

            # conv1d weight
            mlc_name = f"{mlc_lin}.conv1d_weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_lin}.conv1d.weight"],
                    functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                )

        # MLP: fuse gate_proj + up_proj
        mlc_mlp = f"model.layers.{i}.mlp"
        hf_mlp = f"{hf}.layers.{i}.mlp"
        mlc_name = f"{mlc_mlp}.gate_up_proj.weight"
        if mlc_name in named_parameters:
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{hf_mlp}.gate_proj.weight",
                    f"{hf_mlp}.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    def _mlc_to_hf(mlc_name: str) -> str:
        """Convert MLC param name to HF param name by adding language_model prefix."""
        if mlc_name.startswith("model."):
            return mlc_name.replace("model.", f"{hf}.", 1)
        return mlc_name

    def _is_rmsnorm_weight(name: str) -> bool:
        """Check if a parameter is an RMSNorm weight that needs +1.0 offset.

        Qwen3_5RMSNorm uses: output = norm(x) * (1.0 + weight)
          - input_layernorm, post_attention_layernorm, model.norm, q_norm, k_norm
        Qwen3_5RMSNormGated uses: output = norm(x) * weight * silu(gate)
          - linear_attn.norm (gated norm) — does NOT get +1
        """
        return (
            name.endswith("input_layernorm.weight")
            or name.endswith("post_attention_layernorm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or name == "model.norm.weight"
        )

    # All remaining parameters: direct 1:1 mapping with HF prefix
    # Qwen3.5 uses a non-standard RMSNorm: output = norm(x) * (1.0 + weight)
    # Weights are initialized to zeros and learned as offsets from 1.0.
    # TVM's nn.RMSNorm uses: output = norm(x) * weight
    # So we add 1.0 to all RMSNorm weights during loading.
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            hf_name = _mlc_to_hf(mlc_name)
            if _is_rmsnorm_weight(mlc_name):
                mapping.add_mapping(
                    mlc_name,
                    [hf_name],
                    functools.partial(
                        lambda x, dtype: (x.astype("float32") + 1.0).astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
            else:
                mapping.add_mapping(
                    mlc_name,
                    [hf_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )

    return mapping
