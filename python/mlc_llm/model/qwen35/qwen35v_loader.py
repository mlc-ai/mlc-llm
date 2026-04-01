"""Weight mapping from HuggingFace to MLC for Qwen3.5 Vision-Language model.

HF weight layout:
  Language model weights are under model.language_model.*
  Vision encoder weights are under model.visual.*

MLC layout:
  Language model: language_model.model.* (wrapping Qwen35LMHeadModel)
  Vision encoder: visual.*
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen35v_model import Qwen35VConfig, Qwen35VForCausalLM


def _interpolate_pos_embed(pos_weight, src_grid, tgt_h, tgt_w):
    """Bilinear interpolation from src_grid x src_grid to tgt_h x tgt_w, raster order.

    pos_weight: (src_grid*src_grid, hidden) numpy array
    Returns: (tgt_h*tgt_w, hidden) numpy array
    """
    hidden = pos_weight.shape[1]
    pos_2d = pos_weight.reshape(src_grid, src_grid, hidden)

    h_coords = np.linspace(0, src_grid - 1, tgt_h)
    w_coords = np.linspace(0, src_grid - 1, tgt_w)

    h_floor = np.floor(h_coords).astype(np.int64)
    w_floor = np.floor(w_coords).astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, src_grid - 1)
    w_ceil = np.minimum(w_floor + 1, src_grid - 1)
    dh = (h_coords - h_floor).astype(np.float32)
    dw = (w_coords - w_floor).astype(np.float32)

    result = np.zeros((tgt_h, tgt_w, hidden), dtype=pos_weight.dtype)
    for i in range(tgt_h):
        for j in range(tgt_w):
            result[i, j] = (
                (1 - dh[i]) * (1 - dw[j]) * pos_2d[h_floor[i], w_floor[j]]
                + (1 - dh[i]) * dw[j] * pos_2d[h_floor[i], w_ceil[j]]
                + dh[i] * (1 - dw[j]) * pos_2d[h_ceil[i], w_floor[j]]
                + dh[i] * dw[j] * pos_2d[h_ceil[i], w_ceil[j]]
            )

    return result.reshape(tgt_h * tgt_w, hidden)


def huggingface(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    model_config: Qwen35VConfig, quantization: Quantization
) -> ExternMapping:
    """Returns parameter mapping from MLC LLM parameters to HuggingFace parameters."""
    model = Qwen35VForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    mapping = ExternMapping()

    text_config = model_config.text_config

    # HF prefix for language model
    hf_lm = "model.language_model"
    # MLC prefix for language model (wrapped inside VLM)
    mlc_lm = "language_model.model"

    layer_types = text_config.layer_types()
    for i in range(text_config.num_hidden_layers):
        if layer_types[i] == "full_attention":
            # Standard attention: fuse Q/K/V into c_attn
            mlc_attn = f"{mlc_lm}.layers.{i}.self_attn"
            hf_attn = f"{hf_lm}.layers.{i}.self_attn"
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
            mlc_lin = f"{mlc_lm}.layers.{i}.linear_attn"
            hf_lin = f"{hf_lm}.layers.{i}.linear_attn"

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
                        functools.partial(
                            lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype
                        ),
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
        mlc_mlp = f"{mlc_lm}.layers.{i}.mlp"
        hf_mlp = f"{hf_lm}.layers.{i}.mlp"
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

    # ========== Vision encoder weights ==========
    vision_config = model_config.vision_config

    # Conv3D -> Conv2D: sum over temporal dimension.
    # HF uses nn.Conv3d(3, hidden, (2, 16, 16)) for video support.
    # For single images, summing over temporal dim gives equivalent Conv2D.
    # HF shape: (out_channels, in_channels, temporal=2, patch, patch)
    # MLC shape: (out_channels, in_channels, patch, patch)
    conv_mlc = "visual.patch_embed.proj.weight"
    conv_hf = "model.visual.patch_embed.proj.weight"
    if conv_mlc in named_parameters:
        mapping.add_mapping(
            conv_mlc,
            [conv_hf],
            functools.partial(
                lambda w, dtype: w.sum(axis=2).astype(dtype),
                dtype=named_parameters[conv_mlc].dtype,
            ),
        )

    # Position embedding: bilinear interpolation from 48x48 to grid_h x grid_w.
    pos_mlc = "visual.pos_embed"
    pos_hf = "model.visual.pos_embed.weight"
    if pos_mlc in named_parameters:
        src_grid = int(vision_config.num_position_embeddings**0.5)  # 48
        tgt_h = model_config.grid_h
        tgt_w = model_config.grid_w
        mapping.add_mapping(
            pos_mlc,
            [pos_hf],
            functools.partial(
                lambda w, dtype, sg, th, tw: _interpolate_pos_embed(w, sg, th, tw).astype(dtype),
                dtype=named_parameters[pos_mlc].dtype,
                sg=src_grid,
                th=tgt_h,
                tw=tgt_w,
            ),
        )

    # Vision blocks: rename HF linear_fc1/linear_fc2 -> MLC fc1/fc2
    for i in range(vision_config.depth):
        for suffix in ["weight", "bias"]:
            for fc_hf, fc_mlc in [("linear_fc1", "fc1"), ("linear_fc2", "fc2")]:
                mlc_name = f"visual.blocks.{i}.mlp.{fc_mlc}.{suffix}"
                hf_name = f"model.visual.blocks.{i}.mlp.{fc_hf}.{suffix}"
                if mlc_name in named_parameters:
                    mapping.add_mapping(
                        mlc_name,
                        [hf_name],
                        functools.partial(
                            lambda x, dtype: x.astype(dtype),
                            dtype=named_parameters[mlc_name].dtype,
                        ),
                    )

    # Merger: rename HF linear_fc1/linear_fc2 -> MLC fc1/fc2
    for suffix in ["weight", "bias"]:
        for fc_hf, fc_mlc in [("linear_fc1", "fc1"), ("linear_fc2", "fc2")]:
            mlc_name = f"visual.merger.{fc_mlc}.{suffix}"
            hf_name = f"model.visual.merger.{fc_hf}.{suffix}"
            if mlc_name in named_parameters:
                mapping.add_mapping(
                    mlc_name,
                    [hf_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=named_parameters[mlc_name].dtype,
                    ),
                )

    # ========== Remaining weights: 1:1 mapping with prefix translation ==========
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
            or name == f"{mlc_lm}.norm.weight"
        )

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


def _mlc_to_hf(mlc_name: str) -> str:
    """Convert MLC parameter name to HuggingFace parameter name."""
    # Language model: language_model.model.X -> model.language_model.X
    if mlc_name.startswith("language_model.model."):
        return "model.language_model." + mlc_name[len("language_model.model."):]
    if mlc_name.startswith("language_model.lm_head."):
        return "model.language_model." + mlc_name[len("language_model."):]
    # Vision: visual.X -> model.visual.X
    if mlc_name.startswith("visual."):
        return "model." + mlc_name
    return mlc_name
