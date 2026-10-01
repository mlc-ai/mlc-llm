"""Weight mapping from Hugging Face to MLC for the Qwen3.5 vision language model."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen35_loader import map_language_model, map_remaining
from .qwen35v_model import Qwen35VConfig, Qwen35VForCausalLM


def _interpolate_pos_embed(pos_weight, src: int, tgt: int):
    """Resample a src by src grid of position embeddings to tgt by tgt bilinearly, raster order."""
    pos = pos_weight.reshape(src, src, -1)
    coords = np.linspace(0, src - 1, tgt)
    lo = np.floor(coords).astype(np.int64)
    hi = np.minimum(lo + 1, src - 1)
    t = (coords - lo).astype(pos.dtype)[:, None]
    rows = pos[lo] * (1 - t)[:, :, None] + pos[hi] * t[:, :, None]
    out = rows[:, lo] * (1 - t)[None, :, :] + rows[:, hi] * t[None, :, :]
    return out.reshape(tgt * tgt, -1)


def huggingface(model_config: Qwen35VConfig, quantization: Quantization) -> ExternMapping:
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
    mlc_lm = "language_model.model"
    map_language_model(
        mapping, named_parameters, model_config.text_config, "model.language_model", mlc_lm
    )

    # The reference patch embedding is a Conv3D over two identical temporal frames, so
    # summing its weight over that axis gives the equivalent Conv2D.
    conv_mlc = "visual.patch_embed.proj.weight"
    if conv_mlc in named_parameters:
        mapping.add_mapping(
            conv_mlc,
            ["model.visual.patch_embed.proj.weight"],
            functools.partial(
                lambda w, dtype: w.sum(axis=2).astype(dtype),
                dtype=named_parameters[conv_mlc].dtype,
            ),
        )

    pos_mlc = "visual.pos_embed"
    if pos_mlc in named_parameters:
        mapping.add_mapping(
            pos_mlc,
            ["model.visual.pos_embed.weight"],
            functools.partial(
                lambda w, dtype, src, tgt: _interpolate_pos_embed(w, src, tgt).astype(dtype),
                dtype=named_parameters[pos_mlc].dtype,
                src=int(model_config.vision_config.num_position_embeddings**0.5),
                tgt=model_config.grid,
            ),
        )

    for module in [f"visual.blocks.{i}.mlp" for i in range(model_config.vision_config.depth)] + [
        "visual.merger"
    ]:
        for fc in ("fc1", "fc2"):
            for suffix in ("weight", "bias"):
                mlc_name = f"{module}.{fc}.{suffix}"
                if mlc_name in named_parameters:
                    mapping.add_mapping(
                        mlc_name,
                        [f"model.{module}.linear_{fc}.{suffix}"],
                        functools.partial(
                            lambda x, dtype: x.astype(dtype),
                            dtype=named_parameters[mlc_name].dtype,
                        ),
                    )

    map_remaining(mapping, named_parameters, _mlc_to_hf, mlc_lm)
    return mapping


def _mlc_to_hf(mlc_name: str) -> str:
    if mlc_name.startswith("language_model.model."):
        return "model.language_model." + mlc_name[len("language_model.model.") :]
    if mlc_name.startswith("language_model.lm_head."):
        return mlc_name[len("language_model.") :]
    if mlc_name.startswith("visual."):
        return "model." + mlc_name
    return mlc_name
