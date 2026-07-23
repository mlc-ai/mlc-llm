"""HuggingFace parameter mapping for the Gemma4 text model."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.quantization import Quantization

from .gemma4_model import Gemma4Config, Gemma4ForCausalLM, Gemma4SplitScaledEmbedding


def huggingface(model_config: Gemma4Config, quantization: Quantization) -> ExternMapping:
    """Create HF weight mapping for Gemma4 text checkpoints."""

    model = Gemma4ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    def name_transform(name: str) -> str:
        if name.startswith("language_model."):
            name = name[len("language_model.") :]
        if name.startswith("model."):
            return f"model.language_{name}"
        return name

    def num_layers(config: object) -> int:
        return config.text_config.num_hidden_layers  # type: ignore[attr-defined]

    base_loader = make_standard_hf_loader(
        model_cls=Gemma4ForCausalLM,
        include_qkv=False,
        include_gate_up=True,
        gate_up_target_name="gate_up_proj",
        num_layers_getter=num_layers,
        layer_prefix="model.layers",
        name_transform=name_transform,
    )
    mapping = base_loader(model_config, quantization)

    # ---- embed_tokens scale folding ----
    # The TVM quantized-embedding fusion pass (`fused_dequantize_take`)
    # absorbs the post-lookup `* embed_scale` constant multiply into the
    # fused kernel — but the fused kernel only performs dequantize+take,
    # silently dropping the scale.  To compensate we pre-multiply the
    # embedding weights by the scale so that the dequantised values are
    # already correctly scaled.
    _model_dtype = quantization.model_dtype if quantization else "float16"

    # Main embedding: scale = sqrt(hidden_size)
    embed_scale = model_config.text_config.hidden_size ** 0.5
    mlc_embed_name = "model.embed_tokens.weight"
    hf_embed_name = name_transform("model.embed_tokens.weight")
    mapping.param_map.pop(mlc_embed_name, None)
    mapping.map_func.pop(mlc_embed_name, None)
    mapping.add_mapping(
        mlc_embed_name,
        [hf_embed_name],
        functools.partial(
            lambda w, sc=embed_scale, dt=_model_dtype: (w.astype("float32") * sc).astype(dt),
        ),
    )

    # Per-layer embeddings: scale = sqrt(hidden_size_per_layer_input)
    if model_config.text_config.hidden_size_per_layer_input:
        per_layer_scale = model_config.text_config.hidden_size_per_layer_input ** 0.5
        split_embed = model.language_model.model.embed_tokens_per_layer
        if isinstance(split_embed, Gemma4SplitScaledEmbedding):
            shard_dims = split_embed.shard_dims
            offsets = [0]
            for d in shard_dims:
                offsets.append(offsets[-1] + d)

            hf_source_name = name_transform(
                "language_model.model.embed_tokens_per_layer.weight"
            )
            for shard_idx, dim in enumerate(shard_dims):
                mlc_name = (
                    f"model.embed_tokens_per_layer"
                    f".shards.{shard_idx}.weight"
                )
                start = offsets[shard_idx]
                end = offsets[shard_idx + 1]
                mapping.param_map.pop(mlc_name, None)
                mapping.map_func.pop(mlc_name, None)
                mapping.add_mapping(
                    mlc_name,
                    [hf_source_name],
                    functools.partial(
                        lambda w, s=start, e=end, sc=per_layer_scale, dt=_model_dtype: (
                            w[:, s:e].astype("float32") * sc
                        ).astype(dt),
                    ),
                )

    # ---- layer_scalar zero-padding ----
    # Gemma 4 has a per-layer scalar with shape (1,) and dtype float16 = 2
    # bytes.  WebGPU requires storage buffers to be a multiple of 4 bytes, so
    # we pad the parameter to shape (2,) in the model definition.  Here we
    # add the zero-padding during weight conversion.
    n_layers = model_config.text_config.num_hidden_layers
    for i in range(n_layers):
        mlc_name = f"model.layers.{i}.layer_scalar"
        hf_name = name_transform(f"model.layers.{i}.layer_scalar")
        mapping.param_map.pop(mlc_name, None)
        mapping.map_func.pop(mlc_name, None)
        _dtype = quantization.model_dtype if quantization else "float16"
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda w, dt=_dtype: np.concatenate(
                    [w.astype(dt), np.zeros((1,), dtype=dt)],
                ),
            ),
        )

    return mapping
