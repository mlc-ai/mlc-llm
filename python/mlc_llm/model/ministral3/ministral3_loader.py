"""
This file specifies how MLC's Ministral3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Optional, Tuple

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .ministral3_model import Ministral3Config, Mistral3ForConditionalGeneration
from .ministral3_quantization import awq_quant


def _dequantize_block_scale_weight(
    weight: np.ndarray, weight_scale: np.ndarray, block_size: Tuple[int, int]
) -> np.ndarray:
    """Reconstruct float weights from FP8 block-scale storage."""

    rows, cols = weight.shape
    block_rows, block_cols = block_size
    out = np.empty((rows, cols), dtype="float32")
    weight_f32 = weight.astype("float32")
    num_row_blocks, num_col_blocks = weight_scale.shape
    for i in range(num_row_blocks):
        row_start = i * block_rows
        if row_start >= rows:
            break
        row_end = min(row_start + block_rows, rows)
        scale_row = weight_scale[i]
        for j in range(num_col_blocks):
            col_start = j * block_cols
            if col_start >= cols:
                break
            col_end = min(col_start + block_cols, cols)
            out[row_start:row_end, col_start:col_end] = (
                weight_f32[row_start:row_end, col_start:col_end] * scale_row[j]
            )
    return out


def huggingface(model_config: Ministral3Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Ministral3Config
        The configuration of the Ministral3 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Mistral3ForConditionalGeneration(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    raw_params = dict(_named_params)
    if any(name.startswith("language_model.") for name in raw_params):
        named_parameters = {
            name.replace("language_model.", "", 1): value
            for name, value in raw_params.items()
        }
    else:
        named_parameters = raw_params

    mapping = ExternMapping()

    hf_prefix = ""
    if "vision_config" in model_config.kwargs:
        hf_prefix = "language_model."

    def hf(name: str) -> str:
        return f"{hf_prefix}{name}"

    weight_block_size = getattr(model_config, "weight_block_size", None)
    if weight_block_size is not None:
        weight_block_size = tuple(int(x) for x in weight_block_size)
    needs_block_dequant = (
        weight_block_size is not None and quantization.kind != "block-scale-quant"
    )

    def convert_linear_weight(
        weight: np.ndarray,
        scale: Optional[np.ndarray],
        dtype: str,
        source_name: str,
    ):
        if needs_block_dequant:
            if scale is None:
                raise ValueError(f"Missing block-scale metadata for {source_name}.")
            assert weight_block_size is not None
            rows, cols = weight.shape
            block_rows, block_cols = weight_block_size
            num_row_blocks = (rows + block_rows - 1) // block_rows
            num_col_blocks = (cols + block_cols - 1) // block_cols
            scale = np.asarray(scale, dtype="float32").reshape(-1)
            expected = num_row_blocks * num_col_blocks
            if scale.size == 1:
                scale = np.broadcast_to(scale, expected)
            elif scale.size != expected:
                raise ValueError(
                    f"{source_name} weight_scale_inv has {scale.size} elements but "
                    f"expected {expected} (rows={rows}, cols={cols}, block_size={weight_block_size})"
                )
            scale = scale.reshape(num_row_blocks, num_col_blocks)
            weight = _dequantize_block_scale_weight(weight, scale, weight_block_size)
        return weight.astype(dtype)

    def block_weight_sources(base_name: str):
        sources = [hf(base_name)]
        if needs_block_dequant:
            sources.append(hf(base_name.replace(".weight", ".weight_scale_inv")))
        return sources

    def make_concat_func(source_names: Tuple[str, ...], dtype: str):
        def func(*arrays):
            arr_iter = iter(arrays)
            converted = []
            for source in source_names:
                weight = next(arr_iter)
                scale = next(arr_iter) if needs_block_dequant else None
                converted.append(convert_linear_weight(weight, scale, dtype, hf(source)))
            return np.concatenate(converted, axis=0)

        return func

    def make_single_linear_func(source_name: str, dtype: str):
        def func(*arrays):
            weight = arrays[0]
            scale = arrays[1] if needs_block_dequant else None
            return convert_linear_weight(weight, scale, dtype, hf(source_name))

        return func

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        mlc_name = f"{attn}.qkv_proj.weight"
        mlc_param = named_parameters[mlc_name]
        proj_sources = tuple(f"{attn}.{proj}.weight" for proj in ["q_proj", "k_proj", "v_proj"])
        qkv_sources = []
        for source in proj_sources:
            qkv_sources.extend(block_weight_sources(source))
        mapping.add_mapping(
            mlc_name,
            qkv_sources,
            make_concat_func(proj_sources, mlc_param.dtype),
        )

        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        gate_proj_sources = tuple(f"{mlp}.{proj}.weight" for proj in ["gate_proj", "up_proj"])
        gate_sources = []
        for source in gate_proj_sources:
            gate_sources.extend(block_weight_sources(source))
        mapping.add_mapping(
            mlc_name,
            gate_sources,
            make_concat_func(gate_proj_sources, mlc_param.dtype),
        )

        for linear_name in [f"{attn}.o_proj.weight", f"{mlp}.down_proj.weight"]:
            mlc_param = named_parameters[linear_name]
            mapping.add_mapping(
                linear_name,
                block_weight_sources(linear_name),
                make_single_linear_func(linear_name, mlc_param.dtype),
            )
        
        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [hf(mlc_name)],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping


def awq(model_config: Ministral3Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : Ministral3Config
        The configuration of the Ministral3 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),  # type: ignore[attr-defined]
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    hf_prefix = ""
    if "vision_config" in model_config.kwargs:
        hf_prefix = "language_model."

    def hf(name: str) -> str:
        return f"{hf_prefix}{name}"

    mapping = ExternMapping()

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
                    hf(f"{attn}.q_proj.{quantize_suffix}"),
                    hf(f"{attn}.k_proj.{quantize_suffix}"),
                    hf(f"{attn}.v_proj.{quantize_suffix}"),
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # Concat gate and up in MLP
        mlp = f"model.layers.{i}.mlp"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            mlc_name = f"{mlp}.gate_up_proj.{quantize_suffix}"
            assert mlc_name in named_parameters
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    hf(f"{mlp}.gate_proj.{quantize_suffix}"),
                    hf(f"{mlp}.up_proj.{quantize_suffix}"),
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [hf(mlc_name)],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
            )
    return mapping
