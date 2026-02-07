"""
This file specifies how MLC's Ministral3 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools
from typing import Callable, List, Optional, Tuple

import numpy as np

from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization

from .ministral3_model import Ministral3Config, Mistral3ForConditionalGeneration


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
    if isinstance(quantization, BlockScaleQuantize):
        # Convert the model to block-scale quantized model before loading parameters
        model = quantization.quantize_model(model, QuantizeMapping({}, {}), "")
        if model_config.weight_block_size is None:
            raise ValueError(
                "The input Ministral 3 model is not fp8 block quantized. "
                "Thus BlockScaleQuantize is not supported."
            )

    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    raw_params = dict(_named_params)
    if any(name.startswith("language_model.") for name in raw_params):
        named_parameters = {
            name.replace("language_model.", "", 1): value for name, value in raw_params.items()
        }
    else:
        named_parameters = raw_params

    mapping = ExternMapping()

    hf_prefix = ""
    if "vision_config" in model_config.kwargs:
        hf_prefix = "language_model."

    def hf(name: str) -> str:
        return f"{hf_prefix}{name}"

    if (
        not isinstance(quantization, BlockScaleQuantize)
        and model_config.weight_block_size is not None
    ):
        raise ValueError(
            "The input Ministral 3 model is fp8 block quantized. "
            "Please use BlockScaleQuantize for the model."
        )

    # Helper function to add both weight and scale mappings
    def add_weight_and_scale_mapping(
        weight_mlc_name: str,
        weight_hf_names: List[str],
        weight_transform_func: Callable,
        activation_transform_func: Optional[Callable] = None,
    ):
        mlc_param = named_parameters[weight_mlc_name]
        mapping.add_mapping(
            weight_mlc_name,
            weight_hf_names,
            functools.partial(weight_transform_func, dtype=mlc_param.dtype),
        )

        if isinstance(quantization, BlockScaleQuantize):
            weight_scale_mlc_name = f"{weight_mlc_name}_scale_inv"
            if weight_scale_mlc_name in named_parameters:
                weight_scale_hf_names = [f"{name}_scale_inv" for name in weight_hf_names]
                weight_scale_param = named_parameters[weight_scale_mlc_name]
                expected_weight_scale_shape = tuple(int(dim) for dim in weight_scale_param.shape)

                def _weight_scale_transform(*arrays, dtype: str, _transform=weight_transform_func):
                    processed = []
                    for arr in arrays:
                        arr_np = np.asarray(arr)
                        if arr_np.ndim == 0:
                            arr_np = arr_np.reshape((1,))
                        processed.append(arr_np)
                    result = _transform(*processed, dtype=dtype)
                    result = np.asarray(result, dtype=dtype)
                    if result.shape == expected_weight_scale_shape:
                        return result
                    if result.shape == ():
                        return np.full(expected_weight_scale_shape, result.item(), dtype=dtype)
                    if result.shape == (1,) and expected_weight_scale_shape != (1,):
                        return np.broadcast_to(result, expected_weight_scale_shape).astype(dtype)
                    if (
                        result.ndim == 1
                        and result.size > 1
                        and len(expected_weight_scale_shape) >= 2
                        and expected_weight_scale_shape[0] % result.size == 0
                    ):
                        rows_per_segment = expected_weight_scale_shape[0] // result.size
                        tiled = np.repeat(result, rows_per_segment)
                        tiled = tiled.reshape(expected_weight_scale_shape[0], 1)
                        return np.broadcast_to(tiled, expected_weight_scale_shape).astype(dtype)
                    raise ValueError(
                        f"Unexpected weight scale shape {result.shape} for "
                        f"{weight_scale_mlc_name}, expected {expected_weight_scale_shape}"
                    )

                mapping.add_mapping(
                    weight_scale_mlc_name,
                    weight_scale_hf_names,
                    functools.partial(_weight_scale_transform, dtype=weight_scale_param.dtype),
                )
            activation_scale_mlc_name = f"{weight_mlc_name[: -len('.weight')]}.activation_scale"
            if activation_scale_mlc_name in named_parameters:
                activation_scale_hf_names = [
                    f"{name[: -len('.weight')]}.activation_scale" for name in weight_hf_names
                ]
                activation_scale_param = named_parameters[activation_scale_mlc_name]
                transform = activation_transform_func or weight_transform_func
                expected_shape = tuple(int(dim) for dim in activation_scale_param.shape)

                def _activation_scale_transform(*arrays, dtype: str, _transform=transform):
                    result = _transform(*arrays, dtype=dtype)
                    result = np.asarray(result, dtype=dtype)
                    if result.shape == expected_shape:
                        return result
                    if result.shape == ():
                        # HF checkpoint stores a single scale; broadcast across the expected dimension.
                        return np.full(expected_shape, result.item(), dtype=dtype)
                    if result.shape == (1,) and expected_shape != (1,):
                        return np.broadcast_to(result, expected_shape).astype(dtype)
                    if (
                        result.ndim == 1
                        and result.size > 1
                        and len(expected_shape) >= 1
                        and expected_shape[0] % result.size == 0
                    ):
                        rows_per_segment = expected_shape[0] // result.size
                        tiled = np.repeat(result, rows_per_segment)
                        return tiled.reshape(expected_shape).astype(dtype)
                    raise ValueError(
                        f"Unexpected activation scale shape {result.shape} for "
                        f"{activation_scale_mlc_name}, expected {expected_shape}"
                    )

                mapping.add_mapping(
                    activation_scale_mlc_name,
                    activation_scale_hf_names,
                    functools.partial(
                        _activation_scale_transform, dtype=activation_scale_param.dtype
                    ),
                )

    def identity_transform(param: np.ndarray, dtype: str):
        return param.astype(dtype)

    def make_shared_activation_transform(target_name: str):
        def func(first: np.ndarray, *rest: np.ndarray, dtype: str):
            for idx, arr in enumerate(rest, start=1):
                if not np.allclose(arr, first):
                    raise ValueError(
                        f"Activation scales for {target_name} must be identical between "
                        "concatenated sources."
                    )
            return first.astype(dtype)

        return func

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        mlc_name = f"{attn}.qkv_proj.weight"
        proj_sources = [hf(f"{attn}.{proj}.weight") for proj in ["q_proj", "k_proj", "v_proj"]]
        add_weight_and_scale_mapping(
            mlc_name,
            proj_sources,
            lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
            activation_transform_func=make_shared_activation_transform(
                f"{mlc_name}_activation_scale"
            ),
        )

        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{mlp}.gate_up_proj.weight"
        gate_sources = [hf(f"{mlp}.{proj}.weight") for proj in ["gate_proj", "up_proj"]]
        add_weight_and_scale_mapping(
            mlc_name,
            gate_sources,
            lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
            activation_transform_func=make_shared_activation_transform(
                f"{mlc_name}_activation_scale"
            ),
        )

        for linear_name in [f"{attn}.o_proj.weight", f"{mlp}.down_proj.weight"]:
            add_weight_and_scale_mapping(
                linear_name,
                [hf(linear_name)],
                identity_transform,
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
