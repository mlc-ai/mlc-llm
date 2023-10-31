# pylint: disable=missing-docstring
from typing import List

import numpy as np
import tvm
import tvm.testing
from mlc_chat.compiler import QUANTIZATION
from mlc_chat.compiler.quantization import GroupQuantize
from tvm import DataType


def quantize_np(config: GroupQuantize, weight: np.ndarray):
    n, k = weight.shape
    weight_padded = np.pad(
        weight, ((0, 0), (0, (config.group_size - k % config.group_size) % config.group_size))
    )
    n, k = weight_padded.shape
    weight_reshaped = np.reshape(weight_padded, (n, k // config.group_size, config.group_size))
    max_abs = np.maximum(np.max(np.abs(weight_reshaped), axis=-1), 1e-4)
    scale = np.divide(max_abs, config.max_int_value)
    scale_reshaped = np.reshape(scale, (*scale.shape, 1))
    weight_scaled_reshaped = np.clip(
        np.add(
            np.round(np.divide(weight_reshaped, scale_reshaped)),
            config.max_int_value,
        ),
        0,
        config.max_int_value * 2,
    ).astype(config.storage_dtype)
    weight_scaled = np.reshape(
        weight_scaled_reshaped, (n, k // config.num_elem_per_storage, config.num_elem_per_storage)
    )
    indice_k = np.indices(weight_scaled.shape, dtype=config.storage_dtype)[-1]
    quantized_weight = np.sum(
        np.left_shift(weight_scaled, indice_k * DataType(config.quantize_dtype).bits),
        axis=-1,
        dtype=config.storage_dtype,
    )
    return quantized_weight, scale


def dequantize_np(
    config: GroupQuantize,
    weight: np.ndarray,
    scale: np.ndarray,
    out_shape: List[int] = None,
):
    bin_mask = (1 << DataType(config.quantize_dtype).bits) - 1
    max_int = config.max_int_value
    out_shape = (
        [weight.shape[0], weight.shape[1] * config.num_elem_per_storage]
        if out_shape is None
        else out_shape
    )
    weight_repeated = np.repeat(weight, config.num_elem_per_storage, axis=-1)
    scale_repeated = np.repeat(scale, config.group_size, axis=-1)
    indice_j = np.indices(weight_repeated.shape)[1]
    weight_bin = np.bitwise_and(
        np.right_shift(
            weight_repeated,
            (indice_j % config.num_elem_per_storage) * DataType(config.storage_dtype).bits,
        ),
        bin_mask,
    )
    return ((weight_bin - max_int) * scale_repeated)[: out_shape[0]][: out_shape[1]]


def test_quantize(quant_name: str, shape: List[int], dtype: str):
    config = QUANTIZATION[quant_name]
    assert isinstance(config, GroupQuantize)
    weight_np = np.random.random(shape).astype(dtype)
    output = config.quantize_weight(tvm.nd.array(weight_np, device=tvm.device("cuda")))
    quantized_weight, scale = output[0].numpy(), output[1].numpy()
    quantized_weight_ref, scale_ref = quantize_np(config, weight_np)
    tvm.testing.assert_allclose(scale, scale_ref, rtol=1e-3, atol=1e-3)
    tvm.testing.assert_allclose(
        dequantize_np(config, quantized_weight, scale, shape),
        dequantize_np(config, quantized_weight_ref, scale_ref, shape),
        rtol=1e-3,
        atol=0.2,
    )


if __name__ == "__main__":
    test_quantize("q4f16_1", [64, 4096], "float16")
