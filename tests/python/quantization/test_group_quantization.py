# pylint: disable=invalid-name,missing-docstring
from typing import List

import numpy as np
import pytest
import torch
import tvm
import tvm.testing
from tvm import DataType
from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.quantization.group_quantization import (
    GroupQuantize,
    GroupQuantizeEmbedding,
    GroupQuantizeLinear,
)


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
    weight_filtered = np.reshape(weight_scaled_reshaped, (n, k))
    weight_filtered[..., weight.shape[1] :] = 0
    weight_scaled = np.reshape(
        weight_filtered, (n, k // config.num_elem_per_storage, config.num_elem_per_storage)
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
    assert weight.shape[0] == scale.shape[0]
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
            (indice_j % config.num_elem_per_storage) * DataType(config.quantize_dtype).bits,
        ),
        bin_mask,
    )
    assert weight_bin.shape[1] <= scale_repeated.shape[1]
    return ((weight_bin - max_int) * scale_repeated[..., : weight_bin.shape[1]])[
        : out_shape[0], : out_shape[1]
    ]


@pytest.mark.parametrize(
    "quant_name, shape, dtype, device",
    [
        ("q3f16_1", [2, 13], "float16", "cpu"),
        ("q3f16_1", [16, 120], "float16", "cpu"),
        ("q4f16_1", [2, 13], "float16", "cpu"),
        ("q4f16_1", [16, 128], "float16", "cpu"),
        ("q4f32_1", [2, 13], "float32", "cpu"),
        ("q4f32_1", [16, 128], "float32", "cpu"),
    ],
)
def test_quantize_weight(quant_name: str, shape: List[int], dtype: str, device: str):
    config = QUANTIZATION[quant_name]
    assert isinstance(config, GroupQuantize)
    weight_np = np.random.random(shape).astype(dtype)
    output = config.quantize_weight(tvm.nd.array(weight_np, device=tvm.device(device)))
    quantized_weight, scale = output[0].numpy(), output[1].numpy()
    quantized_weight_ref, scale_ref = quantize_np(config, weight_np)
    tvm.testing.assert_allclose(scale, scale_ref, rtol=1e-3, atol=1e-3)
    tvm.testing.assert_allclose(
        dequantize_np(config, quantized_weight, scale, shape),
        dequantize_np(config, quantized_weight_ref, scale_ref, shape),
        rtol=1e-2 if quant_name.startswith("q3") else 1e-3,
        atol=0.4 if quant_name.startswith("q3") else 0.2,
    )


@pytest.mark.parametrize(
    "quant_name, shape, dtype",
    [
        ("q3f16_1", [2, 13], "float16"),
        ("q3f16_1", [16, 120], "float16"),
        ("q4f16_1", [2, 13], "float16"),
        ("q4f16_1", [16, 128], "float16"),
        ("q4f32_1", [2, 13], "float32"),
        ("q4f32_1", [16, 128], "float32"),
    ],
)
def test_dequantize_weight(quant_name: str, shape: List[int], dtype: str):
    class Test(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(shape[1], shape[0], bias=False, dtype=dtype)

        def forward(self, x: nn.Tensor):
            return self.linear(x)

    config = QUANTIZATION[quant_name]
    assert isinstance(config, GroupQuantize)
    num_group = -(shape[1] // -config.group_size)
    weight_np = np.random.randint(
        np.iinfo(config.storage_dtype).min,
        np.iinfo(config.storage_dtype).max,
        (shape[0], config.num_storage_per_group * num_group),
    ).astype(config.storage_dtype)
    scale_np = np.random.random((shape[0], num_group)).astype(config.model_dtype)
    mod = config.quantize_model(Test(), QuantizeMapping({}, {}), "")
    mod.linear.q_weight.data = weight_np
    mod.linear.q_scale.data = scale_np
    model = mod.jit(spec={"forward": {"x": nn.spec.Tensor((shape[1], shape[1]), dtype)}})
    out = model["forward"](
        torch.from_numpy(np.diag(np.ones(shape[1]).astype(dtype)))  # pylint: disable=no-member
    )
    ref = dequantize_np(config, weight_np, scale_np, shape).T
    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "quant_name, shape, dtype",
    [
        ("q3f16_1", [16, 128], "float16"),
        ("q4f16_1", [16, 128], "float16"),
        ("q4f32_1", [16, 128], "float32"),
    ],
)
def test_quantize_model(quant_name: str, shape: List[int], dtype: str):
    class Test(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(shape[0], shape[1], dtype=dtype)
            self.embedding = nn.Embedding(shape[0], shape[1], dtype=dtype)

        def forward(self, x: nn.Tensor):
            return self.linear(x)

    config = QUANTIZATION[quant_name]
    assert isinstance(config, GroupQuantize)
    quant_map = QuantizeMapping({}, {})
    mod = config.quantize_model(Test(), quant_map, "model")
    assert quant_map.param_map["model.linear.weight"] == [
        "model.linear.q_weight",
        "model.linear.q_scale",
    ]
    assert quant_map.map_func["model.linear.weight"] == config.quantize_weight
    assert isinstance(mod.linear, GroupQuantizeLinear)
    assert quant_map.param_map["model.embedding.weight"] == [
        "model.embedding.q_weight",
        "model.embedding.q_scale",
    ]
    assert quant_map.map_func["model.embedding.weight"] == config.quantize_weight
    assert isinstance(mod.embedding, GroupQuantizeEmbedding)


if __name__ == "__main__":
    test_quantize_weight("q4f16_1", [16, 128], "float16", "llvm")
    test_quantize_model("q4f16_1", [16, 128], "float16")
    test_dequantize_weight("q4f16_1", [16, 128], "float16")
