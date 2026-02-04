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
from mlc_llm.quantization import QUANTIZATION, AWQQuantize


def dequantize_np(
    config: AWQQuantize,
    weight: np.ndarray,
    zeros: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    def decode_int_arr(int_arr: np.ndarray, num_elem_per_storage: int, bits: int):
        bin_mask = (1 << bits) - 1
        int_arr_repeated = np.repeat(int_arr, num_elem_per_storage, axis=-1)
        indice_j = np.indices(int_arr_repeated.shape)[1]
        arr_bin = np.bitwise_and(
            np.right_shift(
                int_arr_repeated,
                (indice_j % num_elem_per_storage) * bits,
            ),
            bin_mask,
        )
        return arr_bin

    weight_bin = decode_int_arr(
        weight, config.num_elem_per_storage, DataType(config.quantize_dtype).bits
    )
    zero_bin = decode_int_arr(
        zeros, config.num_elem_per_storage, DataType(config.quantize_dtype).bits
    )
    scale_repeated = np.repeat(scale, config.group_size, axis=-1)
    zero_bin_repeated = np.repeat(zero_bin, config.group_size, axis=-1)
    return (weight_bin - zero_bin_repeated) * scale_repeated


@pytest.mark.parametrize(
    "quant_name, shape, dtype",
    [
        ("q4f16_awq", [2, 4096], "float16"),
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
    assert isinstance(config, AWQQuantize)
    weight_np = np.random.randint(
        np.iinfo(config.storage_dtype).min,
        np.iinfo(config.storage_dtype).max,
        (shape[0], shape[1] // config.num_elem_per_storage),
    ).astype(config.storage_dtype)
    zeros_np = np.random.randint(
        np.iinfo(config.storage_dtype).min,
        np.iinfo(config.storage_dtype).max,
        (shape[0], shape[1] // config.num_elem_per_storage // config.group_size),
    ).astype(config.storage_dtype)
    scale_np = np.random.random((shape[0], shape[1] // config.group_size)).astype(
        config.model_dtype
    )
    mod = config.quantize_model(Test(), QuantizeMapping({}, {}), "")
    mod.linear.qweight.data = weight_np
    mod.linear.qzeros.data = zeros_np
    mod.linear.scales.data = scale_np
    model = mod.jit(spec={"forward": {"x": nn.spec.Tensor((shape[1], shape[1]), dtype)}})
    out = model["forward"](
        torch.from_numpy(np.diag(np.ones(shape[1]).astype(dtype)))  # pylint: disable=no-member
    )
    ref = dequantize_np(config, weight_np, zeros_np, scale_np).T
    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_dequantize_weight("q4f16_awq", [2, 4096], "float16")
