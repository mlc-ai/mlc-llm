# pylint: disable=invalid-name,missing-docstring
from typing import List

import numpy as np
import pytest
import safetensors
import torch
import tvm
import tvm.testing
from tvm import DataType
from tvm.relax.frontend import nn

from mlc_chat.compiler import QUANTIZATION
from mlc_chat.compiler.loader import QuantizeMapping
from mlc_chat.compiler.quantization import AWQQuantize
from mlc_chat.support.auto_device import detect_device
from mlc_chat.support.auto_target import detect_target_and_host


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
        inv_order_map = [0, 4, 1, 5, 2, 6, 3, 7]
        arr_bin = np.bitwise_and(
            np.right_shift(
                int_arr_repeated,
                inv_order_map[indice_j % num_elem_per_storage] * bits,
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


def test_dequantize_weight(quant_name: str, dtype: str):
    target, _ = detect_target_and_host("auto")
    device = detect_device("auto")
    # print(target)
    # print(device)
    with safetensors.safe_open(
        "../Llama-2-7B-AWQ/model.safetensors", framework="numpy", device="cpu"
    ) as in_file:
        q_proj_qweight = in_file.get_tensor("model.layers.0.self_attn.q_proj.qweight").astype(
            "uint32"
        )
        q_proj_qzeros = in_file.get_tensor("model.layers.0.self_attn.q_proj.qzeros").astype(
            "uint32"
        )
        q_proj_scales = in_file.get_tensor("model.layers.0.self_attn.q_proj.scales")
    hf_state_dict = torch.load("/opt/models/llama-2/llama-2-7b-hf/pytorch_model-00001-of-00002.bin")
    q_proj_weight = hf_state_dict["model.layers.0.self_attn.q_proj.weight"]
    print(q_proj_weight)

    class Test(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(
                q_proj_weight.shape[1], q_proj_weight.shape[0], bias=False, dtype=dtype
            )

        def forward(self, x: nn.Tensor):
            return self.linear(x)

    config = QUANTIZATION[quant_name]
    assert isinstance(config, AWQQuantize)
    mod = config.quantize_model(Test(), QuantizeMapping({}, {}), "")
    mod.linear.qweight.data = q_proj_qweight
    mod.linear.qzeros.data = q_proj_qzeros
    mod.linear.scales.data = q_proj_scales
    with target:
        model = mod.jit(
            spec={
                "forward": {
                    "x": nn.spec.Tensor((q_proj_weight.shape[1], q_proj_weight.shape[1]), dtype)
                }
            },
            pipeline="mlc_llm",
            device="cuda:0",
            target=target,
        )
    out = model["forward"](
        torch.from_numpy(np.diag(np.ones(q_proj_weight.shape[1]).astype(dtype))).to(
            "cuda:0"
        )  # pylint: disable=no-member
    )
    print(out)
    print(q_proj_weight.shape)
    print(out.shape)
    # out = dequantize_np(config, q_proj_qweight, q_proj_qzeros, q_proj_scales)
    print(torch.max(torch.abs(out.detach().to("cpu") - q_proj_weight)))


if __name__ == "__main__":
    test_dequantize_weight("q4f16_awq_gemm", "float16")
