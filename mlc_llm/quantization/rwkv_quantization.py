from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from tvm import relax, te, tir, topi

from .quantization import QuantizationSpec
from .quantization import FQuantize, FDequantize


@dataclass
class RWKVQuantizationSpec(QuantizationSpec):
    """The quantization specification for RWKV quantization algorithm."""

    mode: Literal["uint8"]
    nbit: Literal[8]

    def get_quantize_func(
        self, param_info: relax.TensorStructInfo
    ) -> Optional[FQuantize]:
        return encoding_func(self.dtype, self.mode, self.nbit)

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FDequantize]:
        return decoding_func(self.dtype)


def encoding_func(input_dtype: str, quant_dtype: str, nbit: int):
    assert (input_dtype, quant_dtype) == ("float16", "uint8")
    assert nbit == 8
    max_value = 1 << nbit
    assert nbit % 2 == 0
    scale = 1 << (nbit // 2)

    def encode(
        weight: te.Tensor,
    ) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
        weight = weight.astype("float32")
        if weight.shape[0] > weight.shape[1]:
            min_y = topi.min(weight, axis=1, keepdims=True)
            weight = weight - min_y
            min_x = topi.min(weight, axis=0, keepdims=True)
            weight = weight - min_x
        else:
            min_x = topi.min(weight, axis=0, keepdims=True)
            weight = weight - min_x
            min_y = topi.min(weight, axis=1, keepdims=True)
            weight = weight - min_y
        max_x = topi.max(weight, axis=0, keepdims=True)
        weight = weight / max_x
        max_y = topi.max(weight, axis=1, keepdims=True)
        weight = weight / max_y
        weight = topi.clip(topi.floor(weight * max_value), 0, max_value - 1).astype(
            quant_dtype
        )
        min_x = min_x.astype(input_dtype)
        min_y = min_y.astype(input_dtype)
        max_x = topi.divide(max_x, scale).astype(input_dtype)
        max_y = topi.divide(max_y, scale).astype(input_dtype)
        return weight, min_x, max_x, min_y, max_y

    return encode


def decoding_func(input_dtype):
    assert input_dtype == "float16"

    def decode(
        weight: te.Tensor,
        min_x: te.Tensor,
        max_x: te.Tensor,
        min_y: te.Tensor,
        max_y: te.Tensor,
    ) -> te.Tensor:
        x = weight.astype(input_dtype) + tir.const(0.5, input_dtype)
        return x * max_y * max_x + min_y + min_x

    return decode
