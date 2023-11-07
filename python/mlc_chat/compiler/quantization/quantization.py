"""A centralized registry of all existing quantization methods and their configurations."""
from typing import Any, Dict

from .awq_quantization import AWQQuantize
from .group_quantization import GroupQuantize

Quantization = Any
"""Quantization is an object that represents an quantization algorithm. It is required to
have the following fields:

    name : str
        The name of the quantization algorithm, for example, "q4f16_1".

    kind : str
        The kind of quantization algorithm, for example, "group-quant", "faster-transformer".

It is also required to have the following method:

    def quantize_model(self, module: nn.Module) -> nn.Module:
        ...

    def quantize_weight(self, weight: tvm.runtime.NDArray) -> List[tvm.runtime.NDArray]:
        ...
"""

QUANTIZATION: Dict[str, Quantization] = {
    "q4f16_1": GroupQuantize(
        name="q4f16_1",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
    ),
    "q4f16_awq": AWQQuantize(
        name="q4f16_awq",
        kind="awq",
        group_size=128,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
    ),
}
