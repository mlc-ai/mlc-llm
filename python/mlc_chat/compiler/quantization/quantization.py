"""A centralized registry of all existing quantization methods and their configurations."""
from typing import Any, Dict

QuantizeConfig = Any
"""A QuantizeConfig is an object that represents an quantization algorithm. It is required to
have the following fields:

    name : str
        The name of the quantization algorithm, for example, "q4f16_1".

    kind : str
        The kind of quantization algorithm, for example, "group_quant", "faster_transformer".

It is also required to have the following method:

    def quantize(self, module: nn.Module) -> nn.Module:
        ...
"""

QUANT: Dict[str, QuantizeConfig] = {
    "q4f16_1": None,
}
