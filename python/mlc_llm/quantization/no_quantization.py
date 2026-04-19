"""The no quantization config"""

from dataclasses import dataclass


@dataclass
class NoQuantize:
    """Configuration for no quantization"""

    name: str
    kind: str
    model_dtype: str  # "float16", "float32"

    def __post_init__(self):
        assert self.kind == "no-quant"
