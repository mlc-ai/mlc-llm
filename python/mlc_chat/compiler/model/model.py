"""A centralized registry of all existing model architures and their configurations."""
import dataclasses
from typing import Any, Callable, Dict, Tuple

from tvm.relax.frontend import nn

from ..loader import ExternMapping, QuantizeMapping
from ..quantization.quantization import Quantization
from . import llama_config, llama_loader, llama_model, llama_quantization

ModelConfig = Any
"""A ModelConfig is an object that represents a model architecture. It is required to have
a class method `from_file` with the following signature:

    def from_file(cls, path: Path) -> ModelConfig:
        ...
"""

FuncGetExternMap = Callable[[ModelConfig, Quantization], ExternMapping]
FuncQuantization = Callable[[ModelConfig, Quantization], Tuple[nn.Module, QuantizeMapping]]


@dataclasses.dataclass
class Model:
    """All about a model architecture: its configuration, its parameter loader and quantization.

    Parameters
    ----------
    name : str
        The name of the model.

    model : Callable[[ModelConfig], nn.Module]
        A method that creates the `nn.Module` that represents the model from `ModelConfig`.

    config : ModelConfig
        A class that has a `from_file` class method, whose signature is "Path -> ModelConfig".

    source : Dict[str, FuncGetExternMap]
        A dictionary that maps the name of a source format to parameter mapping.

    quantize: Dict[str, FuncQuantization]
        A dictionary that maps the name of a quantization method to quantized model and the
        quantization parameter mapping.
    """

    name: str
    config: ModelConfig
    model: Callable[[ModelConfig], nn.Module]
    source: Dict[str, FuncGetExternMap]
    quantize: Dict[str, FuncQuantization]


MODELS: Dict[str, Model] = {
    "llama": Model(
        name="llama",
        model=llama_model.LlamaForCasualLM,
        config=llama_config.LlamaConfig,
        source={
            "huggingface-torch": llama_loader.huggingface,
            "huggingface-safetensor": llama_loader.huggingface,
        },
        quantize={
            "group-quant": llama_quantization.group_quant,
        },
    )
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = llama_config.CONFIG
