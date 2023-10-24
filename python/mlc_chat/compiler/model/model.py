"""A centralized registry of all existing model architures and their configurations."""
import dataclasses
from typing import Any, Callable, Dict

from tvm.relax.frontend import nn

from ..parameter import ExternMapping, QuantizeMapping
from ..quantization.quantization import QuantizeConfig
from . import llama_config, llama_model, llama_parameter

ModelConfig = Any
"""A ModelConfig is an object that represents a model architecture. It is required to have
a class method `from_file` with the following signature:

    def from_file(cls, path: Path) -> ModelConfig:
        ...
"""

FuncGetExternMap = Callable[[ModelConfig, QuantizeConfig], ExternMapping]
FuncGetQuantMap = Callable[[ModelConfig, QuantizeConfig], QuantizeMapping]


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

    quantize: Dict[str, FuncGetQuantMap]
        A dictionary that maps the name of a quantization method to quantization mapping.
    """

    name: str
    config: ModelConfig
    model: Callable[[ModelConfig], nn.Module]
    source: Dict[str, FuncGetExternMap]
    quantize: Dict[str, FuncGetQuantMap]


MODELS: Dict[str, Model] = {
    "llama": Model(
        name="llama",
        model=llama_model.LlamaForCasualLM,
        config=llama_config.LlamaConfig,
        source={
            "huggingface-torch": llama_parameter.huggingface,
            "huggingface-safetensor": llama_parameter.huggingface,
        },
        quantize={},
    )
}
