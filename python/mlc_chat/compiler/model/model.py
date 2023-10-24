"""A centralized registry of all existing model architures and their configurations."""
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tvm.relax.frontend import nn

from ..parameter import ExternMapping, QuantizeMapping
from . import llama_config, llama_model, llama_parameter

ModelConfig = Any
QuantizeConfig = Any

LoaderType = Callable[[ModelConfig, QuantizeConfig], ExternMapping]
QuantizerType = Callable[[ModelConfig, QuantizeConfig], QuantizeMapping]


@dataclasses.dataclass
class Model:
    """All about a model architecture: its configuration, its parameter loader and quantization."""

    name: str
    model: Callable[[ModelConfig], nn.Module]
    config: Callable[[Path], ModelConfig]
    source_loader_huggingface: Optional[LoaderType] = None
    source_loader_awq: Optional[LoaderType] = None
    quantizer_group_quant: Optional[QuantizerType] = None


MODELS: Dict[str, Model] = {
    "llama": Model(
        name="llama",
        model=llama_model.LlamaForCasualLM,
        config=llama_config.LlamaConfig.from_file,
        source_loader_huggingface=llama_parameter.huggingface,
        source_loader_awq=None,
        quantizer_group_quant=None,
    )
}
