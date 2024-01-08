"""A centralized registry of all existing model architures and their configurations."""
import dataclasses
from typing import Any, Callable, Dict, Tuple

from tvm.relax.frontend import nn

from mlc_chat.loader import ExternMapping, QuantizeMapping
from mlc_chat.quantization.quantization import Quantization

from .gpt2 import gpt2_loader, gpt2_model, gpt2_quantization
from .gpt_bigcode import gpt_bigcode_loader, gpt_bigcode_model, gpt_bigcode_quantization
from .gpt_neox import gpt_neox_loader, gpt_neox_model, gpt_neox_quantization
from .llama import llama_loader, llama_model, llama_quantization
from .mistral import mistral_loader, mistral_model, mistral_quantization
from .mixtral import mixtral_loader, mixtral_model, mixtral_quantization
from .phi import phi_loader, phi_model, phi_quantization

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
        config=llama_model.LlamaConfig,
        source={
            "huggingface-torch": llama_loader.huggingface,
            "huggingface-safetensor": llama_loader.huggingface,
            "awq": llama_loader.awq,
        },
        quantize={
            "no-quant": llama_quantization.no_quant,
            "group-quant": llama_quantization.group_quant,
            "ft-quant": llama_quantization.ft_quant,
            "awq": llama_quantization.awq_quant,
        },
    ),
    "mistral": Model(
        name="mistral",
        model=mistral_model.MistralForCasualLM,
        config=mistral_model.MistralConfig,
        source={
            "huggingface-torch": mistral_loader.huggingface,
            "huggingface-safetensor": mistral_loader.huggingface,
            "awq": mistral_loader.awq,
        },
        quantize={
            "group-quant": mistral_quantization.group_quant,
        },
    ),
    "gpt2": Model(
        name="gpt2",
        model=gpt2_model.GPT2LMHeadModel,
        config=gpt2_model.GPT2Config,
        source={
            "huggingface-torch": gpt2_loader.huggingface,
            "huggingface-safetensor": gpt2_loader.huggingface,
        },
        quantize={
            "no-quant": gpt2_quantization.no_quant,
            "group-quant": gpt2_quantization.group_quant,
        },
    ),
    "mixtral": Model(
        name="mixtral",
        model=mixtral_model.MixtralForCasualLM,
        config=mixtral_model.MixtralConfig,
        source={
            "huggingface-torch": mixtral_loader.huggingface,
            "huggingface-safetensor": mixtral_loader.huggingface,
        },
        quantize={
            "no-quant": mixtral_quantization.no_quant,
            "group-quant": mixtral_quantization.group_quant,
        },
    ),
    "gpt_neox": Model(
        name="gpt_neox",
        model=gpt_neox_model.GPTNeoXForCausalLM,
        config=gpt_neox_model.GPTNeoXConfig,
        source={
            "huggingface-torch": gpt_neox_loader.huggingface,
            "huggingface-safetensor": gpt_neox_loader.huggingface,
        },
        quantize={
            "no-quant": gpt_neox_quantization.no_quant,
            "group-quant": gpt_neox_quantization.group_quant,
        },
    ),
    "gpt_bigcode": Model(
        name="gpt_bigcode",
        model=gpt_bigcode_model.GPTBigCodeForCausalLM,
        config=gpt_bigcode_model.GPTBigCodeConfig,
        source={
            "huggingface-torch": gpt_bigcode_loader.huggingface,
            "huggingface-safetensor": gpt_bigcode_loader.huggingface,
        },
        quantize={
            "no-quant": gpt_bigcode_quantization.no_quant,
            "group-quant": gpt_bigcode_quantization.group_quant,
        },
    ),
    "phi-msft": Model(
        name="phi-msft",
        model=phi_model.PhiForCausalLM,
        config=phi_model.PhiConfig,
        source={
            "huggingface-torch": phi_loader.huggingface,
            "huggingface-safetensor": phi_loader.huggingface,
        },
        quantize={
            "no-quant": phi_quantization.no_quant,
            "group-quant": phi_quantization.group_quant,
        },
    ),
}
