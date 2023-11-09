"""This file specifies how MLC's Llama parameters are quantized using group quantization
or other formats."""
from typing import Tuple, Union

from tvm.relax.frontend import nn

from ..loader import QuantizeMapping
from ..quantization import AWQQuantize, GroupQuantize, NoQuantize
from .llama_model import LlamaConfig, LlamaForCasualLM
from .mistral_model import MistralConfig, MistralForCasualLM


def group_quant(
    model_config: Union[LlamaConfig, MistralConfig],
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using group quantization."""
    if isinstance(model_config, MistralConfig):
        model: nn.Module = MistralForCasualLM(model_config)
    else:
        model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: Union[LlamaConfig, MistralConfig],
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using Activation-aware Weight Quantization(AWQ)."""
    if isinstance(model_config, MistralConfig):
        model: nn.Module = MistralForCasualLM(model_config)
    else:
        model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: LlamaConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model without quantization."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
