"""This file specifies how MLC's Llama parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from ..loader import QuantizeMapping
from ..quantization import AWQQuantize, GroupQuantize
from .llama_model import LlamaConfig, LlamaForCasualLM


def group_quant(
    model_config: LlamaConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model using group quantization."""
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
    model_config: LlamaConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map
