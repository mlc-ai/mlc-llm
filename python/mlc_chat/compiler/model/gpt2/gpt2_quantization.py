"""This file specifies how MLC's GPT-2 parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from ...loader import QuantizeMapping
from ...quantization import AWQQuantize, GroupQuantize, NoQuantize
from .gpt2_model import GPT2Config, GPT2LMHeadModel


def group_quant(
    model_config: GPT2Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPT-2-architecture model using group quantization."""
    model: nn.Module = GPT2LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: GPT2Config,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPT-2-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = GPT2LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: GPT2Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPT-2 model without quantization."""
    model: nn.Module = GPT2LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
