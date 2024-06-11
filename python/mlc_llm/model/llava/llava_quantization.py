"""This file specifies how MLC's Llava parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from ...loader import QuantizeMapping
from ...quantization import AWQQuantize, GroupQuantize, NoQuantize
from .llava_model import LlavaConfig, LlavaForCasualLM


def group_quant(
    model_config: LlavaConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llava model using group quantization."""
    model: nn.Module = LlavaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: LlavaConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llava model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = LlavaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: LlavaConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llava model without quantization."""
    model: nn.Module = LlavaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
