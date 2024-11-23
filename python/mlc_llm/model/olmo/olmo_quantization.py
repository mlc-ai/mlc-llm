"""This file specifies how MLC's OLMo parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import FTQuantize, GroupQuantize, NoQuantize

from .olmo_model import OlmoConfig, OlmoForCausalLM


def group_quant(
    model_config: OlmoConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo-architecture model using group quantization."""
    model: nn.Module = OlmoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: OlmoConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo model using FasterTransformer quantization."""
    model: nn.Module = OlmoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: OlmoConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo model without quantization."""
    model: nn.Module = OlmoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
