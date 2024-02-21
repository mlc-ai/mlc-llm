"""This file specifies how MLC's Mistral parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_chat.loader import QuantizeMapping
from mlc_chat.quantization import AWQQuantize, FTQuantize, GroupQuantize, NoQuantize

from .mixtral_model import MixtralConfig, MixtralForCasualLM


def group_quant(
    model_config: MixtralConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Mixtral-architecture model using group quantization."""
    model: nn.Module = MixtralForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: MixtralConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Mixtral-architecture model using FasterTransformer quantization."""
    model: nn.Module = MixtralForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: MixtralConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Mixtral-architecture model using Activation-aware Weight Quantization(AWQ)."""
    raise NotImplementedError("AWQ is not implemented for Mixtral models.")


def no_quant(
    model_config: MixtralConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Mixtral model without quantization."""
    model: nn.Module = MixtralForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
