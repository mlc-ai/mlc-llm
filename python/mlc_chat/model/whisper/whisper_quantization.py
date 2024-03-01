"""This file specifies how MLC's Whisper parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_chat.loader import QuantizeMapping
from mlc_chat.quantization import GroupQuantize, NoQuantize

from .whisper_model import WhisperConfig, WhisperForConditionalGeneration


def group_quant(
    model_config: WhisperConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTNeoX-architecture model using group quantization."""
    model: nn.Module = WhisperForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    model.dtype = quantization.model_dtype
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: WhisperConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTNeoX model without quantization."""
    model: nn.Module = WhisperForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    model.dtype = quantization.model_dtype
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
