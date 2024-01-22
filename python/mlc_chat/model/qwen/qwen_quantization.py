"""This file specifies how MLC's QWen parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_chat.loader import QuantizeMapping
from mlc_chat.quantization import GroupQuantize, NoQuantize

from .qwen_model import QWenConfig, QWenLMHeadModel


def group_quant(
    model_config: QWenConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a QWen-architecture model using group quantization."""
    model: nn.Module = QWenLMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: QWenConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a QWen model without quantization."""
    model: nn.Module = QWenLMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
