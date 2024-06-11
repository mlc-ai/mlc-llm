"""This file specifies how MLC's QWen parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import FTQuantize, GroupQuantize, NoQuantize

from .qwen_model import QWenConfig, QWenLMHeadModel


def group_quant(
    model_config: QWenConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a QWen-architecture model using group quantization."""
    model: nn.Module = QWenLMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: QWenConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen model using FasterTransformer quantization."""
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
