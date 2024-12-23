"""This file specifies how MLC's BERT parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import NoQuantize

from .jina_model import JinaConfig, JinaModel


def no_quant(
    model_config: JinaConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a BERT model without quantization."""
    model: nn.Module = JinaModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
