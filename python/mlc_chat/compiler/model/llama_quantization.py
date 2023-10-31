"""Quantization specs for Llama."""
from typing import Tuple

from tvm.relax.frontend import nn

from ..parameter import QuantizeMapping
from ..quantization import GroupQuantize
from .llama_config import LlamaConfig
from .llama_model import LlamaForCasualLM


def group_quant(
    model_config: LlamaConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model using group quantization."""
    model: nn.Module = LlamaForCasualLM(model_config)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map
