"""This file specifies how MLC's GPTJ parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_chat.loader import QuantizeMapping
from mlc_chat.quantization import GroupQuantize, NoQuantize

from .gptj_model import GPTJConfig, GPTJForCausalLM


def group_quant(
    model_config: GPTJConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a BaichuanLM-architecture model using group quantization."""
    model: nn.Module = GPTJForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: GPTJConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a BaichuanLM model without quantization."""
    model: nn.Module = GPTJForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map