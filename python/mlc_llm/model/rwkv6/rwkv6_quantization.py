"""This file specifies how MLC's RWKV6 parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from ...loader import QuantizeMapping
from ...quantization import GroupQuantize, NoQuantize
from .rwkv6_model import RWKV6_ForCasualLM, RWKV6Config


def group_quant(
    model_config: RWKV6Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a RWKV4-architecture model using group quantization."""
    model: nn.Module = RWKV6_ForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: RWKV6Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode model without quantization."""
    model: nn.Module = RWKV6_ForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
