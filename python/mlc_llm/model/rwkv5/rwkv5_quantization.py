"""This file specifies how MLC's RWKV5 parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from ...loader import QuantizeMapping
from ...quantization import FTQuantize, GroupQuantize, NoQuantize
from .rwkv5_model import RWKV5_ForCasualLM, RWKV5Config


def group_quant(
    model_config: RWKV5Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a RWKV4-architecture model using group quantization."""
    model: nn.Module = RWKV5_ForCasualLM(model_config)
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
    model_config: RWKV5Config,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a InternLM model using FasterTransformer quantization."""
    model: nn.Module = RWKV5_ForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: RWKV5Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode model without quantization."""
    model: nn.Module = RWKV5_ForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
