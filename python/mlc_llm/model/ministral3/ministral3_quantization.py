"""This file specifies how MLC's Ministral3 parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    AWQQuantize,
    BlockScaleQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
)

from .ministral3_model import Ministral3Config, Mistral3ForConditionalGeneration


def group_quant(
    model_config: Ministral3Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Ministral3-architecture model using group quantization."""
    model: nn.Module = Mistral3ForConditionalGeneration(model_config)
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
    model_config: Ministral3Config,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Ministral3-architecture model using FasterTransformer quantization."""
    model: nn.Module = Mistral3ForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: Ministral3Config,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Ministral3-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = Mistral3ForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: Ministral3Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Ministral3 model without quantization."""
    model: nn.Module = Mistral3ForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: Ministral3Config,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Ministral3 model using block-scale quantization."""
    model: nn.Module = Mistral3ForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
