"""This file specifies how MLC's Deepseek-V2 parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    BlockScaleQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
)

from .deepseek_v2_model import DeepseekV2Config, DeepseekV2ForCausalLM


def group_quant(
    model_config: DeepseekV2Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a DeepseekV2-architecture model using group quantization."""
    model: nn.Module = DeepseekV2ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def ft_quant(
    model_config: DeepseekV2Config,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a DeepseekV2 model using FasterTransformer quantization."""
    model: nn.Module = DeepseekV2ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant(
    model_config: DeepseekV2Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a DeepseekV2 model without quantization."""
    model: nn.Module = DeepseekV2ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: DeepseekV2Config,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a DeepseekV2 model using block-scale quantization."""
    model: nn.Module = DeepseekV2ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
