"""
Minimal quantization for Qwen3-VL.
"""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    BlockScaleQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
)

from .qwen3_vl_config import Qwen3VLConfig
from .qwen3_vl_model import Qwen3VLForConditionalGeneration


def group_quant(
    model_config: Qwen3VLConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3-VL model using group quantization."""
    model: nn.Module = Qwen3VLForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    
    quantization.tensor_parallel_shards = model_config.text_config.tensor_parallel_shards
    
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: Qwen3VLConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3-VL model using FasterTransformer quantization."""
    model: nn.Module = Qwen3VLForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: Qwen3VLConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3-VL model without quantization."""
    model = Qwen3VLForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: Qwen3VLConfig,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3-VL model using block-scale quantization."""
    model: nn.Module = Qwen3VLForConditionalGeneration(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
