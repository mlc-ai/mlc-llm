"""This file specifies how MLC's Sarvam-MoE parameters are quantized using group quantization
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

from .sarvam_moe_model import SarvamMoeConfig, SarvamMoeForCausalLM


def group_quant(
    model_config: SarvamMoeConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a SarvamMoE-architecture model using group quantization."""
    model: nn.Module = SarvamMoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def ft_quant(
    model_config: SarvamMoeConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a SarvamMoE model using FasterTransformer quantization."""
    model: nn.Module = SarvamMoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant(
    model_config: SarvamMoeConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a SarvamMoE model without quantization."""
    model: nn.Module = SarvamMoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: SarvamMoeConfig,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a SarvamMoE model using block-scale quantization."""
    model: nn.Module = SarvamMoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
