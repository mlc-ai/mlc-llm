"""This file specifies how MLC's GLM-4.5-Air MoE parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    BlockScaleQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
    PerTensorQuantize,
)

from .glm4_moe_model import Glm4MoeConfig, Glm4MoeForCausalLM


def group_quant(
    model_config: Glm4MoeConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GLM-4.5-Air MoE model using group quantization."""
    model: nn.Module = Glm4MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def ft_quant(
    model_config: Glm4MoeConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GLM-4.5-Air MoE model using FasterTransformer quantization."""
    model: nn.Module = Glm4MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant(
    model_config: Glm4MoeConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Load a GLM-4.5-Air MoE model without quantization."""
    model: nn.Module = Glm4MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: Glm4MoeConfig,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GLM-4.5-Air MoE model using block-scale quantization."""
    model: nn.Module = Glm4MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def per_tensor_quant(
    model_config: Glm4MoeConfig,
    quantization: PerTensorQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GLM-4.5-Air MoE model using per-tensor quantization."""
    model: nn.Module = Glm4MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
