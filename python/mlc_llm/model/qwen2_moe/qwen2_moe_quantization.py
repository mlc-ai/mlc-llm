"""This file specifies how MLC's QWen2 parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import FTQuantize, GroupQuantize, NoQuantize

from .qwen2_moe_model import Qwen2MoeConfig, Qwen2MoeForCausalLM


def group_quant(
    model_config: Qwen2MoeConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen2MoE-architecture model using group quantization."""
    model: nn.Module = Qwen2MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def ft_quant(
    model_config: Qwen2MoeConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen2MoE model using FasterTransformer quantization."""
    model: nn.Module = Qwen2MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant(
    model_config: Qwen2MoeConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen2MoE model without quantization."""
    model: nn.Module = Qwen2MoeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
