"""This file specifies how MLC's OLMo parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    AWQQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
    PerTensorQuantize,
)

from .olmo_model import OLMoConfig, OLMoForCausalLM


def group_quant(
    model_config: OLMoConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo-architecture model using group quantization."""
    model: nn.Module = OLMoForCausalLM(model_config)
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
    model_config: OLMoConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo-architecture model using FasterTransformer quantization."""
    model: nn.Module = OLMoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: OLMoConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = OLMoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: OLMoConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo model without quantization."""
    model: nn.Module = OLMoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def per_tensor_quant(
    model_config: OLMoConfig,
    quantization: PerTensorQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a OLMo-architecture model using per-tensor quantization."""
    model: nn.Module = OLMoForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
        tensor_parallel_shards=model_config.tensor_parallel_shards,
    )
    return model, quant_map
