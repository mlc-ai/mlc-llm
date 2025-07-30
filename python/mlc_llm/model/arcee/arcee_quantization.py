"""This file specifies how MLC's Arcee parameters are quantized using group quantization
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

from .arcee_model import ArceeConfig, ArceeForCausalLM


def group_quant(
    model_config: ArceeConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize an Arcee-architecture model using group quantization."""
    model: nn.Module = ArceeForCausalLM(model_config)
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
    model_config: ArceeConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize an Arcee-architecture model using FasterTransformer quantization."""
    model: nn.Module = ArceeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: ArceeConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize an Arcee-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = ArceeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: ArceeConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize an Arcee model without quantization."""
    model: nn.Module = ArceeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def per_tensor_quant(
    model_config: ArceeConfig,
    quantization: PerTensorQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize an Arcee-architecture model using per-tensor quantization."""
    model: nn.Module = ArceeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map