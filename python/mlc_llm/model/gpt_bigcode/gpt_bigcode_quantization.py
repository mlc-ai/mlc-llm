"""This file specifies how MLC's GPTBigCode parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import AWQQuantize, FTQuantize, GroupQuantize, NoQuantize

from .gpt_bigcode_model import GPTBigCodeConfig, GPTBigCodeForCausalLM


def group_quant(
    model_config: GPTBigCodeConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode-architecture model using group quantization."""
    model: nn.Module = GPTBigCodeForCausalLM(model_config)
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
    model_config: GPTBigCodeConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode-architecture model using FasterTransformer quantization."""
    model: nn.Module = GPTBigCodeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: GPTBigCodeConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = GPTBigCodeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: GPTBigCodeConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTBigCode model without quantization."""
    model: nn.Module = GPTBigCodeForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
