"""This file specifies how MLC's GPTNeoX parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import FTQuantize, GroupQuantize, NoQuantize

from .gpt_neox_model import GPTNeoXConfig, GPTNeoXForCausalLM


def group_quant(
    model_config: GPTNeoXConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTNeoX-architecture model using group quantization."""
    model: nn.Module = GPTNeoXForCausalLM(model_config)
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
    model_config: GPTNeoXConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTNeoX-architecture model using FasterTransformer quantization."""
    model: nn.Module = GPTNeoXForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: GPTNeoXConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a GPTNeoX model without quantization."""
    model: nn.Module = GPTNeoXForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
