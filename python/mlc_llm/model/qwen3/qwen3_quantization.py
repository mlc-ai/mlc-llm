"""This file specifies how MLC's QWen2 parameters are quantized using group quantization
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

from .qwen3_model import Qwen3Config, Qwen3EmbeddingModel, Qwen3LMHeadModel


def group_quant(
    model_config: Qwen3Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a QWen-architecture model using group quantization."""
    model: nn.Module = Qwen3LMHeadModel(model_config)
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
    model_config: Qwen3Config,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen model using FasterTransformer quantization."""
    model: nn.Module = Qwen3LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: Qwen3Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a QWen model without quantization."""
    model: nn.Module = Qwen3LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def block_scale_quant(
    model_config: Qwen3Config,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3 model using block-scale quantization."""
    model: nn.Module = Qwen3LMHeadModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant_embedding(
    model_config: Qwen3Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3 embedding model without quantization."""
    model: nn.Module = Qwen3EmbeddingModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def group_quant_embedding(
    model_config: Qwen3Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3 embedding model using group quantization."""
    model: nn.Module = Qwen3EmbeddingModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant_embedding(
    model_config: Qwen3Config,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3 embedding model using FasterTransformer quantization."""
    model: nn.Module = Qwen3EmbeddingModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def block_scale_quant_embedding(
    model_config: Qwen3Config,
    quantization: BlockScaleQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3 embedding model using block-scale quantization."""
    model: nn.Module = Qwen3EmbeddingModel(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map
