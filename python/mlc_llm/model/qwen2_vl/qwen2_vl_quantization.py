"""This file specifies how MLC's Qwen2 VL parameters are quantized using group quantization
or other formats."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import FTQuantize, GroupQuantize, NoQuantize

from .qwen2_vl_config import QWen2VLConfig
from .qwen2_vl_model import QWen2VLForCausalLM


def group_quant(
    model_config: QWen2VLConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen2 VL model using group quantization."""
    model: nn.Module = QWen2VLForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards

    # Vision model quantization patterns
    vision_patterns = [
        # Vision transformer attention patterns
        "vision_model.layers.*.attention.q_proj.weight",
        "vision_model.layers.*.attention.k_proj.weight",
        "vision_model.layers.*.attention.v_proj.weight",
        "vision_model.layers.*.attention.o_proj.weight",
        # Vision transformer MLP patterns
        "vision_model.layers.*.mlp.fc1.weight",
        "vision_model.layers.*.mlp.fc2.weight",
        # Vision embeddings
        "vision_model.embeddings.patch_embed.weight",
        # Vision projection
        "vision_projection.linear_1.weight",
        "vision_projection.linear_2.weight",
    ]

    # Add vision patterns to quantization
    for pattern in vision_patterns:
        quantization.add_pattern(pattern)

    # Language model patterns (from qwen2_quantization.py)
    language_patterns = [
        # Attention patterns
        "language_model.layers.*.self_attn.c_attn.weight",
        "language_model.layers.*.self_attn.c_proj.weight",
        # MLP patterns
        "language_model.layers.*.mlp.gate_up_proj.weight",
        "language_model.layers.*.mlp.down_proj.weight",
    ]

    # Add language patterns to quantization
    for pattern in language_patterns:
        quantization.add_pattern(pattern)

    # Quantize the model
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: QWen2VLConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen2 VL model using FasterTransformer quantization."""
    model: nn.Module = QWen2VLForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})

    # Add vision and language patterns similar to group_quant
    vision_patterns = [
        "vision_model.layers.*.attention.*.weight",
        "vision_model.layers.*.mlp.*.weight",
        "vision_model.embeddings.patch_embed.weight",
        "vision_projection.*.weight",
    ]
    
    language_patterns = [
        "language_model.layers.*.self_attn.*.weight",
        "language_model.layers.*.mlp.*.weight",
    ]

    # Add patterns to quantization
    for pattern in vision_patterns + language_patterns:
        quantization.add_pattern(pattern)

    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: QWen2VLConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Load a Qwen2 VL model without quantization."""
    model: nn.Module = QWen2VLForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
