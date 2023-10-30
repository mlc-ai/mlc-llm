"""
Quantization specs for Llama2 architecture.
TODO: add docstring
"""
from typing import Tuple

from tvm.relax.frontend import nn

from ..parameter import QuantizeMapping
from ..quantization import GroupQuantizeConfig
from .llama_model import LlamaForCasualLM


def llama_group_quantization(
    model: LlamaForCasualLM, quant_config: GroupQuantizeConfig
) -> Tuple[nn.Module, QuantizeMapping]:
    quant_map = QuantizeMapping({}, {})
    for i in range(len(model.model.layers)):
        model.model.layers[i] = quant_config.apply(
            model.model.layers[i], quant_map, f"model.layers.{i}"
        )
    model.model.embed_tokens = quant_config.apply(
        model.model.embed_tokens, quant_map, "model.embed_tokens"
    )
    model.lm_head = quant_config.apply(model.lm_head, quant_map, "lm_head")
    return model, quant_map
