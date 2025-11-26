"""
Minimal quantization for Qwen3-VL.
"""
from typing import Any, Dict, Tuple
from tvm.relax.frontend import nn
from mlc_llm.loader import QuantizeMapping

def no_quant(model_config, quantization) -> Tuple[nn.Module, QuantizeMapping]:
    return None, None
