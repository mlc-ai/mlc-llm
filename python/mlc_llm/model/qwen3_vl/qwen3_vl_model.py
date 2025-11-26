"""
Minimal model for Qwen3-VL.
"""
from tvm.relax.frontend import nn
from .qwen3_vl_config import Qwen3VLConfig

class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
