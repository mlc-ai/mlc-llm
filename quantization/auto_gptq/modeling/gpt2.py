from ._base import *


class GPT2GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GPT2Block"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"]
    ]


__all__ = ["GPT2GPTQForCausalLM"]
