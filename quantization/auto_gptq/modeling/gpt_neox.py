from ._base import *


class GPTNeoXGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GPTNeoXLayer"
    layers_block_name = "gpt_neox.layers"
    outside_layer_modules = ["gpt_neox.embed_in", "gpt_neox.final_layer_norm"]
    inside_layer_modules = [
        ["attention.query_key_value"],
        ["attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"]
    ]
    lm_head_name = "embed_out"


__all__ = ["GPTNeoXGPTQForCausalLM"]
