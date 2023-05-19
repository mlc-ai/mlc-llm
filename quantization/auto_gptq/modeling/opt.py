from ._base import *


class OPTGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "OPTDecoderLayer"
    layers_block_name = "model.decoder.layers"
    outside_layer_modules = [
        "model.decoder.embed_tokens", "model.decoder.embed_positions", "model.decoder.project_out",
        "model.decoder.project_in", "model.decoder.final_layer_norm"
    ]
    inside_layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"]
    ]


__all__ = ["OPTGPTQForCausalLM"]
