# pylint: disable=invalid-name,missing-docstring
import pytest

from mlc_chat.compiler import MODELS, QUANTIZATION
from mlc_chat.compiler.quantization.group_quantization import (
    GroupQuantizeEmbedding,
    GroupQuantizeLinear,
    GroupQuantizeMultiLinear,
)


@pytest.mark.parametrize(
    "model_name, quant_name",
    [
        ("llama2_7b", "q4f16_1"),
        ("llama2_13b", "q4f16_1"),
        ("llama2_70b", "q4f16_1"),
    ],
)
def test_llama2_group_quantization(model_name: str, quant_name: str):
    model_info = MODELS["llama"]
    config = model_info.config.from_predefined(model_name)
    model, quant_map = model_info.quantize["group-quant"](config, QUANTIZATION[quant_name])
    assert "model.embed_tokens.weight" in quant_map.param_map
    assert isinstance(model.model.embed_tokens, GroupQuantizeEmbedding)
    assert "lm_head.weight" in quant_map.param_map
    assert isinstance(model.lm_head, GroupQuantizeLinear)
    for i in range(config.num_hidden_layers):
        assert f"model.layers.{i}.self_attn.qkv_proj.weight" in quant_map.param_map
        assert isinstance(model.model.layers[i].self_attn.qkv_proj, GroupQuantizeMultiLinear)
        assert f"model.layers.{i}.self_attn.o_proj.weight" in quant_map.param_map
        assert isinstance(model.model.layers[i].self_attn.o_proj, GroupQuantizeLinear)
        assert f"model.layers.{i}.mlp.gate_up_proj.weight" in quant_map.param_map
        assert isinstance(model.model.layers[i].mlp.gate_up_proj, GroupQuantizeMultiLinear)
        assert f"model.layers.{i}.mlp.down_proj.weight" in quant_map.param_map
        assert isinstance(model.model.layers[i].mlp.down_proj, GroupQuantizeLinear)


if __name__ == "__main__":
    test_llama2_group_quantization("llama2_7b", "q4f16_1")
    test_llama2_group_quantization("llama2_13b", "q4f16_1")
    test_llama2_group_quantization("llama2_70b", "q4f16_1")
