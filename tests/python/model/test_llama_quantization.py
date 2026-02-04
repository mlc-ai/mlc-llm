# pylint: disable=invalid-name,missing-docstring
import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.quantization.group_quantization import (
    GroupQuantizeEmbedding,
    GroupQuantizeLinear,
)


@pytest.mark.parametrize(
    "model_name",
    ["llama2_7b", "llama2_13b", "llama2_70b"],
)
@pytest.mark.parametrize(
    "quant_name",
    ["q3f16_1", "q4f16_1", "q4f32_1"],
)
def test_llama2_group_quantization(model_name: str, quant_name: str):
    model_info = MODELS["llama"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model, quant_map = model_info.quantize["group-quant"](config, QUANTIZATION[quant_name])
    assert "model.embed_tokens.weight" in quant_map.param_map
    assert isinstance(
        model.model.embed_tokens,  # type: ignore[attr-defined]
        GroupQuantizeEmbedding,
    )
    assert "lm_head.weight" in quant_map.param_map
    assert isinstance(model.lm_head, GroupQuantizeLinear)  # type: ignore[attr-defined]
    for i in range(config.num_hidden_layers):
        assert f"model.layers.{i}.self_attn.qkv_proj.weight" in quant_map.param_map
        assert isinstance(
            model.model.layers[i].self_attn.qkv_proj,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )
        assert f"model.layers.{i}.self_attn.o_proj.weight" in quant_map.param_map
        assert isinstance(
            model.model.layers[i].self_attn.o_proj,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )
        assert f"model.layers.{i}.mlp.gate_up_proj.weight" in quant_map.param_map
        assert isinstance(
            model.model.layers[i].mlp.gate_up_proj,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )
        assert f"model.layers.{i}.mlp.down_proj.weight" in quant_map.param_map
        assert isinstance(
            model.model.layers[i].mlp.down_proj,  # type: ignore[attr-defined]
            GroupQuantizeLinear,
        )


@pytest.mark.parametrize(
    "model_name",
    ["llama2_7b", "llama2_13b", "llama2_70b"],
)
@pytest.mark.parametrize(
    "quant_name",
    ["q0f16", "q0f32"],
)
def test_llama2_no_quantization(model_name: str, quant_name: str):
    model_info = MODELS["llama"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    _, quant_map = model_info.quantize["no-quant"](config, QUANTIZATION[quant_name])
    assert len(quant_map.param_map) == 0
    assert len(quant_map.map_func) == 0


if __name__ == "__main__":
    test_llama2_group_quantization("llama2_7b", "q4f16_1")
    test_llama2_group_quantization("llama2_13b", "q4f16_1")
    test_llama2_group_quantization("llama2_70b", "q4f16_1")
