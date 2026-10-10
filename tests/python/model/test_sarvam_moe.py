import numpy as np

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.quantization import QUANTIZATION


def _tiny_config(**overrides):
    config = dict(MODEL_PRESETS["sarvam_moe_tiny"])
    config.update(overrides)
    return MODELS["sarvam_moe"].config.from_dict(config)


def test_sarvam_moe_config_defaults_from_hf_fields():
    config_dict = dict(MODEL_PRESETS["sarvam_moe_tiny"])
    config_dict.pop("prefill_chunk_size")
    config_dict["use_qkv_bias"] = True
    config = MODELS["sarvam_moe"].config.from_dict(config_dict)

    assert config.context_window_size == config_dict["max_position_embeddings"]
    assert config.prefill_chunk_size == config.context_window_size
    assert config.attention_bias is True
    assert config.kwargs["architectures"] == ["SarvamMoEForCausalLM"]


def test_sarvam_moe_model_creation_exports_dense_and_moe_parameters():
    model_info = MODELS["sarvam_moe"]
    model = model_info.model(_tiny_config())
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    params = {name: (param.shape, param.dtype) for name, param in named_params}

    assert params["model.layers.0.mlp.gate_up_proj.weight"] == ([256, 64], "float32")
    assert params["model.layers.0.mlp.down_proj.weight"] == ([64, 128], "float32")
    assert params["model.layers.1.mlp.gate.weight"] == ([4, 64], "float32")
    assert params["model.layers.1.mlp.moe_gate_up_proj.weight"] == ([4, 128, 64], "float32")
    assert params["model.layers.1.mlp.moe_down_proj.weight"] == ([4, 64, 64], "float32")
    assert params["model.layers.1.mlp.expert_bias"] == ([4], "float32")
    assert params["model.layers.1.mlp.shared_expert.gate_up_proj.weight"] == (
        [128, 64],
        "float32",
    )
    assert params["lm_head.weight"] == ([128, 64], "float32")


def test_sarvam_moe_huggingface_mapping_transforms():
    model_info = MODELS["sarvam_moe"]
    mapping = model_info.source["huggingface-safetensor"](_tiny_config(), None)

    assert mapping.param_map["model.embed_tokens.weight"] == ["model.word_embeddings.weight"]
    assert mapping.param_map["model.layers.1.mlp.expert_bias"] == [
        "model.layers.1.mlp.gate.expert_bias"
    ]
    assert mapping.param_map["model.layers.1.mlp.shared_expert.gate_up_proj.weight"] == [
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
    ]
    assert mapping.param_map["model.layers.0.input_layernorm.weight"] == [
        "model.layers.0.input_layernorm.weight"
    ]

    qkv = np.arange(128 * 64, dtype="float32").reshape(128, 64)
    np.testing.assert_array_equal(
        mapping.map_func["model.layers.0.self_attn.c_attn.weight"](qkv),
        qkv,
    )

    gate = np.ones((64, 64), dtype="float32")
    up = np.full((64, 64), 2, dtype="float32")
    np.testing.assert_array_equal(
        mapping.map_func["model.layers.0.mlp.gate_up_proj.weight"](gate, up),
        np.concatenate([gate, up], axis=0),
    )

    expert_inputs = []
    for expert_id in range(4):
        expert_inputs.extend(
            [
                np.full((64, 64), expert_id * 2, dtype="float32"),
                np.full((64, 64), expert_id * 2 + 1, dtype="float32"),
            ]
        )
    packed = mapping.map_func["model.layers.1.mlp.moe_gate_up_proj.weight"](*expert_inputs)
    assert packed.shape == (4, 128, 64)
    np.testing.assert_array_equal(packed[0], np.concatenate(expert_inputs[:2], axis=0))

    down_inputs = [np.full((64, 64), expert_id, dtype="float32") for expert_id in range(4)]
    down = mapping.map_func["model.layers.1.mlp.moe_down_proj.weight"](*down_inputs)
    assert down.shape == (4, 64, 64)
    np.testing.assert_array_equal(down[3], down_inputs[3])


def test_sarvam_moe_q4f16_1_quantizes_moe_experts_in_nk_layout():
    model_info = MODELS["sarvam_moe"]
    quantization = QUANTIZATION["q4f16_1"]
    model, quant_map = model_info.quantize[quantization.kind](_tiny_config(), quantization)
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    params = {name: (param.shape, param.dtype) for name, param in named_params}

    assert quant_map.param_map["model.layers.1.mlp.moe_gate_up_proj.weight"] == [
        "model.layers.1.mlp.moe_gate_up_proj.q_weight",
        "model.layers.1.mlp.moe_gate_up_proj.q_scale",
    ]
    assert params["model.layers.1.mlp.moe_gate_up_proj.q_weight"] == ([4, 128, 8], "uint32")
    assert params["model.layers.1.mlp.moe_gate_up_proj.q_scale"] == ([4, 128, 2], "float16")
    assert params["model.layers.1.mlp.moe_down_proj.q_weight"] == ([4, 64, 8], "uint32")
    assert params["model.layers.1.mlp.moe_down_proj.q_scale"] == ([4, 64, 2], "float16")
