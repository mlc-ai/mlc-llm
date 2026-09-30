# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Qwen3.5 Vision-Language model architecture."""

from tvm.relax.frontend.nn import spec as nn_spec

from mlc_llm.model import MODELS
from mlc_llm.model.qwen35.qwen35_vision import Qwen35VisionConfig, Qwen35VisionModel
from mlc_llm.protocol.artifact_manifest import (
    ImageDecodeProcessor,
    build_compiled_program_artifact,
    build_model_package_manifest,
)
from mlc_llm.quantization import QUANTIZATION

# Four layers with full_attention_interval 4: three DeltaNet layers and one attention layer.
SMALL_QWEN35V_CONFIG = {
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248064,
        "rope_theta": 10000000,
        "hidden_act": "silu",
        "full_attention_interval": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "context_window_size": 1024,
        "prefill_chunk_size": 512,
    },
    "vision_config": {
        "hidden_size": 64,
        "num_heads": 2,
        "depth": 2,
        "intermediate_size": 128,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "out_hidden_size": 256,
        "in_channels": 3,
        "num_position_embeddings": 64,
    },
    # A 4 by 4 patch grid merged 2 by 2 gives 4 image tokens.
    "image_size": 64,
    "image_token_id": 248056,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}

SMALL_VISION_CONFIG = SMALL_QWEN35V_CONFIG["vision_config"]
SMALL_VISION_IMAGE_SIZE = SMALL_QWEN35V_CONFIG["image_size"]


def test_qwen35v_model_registered():
    assert "qwen3_5_vision" in MODELS


def test_qwen35v_creation():
    model_info = MODELS["qwen3_5_vision"]
    config = model_info.config.from_dict(SMALL_QWEN35V_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    param_names = [name for name, _ in named_params]
    assert any(n.startswith("visual.") for n in param_names)
    assert any(n.startswith("language_model.") for n in param_names)
    expected_funcs = [
        "embed",
        "image_embed",
        "embed_image",
        "prefill",
        "decode",
        "batch_prefill",
        "batch_decode",
        "batch_verify",
        "create_paged_kv_cache",
        "create_rnn_state",
    ]
    for func_name in expected_funcs:
        assert func_name in mod


def test_qwen35_vision_encoder_creation():
    config = Qwen35VisionConfig.from_dict(SMALL_VISION_CONFIG)
    model = Qwen35VisionModel(config, SMALL_VISION_IMAGE_SIZE)
    image_size = SMALL_VISION_IMAGE_SIZE
    mod_spec = nn_spec.ModuleSpec.from_raw(
        {
            "forward": {
                "pixel_values": nn_spec.Tensor(
                    [1, config.in_channels, image_size, image_size], "float32"
                ),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        },
        model,
    )
    mod, named_params = model.export_tvm(spec=mod_spec)

    param_names = [name for name, _ in named_params]
    for module in ("patch_embed", "pos_embed", "blocks", "merger"):
        assert any(module in n for n in param_names)


def test_qwen35v_artifact_declares_image_input():
    entry = MODELS["qwen3_5_vision"]
    config = entry.config.from_dict(SMALL_QWEN35V_CONFIG)
    image = entry.artifact.tasks(config)["chat.completions"]["inputs"]["image"]
    processor = ImageDecodeProcessor.model_validate(image["processor"])
    assert (processor.resize.mode, processor.resize.height, processor.resize.width) == (
        "stretch",
        64,
        64,
    )
    assert processor.num_embeddings == config.tokens_per_image == 4
    assert image["prompt"] == {
        "prefix_token_ids": [248053],
        "placeholder_token_id": 248056,
        "suffix_token_ids": [248054],
    }


def test_qwen35v_artifact_points_at_exported_functions():
    entry = MODELS["qwen3_5_vision"]
    config = entry.config.from_dict(SMALL_QWEN35V_CONFIG)
    quantization = QUANTIZATION["q4f16_1"]
    model, _ = entry.quantize[quantization.kind](config, quantization)
    mod, named_parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)

    tasks = entry.artifact.tasks(config)
    programs = entry.artifact.programs(config)
    artifact = build_compiled_program_artifact(
        tasks, programs, named_parameters, symbolic_sizes={"vocab_size": config.vocab_size}
    )
    package = build_model_package_manifest(tasks, named_parameters)
    assert artifact.interface_id == package.interface_id
    assert artifact.parameter_schema_id == package.weights.parameter_schema_id
    assert artifact.resources.estimated_device_memory_bytes > 0

    program = programs["generation"]
    assert program["exports"] == {
        "embed_tokens": "embed",
        "prefill_embeds": "prefill",
        "decode_embeds": "decode",
        "create_kv_cache": "create_tir_paged_kv_cache",
        "create_rnn_state": "create_rnn_state",
    }
    assert program["adapters"] == {"image": "embed_image"}
    exported_functions = {global_var.name_hint for global_var in mod.get_global_vars()}
    assert set(program["exports"].values()) - {"create_tir_paged_kv_cache"} <= exported_functions
    assert set(program["adapters"].values()) <= exported_functions

    # The vision tower stays in the model dtype.
    names = [name for name, _ in named_parameters]
    assert not any(name.startswith("visual.") and name.endswith(".q_weight") for name in names)
    assert any(name.startswith("language_model.") and name.endswith(".q_weight") for name in names)


def test_qwen35v_takes_uint32_pixels_on_webgpu():
    import tvm

    entry = MODELS["qwen3_5_vision"]
    config = entry.config.from_dict(SMALL_QWEN35V_CONFIG)
    with tvm.target.Target("webgpu"):
        model = entry.model(config)
        programs = entry.artifact.programs(config)
    assert model.image_dtype == "uint32"
    assert programs["generation"]["adapter_dtypes"] == {"image": "uint32"}
    assert entry.model(config).image_dtype == "uint8"
    assert "adapter_dtypes" not in entry.artifact.programs(config)["generation"]


if __name__ == "__main__":
    test_qwen35v_model_registered()
    test_qwen35v_creation()
    test_qwen35_vision_encoder_creation()
    test_qwen35v_artifact_declares_image_input()
    test_qwen35v_artifact_points_at_exported_functions()
    test_qwen35v_takes_uint32_pixels_on_webgpu()
