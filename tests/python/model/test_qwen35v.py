# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Qwen3.5 Vision-Language model architecture."""

import pytest

from tvm.relax.frontend.nn import spec as nn_spec

from mlc_llm.model import MODELS
from mlc_llm.model.qwen35.qwen35_vision import Qwen35VisionConfig, Qwen35VisionModel


# Minimal config with small dimensions for fast testing.
# Must exercise both DeltaNet (linear) and full attention layers.
# With full_attention_interval=4 and num_hidden_layers=4:
#   layers 0,1,2 = DeltaNet, layer 3 = full attention.
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
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 256,
        "in_channels": 3,
        "num_position_embeddings": 64,
    },
    # image_size=64, patch_size=16 -> 4x4 grid, spatial_merge_size=2 -> 2x2 -> 4 tokens
    "image_size": 64,
    "image_token_id": 248056,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}

# Standalone vision config matching the vision_config above
SMALL_VISION_CONFIG = SMALL_QWEN35V_CONFIG["vision_config"]
SMALL_VISION_IMAGE_SIZE = SMALL_QWEN35V_CONFIG["image_size"]


def test_qwen35v_model_registered():
    """Verify Qwen3.5 Vision model is in the registry."""
    assert "qwen3_5_vision" in MODELS, "qwen3_5_vision should be registered in MODELS"


def test_qwen35v_creation():
    """Test Qwen3.5V model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include visual (vision encoder) and model (language model) components
    - All expected functions are exported, including create_rnn_state (hybrid architecture)
    """
    model_info = MODELS["qwen3_5_vision"]
    config = model_info.config.from_dict(SMALL_QWEN35V_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Verify VLM composition: params from both components
    param_names = [name for name, _ in named_params]
    has_visual = any(n.startswith("visual.") for n in param_names)
    has_language = any(n.startswith("language_model.") for n in param_names)
    assert has_visual, "Should have visual.* parameters (vision encoder)"
    assert has_language, "Should have language_model.* parameters (text model)"

    # Verify all expected functions are exported
    # create_rnn_state is unique to Qwen3.5's hybrid DeltaNet architecture
    expected_funcs = [
        "embed",
        "image_embed",
        "batch_prefill",
        "batch_decode",
        "batch_verify",
        "create_paged_kv_cache",
        "create_rnn_state",
    ]
    for func_name in expected_funcs:
        assert func_name in mod, f"Module should contain '{func_name}' function"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_qwen35v_config_validation():
    """Test Qwen3.5V configuration has required fields."""
    model_info = MODELS["qwen3_5_vision"]
    config = model_info.config.from_dict(SMALL_QWEN35V_CONFIG)

    # Text config fields
    assert hasattr(config.text_config, "hidden_size") and config.text_config.hidden_size > 0
    assert (
        hasattr(config.text_config, "num_hidden_layers")
        and config.text_config.num_hidden_layers > 0
    )
    assert (
        hasattr(config.text_config, "full_attention_interval")
        and config.text_config.full_attention_interval > 0
    )

    # Vision config fields
    assert hasattr(config.vision_config, "hidden_size") and config.vision_config.hidden_size > 0
    assert hasattr(config.vision_config, "depth") and config.vision_config.depth > 0
    assert hasattr(config.vision_config, "patch_size") and config.vision_config.patch_size > 0
    assert (
        hasattr(config.vision_config, "spatial_merge_size")
        and config.vision_config.spatial_merge_size > 0
    )
    assert (
        hasattr(config.vision_config, "out_hidden_size")
        and config.vision_config.out_hidden_size > 0
    )

    # Computed property: tokens_per_image
    # image_size=64, patch_size=16 -> 4x4 grid, merge_size=2 -> 2x2 merged -> 4 tokens
    assert config.tokens_per_image == 4

    print(
        f"Qwen3.5V Config: text_hidden={config.text_config.hidden_size}, "
        f"vision_hidden={config.vision_config.hidden_size}, "
        f"text_layers={config.text_config.num_hidden_layers}, "
        f"vision_depth={config.vision_config.depth}, "
        f"tokens_per_image={config.tokens_per_image}"
    )


def test_qwen35_vision_encoder_creation():
    """Test Qwen3.5 vision encoder standalone creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Vision model can be created
    - Model exports to TVM IR successfully
    - Named parameters include patch_embed, pos_embed, blocks, and merger components
    """
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

    assert mod is not None
    assert len(named_params) > 0

    param_names = [name for name, _ in named_params]

    has_patch_embed = any("patch_embed" in n for n in param_names)
    has_pos_embed = any("pos_embed" in n for n in param_names)
    has_blocks = any("blocks" in n for n in param_names)
    has_merger = any("merger" in n for n in param_names)

    assert has_patch_embed, "Should have patch_embed parameters"
    assert has_pos_embed, "Should have pos_embed parameters"
    assert has_blocks, "Should have blocks (encoder layer) parameters"
    assert has_merger, "Should have merger parameters"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_qwen35v_model_registered()
    test_qwen35v_creation()
    test_qwen35v_config_validation()
    test_qwen35_vision_encoder_creation()
