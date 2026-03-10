# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Gemma3V vision-language model architecture."""

import pytest

from mlc_llm.model import MODELS


# Minimal config dict with small dimensions for fast testing.
# Mirrors the structure of a real HuggingFace gemma-3-4b-it config.json.
SMALL_GEMMA3V_CONFIG = {
    "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "context_window_size": 1024,
        "prefill_chunk_size": 512,
        "sliding_window_size": 256,
        "sliding_window_pattern": 6,
        "hidden_activation": "gelu_pytorch_tanh",
    },
    "vision_config": {
        "hidden_size": 64,
        # image_size / patch_size = 4 (grid), 4x4 avg_pool -> 1x1 -> 1 token
        "image_size": 56,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "patch_size": 14,
        "num_channels": 3,
        "layer_norm_eps": 1e-6,
    },
    "vocab_size": 262208,
    "mm_tokens_per_image": 1,
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
}


def test_gemma3v_model_registered():
    """Verify Gemma3V model is in the registry."""
    assert "gemma3_v" in MODELS, "gemma3_v should be registered in MODELS"


def test_gemma3v_creation():
    """Test Gemma3V model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include vision_tower, language_model, and projector components
    """
    model_info = MODELS["gemma3_v"]
    config = model_info.config.from_dict(SMALL_GEMMA3V_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Verify VLM composition: params from all three components
    param_names = [name for name, _ in named_params]
    has_vision = any(n.startswith("vision_tower.") for n in param_names)
    has_language = any(n.startswith("language_model.") for n in param_names)
    has_projector = any(n.startswith("multi_modal_projector.") for n in param_names)
    assert has_vision, "Should have vision_tower parameters"
    assert has_language, "Should have language_model parameters"
    assert has_projector, "Should have multi_modal_projector parameters"

    # Verify all expected functions are exported
    expected_funcs = [
        "embed",
        "image_embed",
        "prefill",
        "decode",
        "batch_prefill",
        "batch_decode",
        "batch_verify",
        "create_paged_kv_cache",
    ]
    for func_name in expected_funcs:
        assert func_name in mod, f"Module should contain '{func_name}' function"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_gemma3v_config_validation():
    """Test Gemma3V configuration has required fields."""
    model_info = MODELS["gemma3_v"]
    config = model_info.config.from_dict(SMALL_GEMMA3V_CONFIG)

    # Text config fields
    assert hasattr(config.text_config, "hidden_size") and config.text_config.hidden_size > 0
    assert (
        hasattr(config.text_config, "num_hidden_layers")
        and config.text_config.num_hidden_layers > 0
    )

    # Vision config fields
    assert hasattr(config.vision_config, "image_size") and config.vision_config.image_size > 0
    assert hasattr(config.vision_config, "patch_size") and config.vision_config.patch_size > 0

    # VLM-specific fields
    assert config.mm_tokens_per_image > 0

    print(
        f"Gemma3V Config: text_hidden={config.text_config.hidden_size}, "
        f"vision_hidden={config.vision_config.hidden_size}, "
        f"text_layers={config.text_config.num_hidden_layers}, "
        f"vision_layers={config.vision_config.num_hidden_layers}, "
        f"mm_tokens={config.mm_tokens_per_image}"
    )


if __name__ == "__main__":
    test_gemma3v_model_registered()
    test_gemma3v_creation()
    test_gemma3v_config_validation()
