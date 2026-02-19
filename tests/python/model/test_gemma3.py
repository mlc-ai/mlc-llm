# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Gemma3 model architecture."""

import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS


def test_gemma3_model_registered():
    """Verify Gemma3 model is in the registry."""
    assert "gemma3" in MODELS, "gemma3 should be registered in MODELS"


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma3_2b",
        "gemma3_9b",
    ],
)
def test_gemma3_creation(model_name: str):
    """Test Gemma3 model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from preset
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters are extracted
    """
    model_info = MODELS["gemma3"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Optional: show module structure
    mod.show(black_format=False)

    # Print parameters for debugging
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_gemma3_config_validation():
    """Test Gemma3 configuration has required fields."""
    model_info = MODELS["gemma3"]
    config = model_info.config.from_dict(MODEL_PRESETS["gemma3_2b"])

    # Check required config parameters
    assert hasattr(config, "hidden_size") and config.hidden_size > 0
    assert hasattr(config, "num_hidden_layers") and config.num_hidden_layers > 0
    assert hasattr(config, "num_attention_heads") and config.num_attention_heads > 0
    assert hasattr(config, "vocab_size") and config.vocab_size > 0

    print(
        f"Gemma3 Config: hidden_size={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, "
        f"vocab={config.vocab_size}"
    )


if __name__ == "__main__":
    # Allow running tests directly
    test_gemma3_creation("gemma3_2b")
    test_gemma3_creation("gemma3_9b")
