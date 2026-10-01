# pylint: disable=invalid-name,missing-docstring
"""Unit tests for NemotronH hybrid Mamba2-Attention model architecture."""

import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS


def test_nemotron_h_model_registered():
    """Verify NemotronH model is in the registry."""
    assert "nemotron_h" in MODELS, "nemotron_h should be registered in MODELS"


@pytest.mark.parametrize(
    "model_name",
    [
        "nemotron_h_4b",
    ],
)
def test_nemotron_h_creation(model_name: str):
    """Test NemotronH model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from preset
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters are extracted
    """
    model_info = MODELS["nemotron_h"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),
    )
    assert mod is not None
    assert len(named_params) > 0
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_nemotron_h_config_validation():
    """Test NemotronH configuration has required fields."""
    model_info = MODELS["nemotron_h"]
    config = model_info.config.from_dict(MODEL_PRESETS["nemotron_h_4b"])
    assert hasattr(config, "hidden_size") and config.hidden_size == 3136
    assert config.num_hidden_layers == 42
    assert config.num_attention_layers == 4
    assert config.num_attention_heads == 40
    assert config.num_key_value_heads == 8
    assert config.vocab_size == 131072
    assert config.mamba_num_heads == 96
    assert config.mamba_chunk_size == 256
    assert config.layers_block_type[0] == "mamba"
    assert "attention" in config.layers_block_type
    assert "mlp" in config.layers_block_type
    assert "moe" not in config.layers_block_type
    print(
        f"NemotronH Config: hidden_size={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"attn_layers={config.num_attention_layers}, "
        f"vocab={config.vocab_size}"
    )


def test_nemotron_h_hf_config_loading():
    """Test that NemotronH config loads correctly from HF config.json format."""
    from mlc_llm.model.nemotron_h.nemotron_h_model import NemotronHConfig

    # Simulate what HF config.json looks like
    hf_config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 131072,
        "hidden_size": 3136,
        "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12544,
        "ssm_state_size": 128,
        "mamba_num_heads": 96,
        "mamba_head_dim": 80,
        "conv_kernel": 4,
        "chunk_size": 256,
        "use_conv_bias": True,
        "n_groups": 8,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 262144,
    }
    config = NemotronHConfig.from_dict(hf_config)
    assert len(config.layers_block_type) == 42
    assert config.num_attention_layers == 4
    assert config.mamba_d_conv == 4  # mapped from conv_kernel
    assert config.mamba_chunk_size == 256  # mapped from chunk_size
    assert config.mamba_n_groups == 8  # mapped from n_groups
    assert config.mamba_conv_bias == True  # mapped from use_conv_bias
    assert config.layer_norm_epsilon == 1e-5  # mapped from rms_norm_eps


if __name__ == "__main__":
    test_nemotron_h_creation("nemotron_h_4b")
