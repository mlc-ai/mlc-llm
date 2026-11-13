"""Configuration invariants for LFM3 before model registration."""

import pytest

from mlc_llm.model.lfm3.lfm3_config import Lfm3Config


def make_config(**overrides):
    config = dict(
        vocab_size=128000,
        hidden_size=2048,
        num_hidden_layers=3,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=7680,
        head_dim=128,
        layer_types=["conv", "full_attention", "sliding_attention"],
        context_window_size=262144,
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    config.update(overrides)
    return Lfm3Config(**config)


def test_mixed_attention_counts_and_unadjusted_ffn():
    config = make_config()
    assert config.num_attention_layers == 2
    assert config.num_conv_layers == 1
    assert config.ffn_dim == 7680
    assert config.conv_L_cache == 7


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"layer_types": ["conv", "unknown", "full_attention"]}, "explicit"),
        ({"sliding_window": 0}, "positive"),
        ({"rope_parameters": {}}, "RoPE parameters"),
        ({"block_auto_adjust_ff_dim": True}, "without adjustment"),
    ],
)
def test_invalid_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_config(**overrides)
