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


def test_reference_defaults_and_nullable_head_dim():
    assert Lfm3Config.__dataclass_fields__["intermediate_size"].default == 7680
    assert make_config(head_dim=None).head_dim == 128


@pytest.mark.parametrize("fraction,theta", [(0.5, 1000000.0), (1.0, 10000.0)])
def test_zero_tail_rope_matches_reference(fraction, theta):
    import numpy as np
    import tvm
    from tvm import relax
    from tvm.relax.frontend import nn

    from mlc_llm.model.lfm3.lfm3_model import _apply_rope

    class Rotary(nn.Module):
        def forward(self, x, positions):
            return _apply_rope(x, positions, theta, fraction)

    mod, _ = Rotary().export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([3, 2, 128], "float32"),
                "positions": nn.spec.Tensor([3], "int32"),
            }
        }
    )
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    x = np.random.default_rng(0).standard_normal((3, 2, 128)).astype("float32")
    positions = np.array([0, 255, 8192], dtype="int32")
    pairs = int(fraction * 128) // 2
    frequencies = np.zeros(64, dtype="float32")
    frequencies[:pairs] = 1.0 / (theta ** (np.arange(0, 2 * pairs, 2, dtype="float32") / 128))
    angles = positions[:, None].astype("float32") * frequencies[None, :]
    angles = np.concatenate([angles, angles], axis=-1)[:, None, :]
    rotated = np.concatenate([-x[..., 64:], x[..., :64]], axis=-1)
    expected = x * np.cos(angles) + rotated * np.sin(angles)
    actual = vm["forward"](tvm.runtime.tensor(x), tvm.runtime.tensor(positions)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    if fraction < 1:
        np.testing.assert_array_equal(actual[..., pairs:64], x[..., pairs:64])
        np.testing.assert_array_equal(actual[..., 64 + pairs :], x[..., 64 + pairs :])
