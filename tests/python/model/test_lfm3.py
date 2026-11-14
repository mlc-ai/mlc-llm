"""LFM3 configuration, rotary embedding, loader, and quantization tests."""

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
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_zero_tail_rope_matches_reference(fraction, theta, dtype):
    import numpy as np
    import tvm_tirx
    from tvm import relax
    from tvm.relax.frontend import nn

    from mlc_llm.model.lfm3.lfm3_model import _apply_rope

    class Rotary(nn.Module):
        def forward(self, x, positions):
            return _apply_rope(x, positions, theta, fraction)

    mod, _ = Rotary().export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([3, 2, 128], dtype),
                "positions": nn.spec.Tensor([3], "int32"),
            }
        }
    )
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    x = np.random.default_rng(0).standard_normal((3, 2, 128)).astype(dtype)
    positions = np.array([0, 255, 262143], dtype="int32")
    pairs = int(fraction * 128) // 2
    frequencies = np.zeros(64, dtype="float32")
    frequencies[:pairs] = 1.0 / (theta ** (np.arange(0, 2 * pairs, 2, dtype="float32") / 128))
    angles = positions[:, None].astype("float32") * frequencies[None, :]
    angles = np.concatenate([angles, angles], axis=-1)[:, None, :]
    rotated = np.concatenate([-x[..., 64:], x[..., :64]], axis=-1)
    cos = np.cos(angles).astype(dtype).astype("float32")
    sin = np.sin(angles).astype(dtype).astype("float32")
    expected = (x.astype("float32") * cos + rotated.astype("float32") * sin).astype(dtype)
    actual = vm["forward"](tvm_tirx.runtime.tensor(x), tvm_tirx.runtime.tensor(positions)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    if fraction < 1:
        np.testing.assert_array_equal(actual[..., pairs:64], x[..., pairs:64])
        np.testing.assert_array_equal(actual[..., 64 + pairs :], x[..., 64 + pairs :])


def test_hybrid_export_and_loader_names():
    from mlc_llm.model.lfm3.lfm3_loader import huggingface
    from mlc_llm.model.lfm3.lfm3_model import Lfm3ForCausalLM
    from mlc_llm.quantization import QUANTIZATION

    config = make_config(
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        vocab_size=256,
    )
    model = Lfm3ForCausalLM(config)
    assert model.attn_kind == ["mha", "mha_sliding"]
    assert model.num_conv_layers == 1
    mapping = huggingface(config, QUANTIZATION["q0f32"]).param_map
    for index in (1, 2):
        prefix = f"model.layers.{index}.self_attn"
        assert mapping[f"{prefix}.qkv_proj.weight"] == [
            f"{prefix}.{name}_proj.weight" for name in "qkv"
        ]
        assert mapping[f"{prefix}.q_layernorm.weight"] == [f"{prefix}.q_norm.weight"]
        assert mapping[f"{prefix}.k_layernorm.weight"] == [f"{prefix}.k_norm.weight"]
        assert mapping[f"{prefix}.out_proj.weight"] == [f"{prefix}.o_proj.weight"]
        assert mapping[f"{prefix}.gate_proj.weight"] == [f"{prefix}.gate_proj.weight"]
    assert "model.layers.0.post_operator_norm.weight" in mapping
    assert "model.layers.0.post_ffn_norm.weight" in mapping


@pytest.mark.parametrize("quant_name", ["q0f16", "q4f16_1", "q4f32_1"])
def test_registered_quantization_exports_hybrid_model(quant_name):
    import copy

    from mlc_llm.model import MODELS
    from mlc_llm.quantization import QUANTIZATION

    config = make_config(
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        vocab_size=256,
    )
    registered = MODELS["lfm3"]
    quantization = copy.deepcopy(QUANTIZATION[quant_name])
    model, mapping = registered.quantize[quantization.kind](config, quantization)
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    params = dict(named_params)
    assert model.attn_kind == ["mha", "mha_sliding"]
    assert model.num_conv_layers == 1
    assert params["model.layers.0.post_operator_norm.weight"].dtype == quantization.model_dtype
    if quantization.kind == "no-quant":
        assert not mapping.param_map
        assert params["model.layers.1.self_attn.qkv_proj.weight"].dtype == "float16"
    else:
        for layer in (1, 2):
            name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
            assert name not in params
            assert len(mapping.param_map[name]) == 2
            assert all(target in params for target in mapping.param_map[name])
        assert "model.layers.0.post_operator_norm.weight" not in mapping.param_map


def test_sliding_attention_without_full_attention_rope():
    from mlc_llm.model.lfm3.lfm3_model import Lfm3ForCausalLM

    config = make_config(
        num_hidden_layers=2,
        layer_types=["conv", "sliding_attention"],
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    )
    model = Lfm3ForCausalLM(config)
    assert model.attn_kind == ["mha_sliding"]
    assert model.num_attention_layers == 1
    assert model.num_conv_layers == 1


def test_prefill_chunks_stay_inside_sliding_window():
    from mlc_llm.interface.compile import ModelConfigOverride

    config = make_config(prefill_chunk_size=1024)
    assert config.prefill_chunk_size == 256
    assert config.context_window_size == 262144
    assert ModelConfigOverride(prefill_chunk_size=8192).apply(config).prefill_chunk_size == 256
    assert make_config(prefill_chunk_size=128).prefill_chunk_size == 128
    assert (
        make_config(
            num_hidden_layers=2,
            layer_types=["conv", "full_attention"],
            prefill_chunk_size=1024,
        ).prefill_chunk_size
        == 1024
    )
