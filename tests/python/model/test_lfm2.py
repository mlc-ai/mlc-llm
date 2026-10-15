"""Unit tests for the LFM2 model architecture."""

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.frontend import nn

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.model.lfm2.lfm2_model import _causal_depthwise_conv1d
from mlc_llm.quantization import QUANTIZATION


def _layer_types(num_layers: int, attention_layers: list) -> list:
    return ["full_attention" if i in attention_layers else "conv" for i in range(num_layers)]


# LFM2 and LFM2.5 checkpoints spell the same settings differently.
# LFM2-350M: block_ff_dim with the FFN width auto-adjusted, and full_attn_idxs.
LFM2_350M = {
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 65536,
    "block_ff_dim": 6656,
    "full_attn_idxs": [2, 5, 8, 10, 12, 14],
    "rope_theta": 1000000.0,
    "max_position_embeddings": 128000,
}
# LFM2.5-2.6B: intermediate_size used as is, layer_types and rope_parameters.
LFM2_5_2_6B = {
    "hidden_size": 2048,
    "num_hidden_layers": 30,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128000,
    "intermediate_size": 10752,
    "block_auto_adjust_ff_dim": False,
    "layer_types": _layer_types(30, [2, 5, 9, 13, 17, 21, 24, 27]),
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "max_position_embeddings": 131072,
}


def test_lfm2_model_registered():
    """Verify LFM2 model is in the registry."""
    assert "lfm2" in MODELS, "lfm2 should be registered in MODELS"


@pytest.mark.parametrize(
    "hf_config, attention_layers, ffn_dim, rope_theta, num_params",
    [
        pytest.param(LFM2_350M, [2, 5, 8, 10, 12, 14], 4608, 1e6, 354483968, id="LFM2-350M"),
        pytest.param(
            LFM2_5_2_6B, [2, 5, 9, 13, 17, 21, 24, 27], 10752, 1e7, 2697198592, id="LFM2.5-2.6B"
        ),
    ],
)
def test_lfm2_config(hf_config, attention_layers, ffn_dim, rope_theta, num_params):
    """The config reads both spellings as HF does; parameter totals match the HF checkpoints."""
    config_cls = MODELS["lfm2"].config
    config = config_cls.from_dict(hf_config)
    assert [i for i, t in enumerate(config.layer_types) if t == "full_attention"] == (
        attention_layers
    )
    assert config.ffn_dim == ffn_dim
    assert config.rope_theta == rope_theta
    model = MODELS["lfm2"].model(config)
    assert sum(int(np.prod([int(x) for x in p.shape])) for _, p in model.named_parameters()) == (
        num_params
    )
    # The config is stable when saved to and loaded from mlc-chat-config.json.
    assert config_cls.from_dict(config.asdict()).asdict() == config.asdict()


def test_lfm2_creation():
    """Test LFM2 model creation and export to TVM IR in float16."""
    model_info = MODELS["lfm2"]
    config = model_info.config.from_dict(MODEL_PRESETS["lfm2.5_1.2b"])
    model = model_info.model(config)
    model.to("float16")
    mod, named_params = model.export_tvm(spec=model.get_default_spec())
    assert mod is not None
    names = {name for name, _ in named_params}
    assert "model.layers.0.conv.conv.weight" in names
    assert "model.layers.2.self_attn.q_layernorm.weight" in names
    assert "model.embedding_norm.weight" in names


def test_lfm2_config_defaults_and_overrides():
    """Without a layout every layer is attention, as in HF, which is rejected because there is no
    conv layer; tie_embedding and the batch cap."""
    hf_config = dict(LFM2_350M)
    hf_config.pop("full_attn_idxs")
    with pytest.raises(ValueError, match="without a conv layer"):
        MODELS["lfm2"].config.from_dict(hf_config)
    config = MODELS["lfm2"].config.from_dict(
        {**LFM2_350M, "tie_embedding": False, "max_batch_size": 128}
    )
    assert not config.tie_word_embeddings
    assert config.max_batch_size == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_parameters": {"rope_theta": 1e6, "rope_type": "yarn"}},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {"rope_scaling": {"type": "dynamic", "factor": 2.0}},
        {"conv_bias": True},
        {"num_key_value_heads": 5},
    ],
)
def test_lfm2_rejected_configs(overrides):
    """Settings the port does not implement are rejected instead of silently ignored."""
    with pytest.raises(ValueError):
        MODELS["lfm2"].config.from_dict({**LFM2_350M, **overrides})


@pytest.mark.parametrize("chunks", [[9], [1, 1, 7], [2, 3, 4]])
def test_lfm2_conv_matches_reference(chunks):
    """The causal conv carried across prefill chunks equals one numpy conv over the whole input."""
    channels, kernel_size = 4, 3

    class _Conv(nn.Module):
        def forward(self, x: nn.Tensor, state: nn.Tensor, weight: nn.Tensor):
            return _causal_depthwise_conv1d(x, state, weight)

    mod, _ = _Conv().export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([1, "s", channels], "float32"),
                "state": nn.spec.Tensor([1, kernel_size - 1, channels], "float32"),
                "weight": nn.spec.Tensor([channels, 1, kernel_size], "float32"),
            }
        }
    )
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, sum(chunks), channels)).astype("float32")
    weight = rng.standard_normal((channels, 1, kernel_size)).astype("float32")

    state = np.zeros((1, kernel_size - 1, channels), "float32")
    outputs, start = [], 0
    for size in chunks:
        out, state = vm["forward"](
            tvm.runtime.tensor(x[:, start : start + size]),
            tvm.runtime.tensor(state),
            tvm.runtime.tensor(weight),
        )
        outputs.append(out.numpy())
        state = state.numpy()
        start += size

    padded = np.concatenate([np.zeros((1, kernel_size - 1, channels), "float32"), x], axis=1)
    expected = sum(padded[:, k : k + x.shape[1]] * weight[:, 0, k] for k in range(kernel_size))
    np.testing.assert_allclose(np.concatenate(outputs, axis=1), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state, x[:, -(kernel_size - 1) :], rtol=0, atol=0)


def test_lfm2_loader_fused_order():
    """q, k, v and w1 (gate), w3 (up) are fused in the order the model splits them."""
    config = MODELS["lfm2"].config.from_dict(MODEL_PRESETS["lfm2.5_1.2b"])
    attention_layer = config.layer_types.index("full_attention")
    param_map = (
        MODELS["lfm2"].source["huggingface-safetensor"](config, QUANTIZATION["q0f32"]).param_map
    )
    prefix = f"model.layers.{attention_layer}"
    assert param_map[f"{prefix}.self_attn.qkv_proj.weight"] == [
        f"{prefix}.self_attn.{p}_proj.weight" for p in "qkv"
    ]
    assert param_map["model.layers.0.feed_forward.gate_up_proj.weight"] == [
        "model.layers.0.feed_forward.w1.weight",
        "model.layers.0.feed_forward.w3.weight",
    ]
    assert param_map["model.layers.0.feed_forward.down_proj.weight"] == [
        "model.layers.0.feed_forward.w2.weight"
    ]
