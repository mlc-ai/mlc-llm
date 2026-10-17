"""Unit tests for the SmolLM3 model architecture."""

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.frontend import nn

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.model.smollm3.smollm3_model import _apply_rope


def test_smollm3_model_registered():
    """Verify SmolLM3 model is in the registry."""
    assert "smollm3" in MODELS, "smollm3 should be registered in MODELS"


def test_smollm3_creation():
    """Test SmolLM3 model creation and export; only the RoPE layers rotate q and k."""
    model_info = MODELS["smollm3"]
    config = model_info.config.from_dict(MODEL_PRESETS["smollm3_3b"])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(spec=model.get_default_spec())
    assert len(named_params) > 0
    prefill = mod["batch_prefill"].script(show_meta=False)
    assert prefill.count("R.call_tir(Module.rope") == sum(config.no_rope_layers) == 27
    # Every fourth layer (3, 7, ...) skips RoPE, as in HF.
    use_rope = [layer.self_attn.use_rope for layer in model.model.layers]
    assert [i for i, rope in enumerate(use_rope) if not rope] == list(range(3, 36, 4))


def test_smollm3_default_no_rope_layers():
    """Without no_rope_layers every no_rope_layer_interval-th layer skips RoPE, as in HF."""
    preset = dict(MODEL_PRESETS["smollm3_3b"])
    preset.pop("no_rope_layers")
    config = MODELS["smollm3"].config.from_dict(preset)
    assert config.no_rope_layers == MODEL_PRESETS["smollm3_3b"]["no_rope_layers"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_scaling": {"rope_type": "yarn", "factor": 2.0}},
        {"rope_parameters": {"rope_theta": 5000000.0, "rope_type": "yarn"}},
        {"use_sliding_window": True},
        {"pipeline_parallel_stages": 2},
    ],
)
def test_smollm3_rejected_configs(overrides):
    """Settings the port does not implement are rejected instead of silently ignored."""
    with pytest.raises(ValueError):
        MODELS["smollm3"].config.from_dict({**MODEL_PRESETS["smollm3_3b"], **overrides})


def test_smollm3_rope_theta():
    """rope_theta is required; default-type rope_parameters (the transformers 5 form) set it."""
    preset = dict(MODEL_PRESETS["smollm3_3b"])
    theta = preset.pop("rope_theta")
    with pytest.raises(ValueError):
        MODELS["smollm3"].config.from_dict(preset)
    config = MODELS["smollm3"].config.from_dict(
        {**preset, "rope_parameters": {"rope_theta": theta, "rope_type": "default"}}
    )
    assert config.position_embedding_base == theta


def test_smollm3_rope_matches_reference():
    """_apply_rope is Llama's rotate-half RoPE with the default frequencies."""
    num_tokens, num_heads, head_dim, theta = 5, 3, 16, 5000000.0

    class _Rope(nn.Module):
        def forward(self, x: nn.Tensor, positions: nn.Tensor):
            return _apply_rope(x, positions, theta)

    mod, _ = _Rope().export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([num_tokens, num_heads, head_dim], "float32"),
                "positions": nn.spec.Tensor([num_tokens], "int32"),
            }
        }
    )
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    rng = np.random.default_rng(0)
    x = rng.standard_normal((num_tokens, num_heads, head_dim)).astype("float32")
    positions = np.array([0, 1, 7, 100, 4095], dtype="int32")
    out = vm["forward"](tvm.runtime.tensor(x), tvm.runtime.tensor(positions)).numpy()

    half = head_dim // 2
    inv_freq = theta ** (-np.arange(half, dtype="float64") * 2 / head_dim)
    angle = positions[:, None].astype("float64") * inv_freq[None, :]
    cos = np.concatenate([np.cos(angle)] * 2, axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(angle)] * 2, axis=-1)[:, None, :]
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    np.testing.assert_allclose(out, x * cos + rotated * sin, rtol=1e-5, atol=1e-5)
