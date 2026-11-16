"""Pre-RoPE projection parity; this is not full Gemma 4 decoder support."""

import json
from pathlib import Path

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.gemma4.gemma4_config import Gemma4TextConfig
from mlc_llm.model.gemma4.gemma4_projection import (
    Gemma4ProjectionConfig,
    Gemma4TextProjections,
    projection_huggingface,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures/gemma4_projection.json").read_text())


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["model"])
def test_actual_checkpoint_projection_shapes_and_names(fixture):
    config = Gemma4ProjectionConfig.from_text_config(fixture["text_config"])
    for index in fixture["layers"]:
        model = Gemma4TextProjections(config, index)
        _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        params = dict(params)
        dim, heads, shared = config.dimensions(index)
        assert tuple(params["q_proj.weight"].shape) == (
            config.num_attention_heads * dim,
            config.hidden_size,
        )
        assert tuple(params["k_proj.weight"].shape) == (heads * dim, config.hidden_size)
        assert ("v_proj.weight" in params) is not shared
        mapping = projection_huggingface(config, index)
        source = {key for names in mapping.param_map.values() for key in names}
        expected = {
            key
            for key in fixture["attention_keys"]
            if key.startswith(f"model.language_model.layers.{index}.self_attn.")
            and not key.endswith("o_proj.weight")
        }
        assert source == expected
        assert set(mapping.param_map) == set(params)
    # A component config must not accidentally enable either unsupported full decoder.
    with pytest.raises(ValueError):
        Gemma4TextConfig.from_dict(fixture["text_config"])


@pytest.mark.parametrize("global_layer", [False, True])
def test_tiny_compiled_projection_matches_hf(global_layer):
    torch = pytest.importorskip("torch")
    ref = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    raw = dict(FIXTURES[0]["text_config"])
    raw.update(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
    )
    config = Gemma4ProjectionConfig.from_text_config(raw)
    layer = int(global_layer)
    reference = ref.Gemma4TextAttention(ref.Gemma4TextConfig(**raw), layer).eval()
    torch.manual_seed(17)
    with torch.no_grad():
        for value in reference.parameters():
            value.uniform_(-0.5, 0.5)
    model = Gemma4TextProjections(config, layer)
    mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    state = reference.state_dict()
    weights = [tvm_tirx.runtime.tensor(state[name].numpy()) for name, _ in params]
    hidden = torch.randn(1, 3, 16)
    with torch.no_grad():
        q = reference.q_proj(hidden).reshape(1, 3, 4, model.head_dim)
        k = reference.k_proj(hidden).reshape(1, 3, model.num_kv_heads, model.head_dim)
        v = k if reference.v_proj is None else reference.v_proj(hidden).reshape(k.shape)
        expected = [reference.q_norm(q), reference.k_norm(k), reference.v_norm(v)]
    actual = vm["forward"](tvm_tirx.runtime.tensor(hidden.numpy()), weights)
    for result, target in zip(actual, expected):
        np.testing.assert_allclose(result.numpy(), target.numpy(), atol=1e-6, rtol=1e-5)
    if global_layer:
        assert not torch.allclose(expected[1], expected[2]), "K and V normalization must differ"


@pytest.mark.parametrize(
    "changes",
    [
        {"num_kv_shared_layers": 1},
        {"attention_bias": True},
        {"num_global_key_value_heads": 3},
        {"hidden_size": 0},
    ],
)
def test_invalid_projection_config(changes):
    raw = dict(FIXTURES[0]["text_config"])
    raw.update(changes)
    with pytest.raises(ValueError):
        Gemma4ProjectionConfig.from_text_config(raw)
