"""HF parity for routing and parallel branches, without claiming expert kernels."""

import json
from pathlib import Path

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.gemma4.gemma4_moe_components import (
    Gemma4MoEComponentConfig,
    Gemma4ParallelFeedForward,
    Gemma4Router,
    component_huggingface,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/gemma4_projection.json").read_text())[1]


def compile_component(model, state):
    mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    weights = [tvm_tirx.runtime.tensor(state[name].detach().numpy()) for name, _ in params]
    return relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu()), weights


def test_router_and_parallel_branches_match_hf():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    raw = dict(FIXTURE["text_config"])
    raw.update(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
    )
    config = Gemma4MoEComponentConfig.from_text_config(raw)
    reference = hf.Gemma4TextDecoderLayer(hf.Gemma4TextConfig(**raw), 0).eval()
    torch.manual_seed(59)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.7, 0.7)
        reference.router.per_expert_scale.copy_(torch.tensor([0.3, 1.1, 1.7, 2.5]))
    hidden = torch.randn(5, 16)
    router_vm, router_params = compile_component(
        Gemma4Router(config), reference.router.state_dict()
    )
    branch_vm, branch_params = compile_component(
        Gemma4ParallelFeedForward(config), reference.state_dict()
    )
    with torch.no_grad():
        probs, weights, indices = reference.router(hidden)
        expert_input = reference.pre_feedforward_layernorm_2(hidden)
        experts = reference.experts(expert_input, indices, weights)
        dense = reference.post_feedforward_layernorm_1(
            reference.mlp(reference.pre_feedforward_layernorm(hidden))
        )
        expected = hidden + reference.post_feedforward_layernorm(
            dense + reference.post_feedforward_layernorm_2(experts)
        )
    tensor = tvm_tirx.runtime.tensor(hidden.numpy())
    result = router_vm["forward"](tensor, router_params)
    np.testing.assert_allclose(result[0].numpy(), probs.numpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(result[1].numpy(), weights.numpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(result[2].numpy(), indices.numpy())
    assert not np.allclose(result[1].numpy().sum(-1), 1), (
        "Expert scaling occurs after top-k normalization"
    )
    actual_input = branch_vm["expert_input"](tensor, branch_params).numpy()
    np.testing.assert_allclose(actual_input, expert_input.numpy(), atol=1e-6, rtol=1e-5)
    actual = branch_vm["forward"](
        tensor, tvm_tirx.runtime.tensor(experts.numpy()), branch_params
    ).numpy()
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


def test_actual_config_export_and_mapping():
    config = Gemma4MoEComponentConfig.from_text_config(FIXTURE["text_config"])
    assert config.num_experts == 128 and config.top_k_experts == 8
    mapped_sources = set()
    for model in [Gemma4Router(config), Gemma4ParallelFeedForward(config)]:
        _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        mapping = component_huggingface(model, 0)
        assert set(mapping.param_map) == {n for n, _ in params}
        assert all(len(names) == 1 for names in mapping.param_map.values())
        mapped_sources.update(n for names in mapping.param_map.values() for n in names)
        if isinstance(model, Gemma4Router):
            shapes = {n: tuple(p.shape) for n, p in params}
            assert shapes["proj.weight"] == (128, 2816)
            assert shapes["per_expert_scale"] == (128,)

    assert mapped_sources == set(FIXTURE["feedforward_keys"])


@pytest.mark.parametrize(
    "change",
    [
        {"num_experts": 0},
        {"top_k_experts": 129},
        {"enable_moe_block": False},
        {"num_kv_shared_layers": 1},
    ],
)
def test_invalid_moe_config(change):
    raw = dict(FIXTURE["text_config"])
    raw.update(change)
    with pytest.raises(ValueError):
        Gemma4MoEComponentConfig.from_text_config(raw)
