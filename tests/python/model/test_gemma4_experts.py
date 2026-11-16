"""Expert dispatch parity for decode and grouped prefill, without full decoder claims."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlc_llm.model.gemma4.gemma4_experts import Gemma4Experts, Gemma4ExpertsConfig, huggingface


def test_parameter_mapping():
    config = Gemma4ExpertsConfig(2816, 704, 128, 8)
    model = Gemma4Experts(config)
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    assert {name: tuple(p.shape) for name, p in params} == {
        "gate_up_proj.weight": (128, 1408, 2816),
        "down_proj.weight": (128, 2816, 704),
    }
    mapping = huggingface(config, 0)
    assert mapping.param_map == {
        "gate_up_proj.weight": ["model.language_model.layers.0.experts.gate_up_proj"],
        "down_proj.weight": ["model.language_model.layers.0.experts.down_proj"],
    }
    with pytest.raises(ValueError):
        model.to("float16")


@pytest.mark.skipif(
    os.getenv("MLC_GEMMA4_EXPERT_NUMERICAL") != "1", reason="Opt in to Metal expert execution"
)
def test_compiled_experts_match_hf(tmp_path):
    import torch
    import tvm_tirx
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextExperts
    from tvm import relax

    from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
    from mlc_llm.support import auto_target

    torch.set_num_threads(2)
    torch.manual_seed(61)
    config = Gemma4ExpertsConfig(16, 8, 4, 2)
    reference = Gemma4TextExperts(
        Gemma4TextConfig(
            hidden_size=16,
            moe_intermediate_size=8,
            num_experts=4,
            top_k_experts=2,
            hidden_activation="gelu_pytorch_tanh",
        )
    ).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.4, 0.4)
    model = Gemma4Experts(config)
    mod, params, ext = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    target, build = auto_target.detect_target_and_host("metal")
    library = tmp_path / "experts.dylib"
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=dict(
                    params=schema,
                    pipeline_parallel_stages=1,
                    tensor_parallel_shards=1,
                    prefill_chunk_size=8,
                ),
                ext_mods=ext,
                variable_bounds={"tokens": 8, "batch_size": 1, "vocab_size": 8},
            ),
        )
    device = tvm_tirx.metal()
    vm = relax.VirtualMachine(tvm_tirx.runtime.load_module(str(library)), device)
    state = reference.state_dict()
    packed = [
        tvm_tirx.runtime.tensor(state[n.removesuffix(".weight")].detach().numpy(), device=device)
        for n, _ in params
    ]
    for tokens in (1, 3, 8):
        hidden = torch.randn(tokens, 16)
        # Alternate ranks and leave some experts unused for small batches.
        indices = torch.tensor(
            [[3, 1], [0, 2], [2, 3], [1, 0], [3, 0], [2, 1], [1, 3], [0, 1]][:tokens]
        )
        weights = torch.linspace(0.2, 1.7, tokens * 2).reshape(tokens, 2)
        with torch.no_grad():
            expected = reference(hidden, indices, weights).numpy()
        args = [
            tvm_tirx.runtime.tensor(x, device=device)
            for x in (hidden.numpy(), indices.numpy().astype("int32"), weights.numpy())
        ]
        actual = vm["prefill"](*args, packed).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        if tokens == 1:
            decoded = vm["decode"](*args, packed).numpy()
            np.testing.assert_allclose(decoded, expected, atol=2e-5, rtol=2e-5)
