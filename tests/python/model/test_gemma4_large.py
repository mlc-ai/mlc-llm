"""Large Gemma 4 text architecture and uniform-cache equivalence."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tvm_tirx
from tvm import relax, te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op
from tvm_tirx import tirx

from mlc_llm.model import MODELS
from mlc_llm.model.gemma4.gemma4_large_loader import huggingface
from mlc_llm.model.gemma4.gemma4_large_model import (
    Gemma4LargeConfig,
    Gemma4LargeForCausalLM,
    pack_cache_heads,
)
from mlc_llm.model.gemma4.gemma4_moe import Gemma4LargeExperts, Gemma4Router
from mlc_llm.quantization import QUANTIZATION

CHECKPOINTS = json.loads((Path(__file__).parent / "fixtures/gemma4_large.json").read_text())


def tiny(moe=False):
    return Gemma4LargeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        vocab_size=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        enable_moe_block=moe,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
    )


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("quant", ["q0f32", "q0f16", "q4f16_1", "q4f32_1"])
def test_exports(moe, quant):
    c = tiny(moe)
    q = QUANTIZATION[quant]
    m, _ = MODELS["gemma4_large_text"].quantize[q.kind](c, q)
    _, params, _ = m.export_tvm(spec=m.get_default_spec(), allow_extern=True)
    assert params
    mapping = huggingface(c, q)
    assert "language_model.layers.1.self_attn.v_proj.weight" not in mapping.param_map
    assert "language_model.layers.0.self_attn.v_proj.weight" in mapping.param_map


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda f: f["model"])
def test_checkpoint_mapping(checkpoint):
    config = Gemma4LargeConfig.from_dict({"text_config": checkpoint["text_config"]})
    mapping = huggingface(config, QUANTIZATION["q4f16_1"])
    sources = {name for names in mapping.param_map.values() for name in names}
    prefixes = tuple(
        f"model.language_model.layers.{layer}.self_attn." for layer in checkpoint["layers"]
    )
    assert {name for name in sources if name.startswith(prefixes)} == set(
        checkpoint["attention_keys"]
    )
    if "feedforward_keys" in checkpoint:
        layer = "model.language_model.layers.0."
        branches = tuple(
            layer + name
            for name in ("mlp.", "router.", "experts.", "pre_feedforward_", "post_feedforward_")
        )
        assert {name for name in sources if name.startswith(branches)} == set(
            checkpoint["feedforward_keys"]
        )


class DenseCache:
    """CPU test cache: exact causal/sliding full-prompt attention, no runtime claims."""

    def __init__(self, c):
        self.c = c

    def get_query_positions(self, length):
        return op.tensor_expr_op(
            lambda: te.compute((length,), lambda i: i.astype("int32")), "positions", []
        )

    def attention_with_fused_qkv(self, index, qkv, num_q_heads, sm_scale):
        c = self.c
        q, k, v = op.split(qkv, [num_q_heads, num_q_heads + c.num_key_value_heads], axis=2)
        k = pack_cache_heads(k, num_q_heads, c.physical_head_dim)
        v = pack_cache_heads(v, num_q_heads, c.physical_head_dim)
        q, k, v = [op.permute_dims(x, (0, 2, 1, 3)) for x in (q, k, v)]
        scores = op.matmul(q, op.permute_dims(k, (0, 1, 3, 2)))

        def mask(x):
            def value(b, h, i, j):
                valid = j <= i
                if c.layer_types[index] == "sliding_attention":
                    valid = tirx.all(valid, j > i - c.sliding_window)
                return tirx.if_then_else(
                    valid, x[b, h, i, j], tirx.const(-3.4028234663852886e38, "float32")
                )

            return te.compute(x.shape, value)

        scores = op.tensor_expr_op(mask, "causal_mask", [scores])
        return op.permute_dims(op.matmul(op.softmax(scores, axis=-1), v), (0, 2, 1, 3))


class CPUExperts(Gemma4LargeExperts):
    """Test-only dense execution of selected experts; GPU kernels tested separately."""

    def forward(self, hidden, indices, weights):
        c = self.config
        selected = op.take(self.gate_up_proj.weight, indices, axis=0)
        values = op.matmul(selected, op.reshape(hidden, (hidden.shape[0], 1, c.hidden_size, 1)))
        values = op.reshape(values, (hidden.shape[0], c.top_k_experts, 2 * c.moe_intermediate_size))
        gate, up = op.split(values, 2, axis=-1)
        values = op.gelu(gate, approximate="tanh") * up
        result = op.matmul(
            op.take(self.down_proj.weight, indices, axis=0),
            op.reshape(values, (hidden.shape[0], c.top_k_experts, c.moe_intermediate_size, 1)),
        )
        result = op.reshape(result, (hidden.shape[0], c.top_k_experts, c.hidden_size))
        return op.sum(result * op.reshape(weights, (hidden.shape[0], c.top_k_experts, 1)), axis=1)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("length", [3, 7])
def test_cpu_full_decoder_matches_hf(moe, length):
    torch = pytest.importorskip("torch")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextModel

    torch.set_num_threads(2)
    torch.manual_seed(37)
    c = tiny(moe)
    hc = Gemma4TextConfig(
        **{
            k: v
            for k, v in dataclasses.asdict(c).items()
            if k
            not in [
                "kwargs",
                "context_window_size",
                "prefill_chunk_size",
                "sliding_window_size",
                "max_batch_size",
                "tensor_parallel_shards",
            ]
        }
    )
    hc._attn_implementation = "eager"
    ref = Gemma4TextModel(hc).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.uniform_(0.8, 1.2) if "norm" in name or name.endswith(".scale") else p.uniform_(
                -0.2, 0.2
            )

    class Module(nn.Module):
        def __init__(self):
            self.model = Gemma4LargeForCausalLM(c)
            if moe:
                for layer in self.model.language_model.layers:
                    layer.experts = CPUExperts(c)

        def forward(self, ids):
            hidden = op.reshape(self.model.embed(ids), (1, ids.shape[0], c.hidden_size))
            return self.model.get_logits(self.model.language_model(hidden, DenseCache(c)))

    m = Module()
    spec = nn.spec.ModuleSpec.from_raw(
        {
            "forward": {
                "ids": nn.spec.Tensor(("seq_len",), "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            }
        },
        m,
    )
    mod, params, _ = m.export_tvm(spec=spec, allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    mapping = huggingface(c, QUANTIZATION["q0f32"])
    state = ref.state_dict()
    packed = []
    for name, _ in params:
        key = name.removeprefix("model.")
        values = [
            state[s.removeprefix("model.language_model.")].detach().numpy()
            for s in mapping.param_map[key]
        ]
        packed.append(tvm_tirx.runtime.tensor(mapping.map_func[key](*values)))
    ids = torch.arange(3, 3 + length)[None]
    with torch.no_grad():
        hidden = ref(input_ids=ids, use_cache=False).last_hidden_state
        expected = torch.nn.functional.linear(hidden, ref.embed_tokens.weight).float()
        expected = torch.tanh(expected / c.final_logit_softcapping) * c.final_logit_softcapping
    actual = vm["forward"](tvm_tirx.runtime.tensor(ids[0].numpy().astype("int32")), packed).numpy()
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rms_norm_eps", 0),
        ("rms_norm_eps", float("nan")),
        ("final_logit_softcapping", 0),
        ("final_logit_softcapping", float("inf")),
    ],
)
def test_invalid_numeric_config(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(tiny(), **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"rope_type": "linear"},
        {"rope_theta": 0},
        {"partial_rotary_factor": 0},
        {"partial_rotary_factor": 0.3},
        {"factor": 2},
    ],
)
def test_invalid_rope_config(change):
    c = tiny()
    c.rope_parameters["full_attention"].update(change)
    with pytest.raises(ValueError):
        c.__post_init__()


def test_router_remains_unquantized():
    q = QUANTIZATION["q4f16_1"]
    m, _ = MODELS["gemma4_large_text"].quantize[q.kind](tiny(True), q)
    for layer in m.language_model.layers:
        assert isinstance(layer.router.proj, nn.Linear)
        assert layer.router.proj.weight.dtype == "float16"


def test_fp16_router_matches_hf():
    torch = pytest.importorskip("torch")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextRouter

    c = tiny(True)
    torch.manual_seed(123)
    hc = Gemma4TextConfig(hidden_size=32, num_experts=4, top_k_experts=2)
    reference = Gemma4TextRouter(hc).half().eval()
    with torch.no_grad():
        reference.proj.weight.uniform_(-0.2, 0.2)
        reference.per_expert_scale.uniform_(0.7, 1.3)
    model = Gemma4Router(c)
    model.to("float16")
    spec = nn.spec.ModuleSpec.from_raw(
        {
            "forward": {
                "hidden": nn.spec.Tensor((7, 32), "float16"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            }
        },
        model,
    )
    mod, params, _ = model.export_tvm(spec=spec, allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    state = reference.state_dict()
    packed = [tvm_tirx.runtime.tensor(state[n].numpy()) for n, _ in params]
    x = torch.randn(7, 32).half()
    with torch.no_grad():
        expected = reference(x)
    got = vm["forward"](tvm_tirx.runtime.tensor(x.numpy()), packed)
    np.testing.assert_array_equal(got[2].numpy(), expected[2].numpy())
    np.testing.assert_allclose(got[0].numpy(), expected[0].numpy(), atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(got[1].numpy(), expected[1].numpy(), atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(
    os.environ.get("MLC_GEMMA4_LARGE_NUMERICAL") != "1",
    reason="Opt in to real Metal cache and expert execution",
)
@pytest.mark.parametrize("moe", [False, True])
def test_native_cached_decoder_matches_hf(tmp_path, moe):
    from types import SimpleNamespace

    import torch
    import tvm_ffi
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextModel
    from tvm_ffi import Shape

    from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
    from mlc_llm.support import auto_target

    torch.set_num_threads(2)
    torch.manual_seed(37)
    c = dataclasses.replace(
        tiny(moe), head_dim=256, global_head_dim=512, context_window_size=32, prefill_chunk_size=4
    )
    hc = Gemma4TextConfig(**c.asdict())
    hc._attn_implementation = "eager"
    reference = Gemma4TextModel(hc).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "norm" in name or name.endswith(".scale"):
                parameter.uniform_(0.8, 1.2)
            else:
                parameter.uniform_(-0.05, 0.05)
    target, build = auto_target.detect_target_and_host("metal")
    model = Gemma4LargeForCausalLM(c)
    with target:
        mod, parameters, external = model.export_tvm(
            spec=model.get_default_spec(), allow_extern=True
        )
    metadata = dict(
        params=[dict(name=n, shape=list(p.shape), dtype=p.dtype) for n, p in parameters],
        model_type="gemma4_large_text",
        quantization="q0f32",
        context_window_size=32,
        prefill_chunk_size=4,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        max_batch_size=1,
    )
    library = tmp_path / "model.dylib"
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                ext_mods=external,
                variable_bounds=dict(seq_len=4, batch_size=1, total_seq_len=32),
            ),
        )
    device = tvm_tirx.metal()
    vm = relax.VirtualMachine(tvm_tirx.runtime.load_module(str(library)), device)
    mapping = huggingface(c, QUANTIZATION["q0f32"])
    state = reference.state_dict()
    packed = [
        tvm_tirx.runtime.tensor(
            mapping.map_func[n](
                *[
                    state[s.removeprefix("model.language_model.")].numpy()
                    for s in mapping.param_map[n]
                ]
            ),
            device,
        )
        for n, _ in parameters
    ]
    cache = vm["create_tir_paged_kv_cache"](*[Shape([v]) for v in [1, 32, 4, 16, 0]])
    add = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    begin = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    add(cache, 0)
    ids = np.arange(3, 15, dtype="int32")
    start = 0
    for length in (3, 6, 7, 8, 12):
        current = ids[start:length]
        embedded = vm["embed"](tvm_tirx.runtime.tensor(current, device), packed).numpy()
        begin(cache, Shape([0]), Shape([len(current)]))
        try:
            actual = vm["decode" if len(current) == 1 else "prefill"](
                tvm_tirx.runtime.tensor(embedded[None], device), cache, packed
            )[0].numpy()
        finally:
            end(cache)
        with torch.no_grad():
            hidden = reference(
                input_ids=torch.tensor(ids[:length][None]), use_cache=False
            ).last_hidden_state[:, -1:]
            expected = torch.nn.functional.linear(hidden, reference.embed_tokens.weight).float()
            expected = torch.tanh(expected / 30) * 30
        np.testing.assert_allclose(actual, expected.numpy(), atol=3e-4, rtol=3e-4)
        assert actual.argmax() == expected.numpy().argmax()
        start = length
