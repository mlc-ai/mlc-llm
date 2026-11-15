"""Text/audio Thinker configuration, loader and optional real Metal parity."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlc_llm.model.qwen2_5_omni.thinker_loader import huggingface
from mlc_llm.model.qwen2_5_omni.thinker_model import Qwen25OmniThinker, Qwen25OmniThinkerConfig
from mlc_llm.quantization import QUANTIZATION


def tiny_config(**changes):
    values = dict(
        hidden_act="silu",
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        vocab_size=64,
        context_window_size=32,
        prefill_chunk_size=16,
    )
    values.update(changes)
    return values


def test_nested_thinker_config_and_overrides():
    values = tiny_config()
    source = {"thinker_config": {"text_config": values}, "prefill_chunk_size": 8}
    config = Qwen25OmniThinkerConfig.from_dict(source)
    assert config.hidden_size == 128 and config.prefill_chunk_size == 8
    assert config.input_modalities == ("text", "audio")
    assert source["thinker_config"]["text_config"]["prefill_chunk_size"] == 16
    assert Qwen25OmniThinkerConfig.from_dict(config.asdict()) == config


@pytest.mark.parametrize(
    "change",
    [
        {"input_modalities": ["text", "image"]},
        {"input_modalities": ["text", "audio", "video"]},
        {"rope_scaling": {"rope_type": "yarn"}},
        {"rope_parameters": {"rope_type": "linear", "factor": 2}},
        {"rope_scaling": {"partial_rotary_factor": 0.5}},
        {"use_sliding_window": True},
        {"layer_types": ["full_attention", "sliding_attention"]},
        {"attention_bias": False},
        {"tie_word_embeddings": True},
    ],
)
def test_reject_unsupported_position_or_attention_semantics(change):
    with pytest.raises(ValueError):
        Qwen25OmniThinkerConfig.from_dict(tiny_config(**change))


@pytest.fixture(scope="module")
def thinker_fixture():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniThinkerConfig.from_dict(tiny_config())
    hf_config = hf.Qwen2_5OmniTextConfig(
        **{k: v for k, v in tiny_config().items() if k not in ("rope_theta",)},
        max_position_embeddings=32,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000,
            "mrope_section": [8, 12, 12],
        },
    )
    hf_config._attn_implementation = "eager"
    torch.manual_seed(23)
    reference = hf.Qwen2_5OmniThinkerTextModel(hf_config).eval()
    head = torch.nn.Linear(128, 64, bias=False).eval()
    state = {
        f"thinker.model.{name}": value.detach().numpy()
        for name, value in reference.state_dict().items()
    }
    state["thinker.lm_head.weight"] = head.weight.detach().numpy()
    model = Qwen25OmniThinker(config)
    mod, params, external = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = huggingface(config, QUANTIZATION["q0f32"])
    values = {}
    for name, param in params:
        sources = mapping.param_map[name]
        assert all(source.startswith("thinker.") for source in sources)
        values[name] = mapping.map_func[name](*[state[source] for source in sources])
        assert values[name].shape == tuple(param.shape)
    assert set(values) == set(mapping.param_map)
    return torch, reference, head, config, mod, params, external, values


def test_exact_thinker_mapping(thinker_fixture):
    _, _, _, _, _, params, _, values = thinker_fixture
    assert len(params) == len(values) and len(params) > 0


@pytest.mark.skipif(
    os.getenv("MLC_OMNI_THINKER_NUMERICAL") != "1", reason="Opt in to isolated Metal execution"
)
def test_compiled_text_audio_prefill_decode(thinker_fixture, tmp_path):
    import tvm_ffi
    import tvm_tirx
    from tvm import relax
    from tvm_ffi import Shape

    from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
    from mlc_llm.support import auto_target

    torch, reference, head, config, mod, params, external, values = thinker_fixture
    schema = [{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params]
    metadata = dict(
        params=schema,
        context_window_size=32,
        prefill_chunk_size=16,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        model_type="qwen2_5_omni_thinker",
    )
    target, build = auto_target.detect_target_and_host("metal")
    library = tmp_path / "thinker.dylib"
    with target, tvm_tirx.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        build(
            mod,
            SimpleNamespace(target=target, output=library, system_lib_prefix=""),
            pipeline=relax.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                ext_mods=external,
                variable_bounds={"seq_len": 16, "batch_size": 1, "total_seq_len": 32},
            ),
        )
    device = tvm_tirx.metal()
    vm = relax.VirtualMachine(tvm_tirx.runtime.load_module(str(library)), device)
    packed = [tvm_tirx.runtime.tensor(values[n], device=device) for n, _ in params]
    add = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    begin = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    for audio in (False, True):
        ids = torch.tensor([[3, 5, 7, 9, 11]])
        with torch.no_grad():
            embeds = reference.embed_tokens(ids)
            if audio:
                embeds[:, 1:3] = torch.linspace(-0.2, 0.2, 256).reshape(1, 2, 128)
            expected = head(
                reference(inputs_embeds=embeds, use_cache=False).last_hidden_state
            ).numpy()
        cache = vm["create_tir_paged_kv_cache"](
            Shape([1]), Shape([32]), Shape([16]), Shape([16]), Shape([0])
        )
        add(cache, 0)
        begin(cache, Shape([0]), Shape([4]))
        prefill = vm["prefill"](
            tvm_tirx.runtime.tensor(embeds[:, :4].numpy(), device=device), cache, packed
        )[0].numpy()
        end(cache)
        begin(cache, Shape([0]), Shape([1]))
        decode = vm["decode"](
            tvm_tirx.runtime.tensor(embeds[:, 4:].numpy(), device=device), cache, packed
        )[0].numpy()
        end(cache)
        np.testing.assert_allclose(prefill[0, -1], expected[0, 3], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(decode[0, -1], expected[0, 4], atol=2e-5, rtol=2e-5)

        # Reset must clear sequence membership and positions, not merely return an alias.
        vm["reset_state"](cache)
        add(cache, 0)
        begin(cache, Shape([0]), Shape([5]))
        replay = vm["prefill"](
            tvm_tirx.runtime.tensor(embeds.numpy(), device=device), cache, packed
        )[0].numpy()
        end(cache)
        np.testing.assert_allclose(replay[0, -1], expected[0, 4], atol=2e-5, rtol=2e-5)


def test_projection_precision_is_scoped_to_thinker():
    from mlc_llm.model.qwen2.qwen2_model import QWen2Config, QWen2LMHeadModel

    base_config = QWen2Config.from_dict(tiny_config())
    thinker_config = Qwen25OmniThinkerConfig.from_dict(tiny_config())
    base = QWen2LMHeadModel(base_config)
    thinker = Qwen25OmniThinker(thinker_config)
    base.to("float16")
    thinker.to("float16")
    assert "projection_out_dtype" not in base_config.asdict()
    assert "projection_out_dtype" not in thinker_config.asdict()
    for plain, precise in zip(base.model.layers, thinker.model.layers):
        assert plain.self_attn.c_attn.bias.dtype == "float16"
        assert precise.self_attn.c_attn.bias.dtype == "float32"
        for old, new in (
            (plain.self_attn.c_attn, precise.self_attn.c_attn),
            (plain.self_attn.o_proj, precise.self_attn.o_proj),
            (plain.mlp.gate_up_proj, precise.mlp.gate_up_proj),
            (plain.mlp.down_proj, precise.mlp.down_proj),
        ):
            assert old.out_dtype is None
            assert new.out_dtype == "float32"
            assert old.weight.dtype == new.weight.dtype == "float16"


@pytest.mark.parametrize("quantization", ["q0f16", "q4f16_1"])
def test_mixed_precision_exports_after_quantization(quantization):
    from mlc_llm.quantization.model_quantization import make_quantization_functions

    config = Qwen25OmniThinkerConfig.from_dict(tiny_config())
    quant = QUANTIZATION[quantization]
    model, _ = make_quantization_functions(Qwen25OmniThinker)[quant.kind](config, quant)
    mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    assert dict(params)["model.layers.0.self_attn.c_attn.bias"].dtype == "float32"
    assert mod["prefill"].ret_ty.fields[0].dtype == "float32"


def test_fp16_projection_accumulates_before_rounding():
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend import nn

    from mlc_llm.model.qwen2.qwen2_model import QWen2Attention

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = QWen2Attention(
                Qwen25OmniThinkerConfig.from_dict(tiny_config())
            ).c_attn
            self.to("float16")

        def forward(self, x):
            return self.projection(x).astype(x.dtype)

    model = Projection()
    projection = model.projection
    projection.weight.data = np.ones(projection.weight.shape, dtype="float16")
    projection.bias.data = np.full(projection.bias.shape, 0.25, dtype="float32")
    # Half accumulation loses unit terms next to 2048; float32 must retain them.
    inputs = np.tile(np.array([2048, 1, -2048, 1], dtype="float16"), 32)[None]
    compiled = model.jit(spec={"forward": {"x": nn.spec.Tensor((1, 128), "float16")}})
    actual = compiled["forward"](torch.tensor(inputs))
    np.testing.assert_array_equal(actual, np.full((1, 256), 64.25, dtype="float16"))
