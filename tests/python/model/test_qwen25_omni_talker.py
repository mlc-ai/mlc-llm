"""Talker checkpoint mapping and compiled embedding/projection parity."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn

from mlc_llm.model.qwen2_5_omni.talker_loader import huggingface
from mlc_llm.model.qwen2_5_omni.talker_model import Qwen25OmniTalker, Qwen25OmniTalkerConfig
from mlc_llm.quantization import QUANTIZATION


def tiny_config(**updates):
    values = dict(
        hidden_act="silu",
        hidden_size=128,
        head_dim=64,
        embedding_size=192,
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
    values.update(updates)
    return values


@pytest.fixture(scope="module")
def models():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniTalkerConfig.from_dict({"talker_config": tiny_config()})
    hf_config = hf.Qwen2_5OmniTalkerConfig(
        **tiny_config(),
        pad_token_id=0,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000,
            "mrope_section": [16, 16, 0],
        },
    )
    torch.manual_seed(83)
    reference = hf.Qwen2_5OmniTalkerForConditionalGeneration(hf_config).eval()
    state = {
        f"talker.{name}": value.detach().numpy() for name, value in reference.state_dict().items()
    }
    model = Qwen25OmniTalker(config)
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = huggingface(config, QUANTIZATION["q0f32"])
    mapped = {}
    used = set()
    for name, param in params:
        sources = mapping.param_map[name]
        used.update(sources)
        mapped[name] = mapping.map_func[name](*[state[key] for key in sources])
        assert mapped[name].shape == tuple(param.shape)
    assert used == set(state)
    subset = nn.spec.ModuleSpec.from_raw(
        {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "project": {
                "embeddings": nn.spec.Tensor(["seq_len", 192], "float32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        },
        model,
    )
    mod, subset_params, _ = model.export_tvm(spec=subset, allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    args = [tvm_tirx.runtime.tensor(mapped[name]) for name, _ in subset_params]
    return torch, reference, vm, args, config, model, mapped


@pytest.mark.parametrize("length", [1, 2, 7])
def test_codec_embedding_and_projection(models, length):
    torch, reference, vm, params, *_ = models
    ids = np.arange(length, dtype="int32")
    thinker = np.random.default_rng(length).normal(size=(length, 192)).astype("float32")
    with torch.no_grad():
        expected_embed = reference.model.embed_tokens(torch.from_numpy(ids).long()).numpy()
        expected_project = reference.thinker_to_talker_proj(
            torch.from_numpy(thinker + expected_embed)
        ).numpy()
    actual_embed = vm["embed"](tvm_tirx.runtime.tensor(ids), params).numpy()
    np.testing.assert_array_equal(actual_embed, expected_embed)
    actual_project = vm["project"](tvm_tirx.runtime.tensor(thinker + actual_embed), params).numpy()
    np.testing.assert_allclose(actual_project, expected_project, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "updates",
    [
        {"embedding_size": 0},
        {"embedding_size": True},
        {"tensor_parallel_shards": 2},
        {"input_modalities": ["image"]},
        {"tie_word_embeddings": True},
        {"rope_scaling": {"rope_type": "yarn"}},
        {"use_sliding_window": True},
    ],
)
def test_reject_unsupported_config(updates):
    with pytest.raises(ValueError):
        Qwen25OmniTalkerConfig.from_dict(tiny_config(**updates))


def test_nested_config_overrides_and_roundtrip():
    source = {"talker_config": tiny_config(), "prefill_chunk_size": 8}
    config = Qwen25OmniTalkerConfig.from_dict(source)
    assert config.embedding_size == 192 and config.prefill_chunk_size == 8
    assert source["talker_config"]["prefill_chunk_size"] == 16
    assert Qwen25OmniTalkerConfig.from_dict(config.asdict()) == config


@pytest.mark.skipif(os.getenv("MLC_OMNI_TALKER_NUMERICAL") != "1", reason="Opt in to Metal")
def test_cached_prefill_decode_reset(models, tmp_path):
    import tvm_ffi
    from tvm_ffi import Shape

    from mlc_llm.compiler_pass import pipeline as _pipeline_registration  # noqa: F401
    from mlc_llm.support import auto_target

    torch, reference, _, _, config, model, mapped = models
    mod, params, external = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    metadata = dict(
        params=[{"name": n, "shape": list(p.shape), "dtype": p.dtype} for n, p in params],
        context_window_size=32,
        prefill_chunk_size=16,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        kv_state_kind="kv_cache",
        model_type="qwen25_omni_talker",
    )
    target, build = auto_target.detect_target_and_host("metal")
    library = tmp_path / "talker.dylib"
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
    packed = [tvm_tirx.runtime.tensor(mapped[n], device=device) for n, _ in params]
    add = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    begin = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    cache = vm["create_tir_paged_kv_cache"](
        Shape([1]), Shape([32]), Shape([16]), Shape([16]), Shape([0])
    )
    rng = np.random.default_rng(97)
    initial = torch.tensor(rng.normal(size=(1, 4, 192)).astype("float32"))
    # Explicit host alignment: the final two Thinker states receive codec pad/BOS.
    with torch.no_grad():
        initial[:, -2] += reference.model.embed_tokens(torch.tensor([1]))
        initial[:, -1] += reference.model.embed_tokens(torch.tensor([2]))

    def execute(embeddings, method):
        flat = embeddings.reshape(-1, 192).numpy()
        projected = vm["project"](tvm_tirx.runtime.tensor(flat, device=device), packed)
        shaped = tvm_tirx.runtime.tensor(
            projected.numpy().reshape(1, flat.shape[0], 128), device=device
        )
        begin(cache, Shape([0]), Shape([flat.shape[0]]))
        result = vm[method](shaped, cache, packed)[0].numpy()[0, -1]
        end(cache)
        return result

    for replay in range(2):
        if replay:
            vm["reset_state"](cache)
        add(cache, 0)
        with torch.no_grad():
            expected = reference(inputs_embeds=initial.clone(), use_cache=True)
        actual = execute(initial, "prefill")
        np.testing.assert_allclose(actual, expected.logits[0, -1].numpy(), atol=3e-5, rtol=3e-5)
        for step in range(2):
            token = int(expected.logits[0, -1].argmax())
            assert int(actual.argmax()) == token
            reply = torch.tensor(rng.normal(size=(1, 1, 192)).astype("float32"))
            with torch.no_grad():
                aligned = reference.model.embed_tokens(torch.tensor([[token]])) + reply
                expected = reference(
                    input_ids=torch.tensor([[token]]),
                    thinker_reply_part=reply,
                    past_key_values=expected.past_key_values,
                    use_cache=True,
                )
            actual = execute(aligned, "decode")
            np.testing.assert_allclose(actual, expected.logits[0, -1].numpy(), atol=3e-5, rtol=3e-5)
