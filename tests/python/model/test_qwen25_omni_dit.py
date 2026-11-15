"""Compiled DiT flow prediction against independent HF layers and ODE solver."""

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.qwen2_5_omni.dit_loader import huggingface
from mlc_llm.model.qwen2_5_omni.dit_model import Qwen25OmniDiT, Qwen25OmniDiTConfig
from mlc_llm.model.qwen2_5_omni.dit_sampling import sample_flow


@pytest.fixture(scope="module")
def models():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    c = Qwen25OmniDiTConfig(
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=8,
        emb_dim=8,
        mel_dim=4,
        enc_dim=4,
        enc_emb_dim=4,
        num_embeds=17,
        block_size=4,
        look_ahead_layers=(1,),
        look_backward_layers=(0, 2),
    )
    torch.manual_seed(31)
    hc = hf.Qwen2_5OmniDiTConfig(
        **{k: v for k, v in vars(c).items() if k != "rope_theta"},
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
        enc_channels=(8, 8, 8, 8, 24),
        enc_attention_channels=4,
        enc_res2net_scale=2,
        enc_se_channels=4,
    )
    hc._attn_implementation = "sdpa"
    reference = hf.Qwen2_5OmniToken2WavDiTModel(hc).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.15, 0.15)
    model = Qwen25OmniDiT(c)
    mod, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    state = reference.state_dict()
    state["rotary_embed.inv_freq"] = reference.rotary_embed.inv_freq
    mapping = huggingface(c)
    assert set(mapping.param_map) == {name for name, _ in named}
    prefix = "token2wav.code2wav_dit_model."
    expected_names = {
        prefix + name
        for name in state
        if not name.startswith("input_embed.spk_encoder.") and name != "rotary_embed.inv_freq"
    }
    assert {name for names in mapping.param_map.values() for name in names} == expected_names
    params = [
        tvm_tirx.runtime.tensor(
            mapping.map_func[name](
                *[state[key[len(prefix) :]].detach().numpy() for key in mapping.param_map[name]]
            )
        )
        for name, _ in named
    ]

    def actual(*args):
        return vm["forward"](*[tvm_tirx.runtime.tensor(x) for x in args], params).numpy()

    def expected(noise, time, code, ref, speaker):
        # Exact HF flow stages after its explicitly separated ECAPA conditioner.
        with torch.no_grad():
            x, t, ids, r, s = [torch.from_numpy(x) for x in (noise, time, code, ref, speaker)]
            text = reference.text_embed(ids[None])
            h = reference.input_embed.proj(
                torch.cat(
                    (
                        x[None],
                        r[None, None].expand(1, len(x), -1),
                        text,
                        s[None, None].expand(1, len(x), -1),
                    ),
                    dim=-1,
                )
            )
            emb = reference.time_embed(t)
            pos = reference.rotary_embed(h, torch.arange(len(x))[None])
            diff = reference._create_block_diff(h)
            for block in reference.transformer_blocks:
                h = block(h, emb, position_embeddings=pos, block_diff=diff)
            return reference.proj_out(reference.norm_out(h, emb))[0].numpy()

    return c, actual, expected, hf, torch, reference


@pytest.mark.parametrize("codes", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("time", [0.0, 0.37, 1.0])
def test_forward(models, codes, time):
    c, actual, expected, _, _, _ = models
    rng = np.random.default_rng(codes)
    args = (
        rng.normal(size=(codes * c.repeats, c.mel_dim)).astype("float32"),
        np.array([time], dtype="float32"),
        rng.integers(0, c.num_embeds + 1, size=codes, dtype="int32"),
        rng.normal(size=c.enc_dim).astype("float32"),
        rng.normal(size=c.enc_emb_dim).astype("float32"),
    )
    np.testing.assert_allclose(actual(*args), expected(*args), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("guidance", [0.0, 0.5])
def test_pinned_solver(models, guidance):
    c, actual, expected, hf, torch, _ = models
    rng = np.random.default_rng(33)
    noise = rng.normal(size=(6, c.mel_dim)).astype("float32")
    code = np.array([1, 2, 3], dtype="int32")
    reference = rng.normal(size=c.enc_dim).astype("float32")
    null = rng.normal(size=c.enc_dim).astype("float32")
    speaker = rng.normal(size=c.enc_emb_dim).astype("float32")
    times = np.array([0, 0.12, 0.7, 1], dtype="float32")

    def fn(time, state):
        p = expected(state.numpy(), time.reshape(1).numpy(), code, reference, speaker)
        if guidance:
            p = (
                p
                + (
                    p
                    - expected(
                        state.numpy(),
                        time.reshape(1).numpy(),
                        np.zeros_like(code),
                        null,
                        np.zeros_like(speaker),
                    )
                )
                * guidance
            )
        return torch.from_numpy(p)

    want = (
        hf.RungeKutta4ODESolver(fn, torch.from_numpy(noise))
        .integrate(torch.from_numpy(times))[-1]
        .numpy()
    )
    got = sample_flow(actual, noise, code, reference, null, speaker, times, guidance)
    np.testing.assert_allclose(got, want, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("null_condition", [False, True])
def test_full_hf_forward_with_external_conditioner(models, null_condition):
    c, actual, _, _, torch, reference = models
    rng = np.random.default_rng(39)
    noise = rng.normal(size=(10, c.mel_dim)).astype("float32")
    code = rng.integers(0, c.num_embeds + 1, size=5, dtype="int32")
    speaker = rng.normal(size=c.enc_emb_dim).astype("float32")
    mel = rng.normal(size=(1, 32, c.mel_dim)).astype("float32")
    if null_condition:
        code[:] = 0
        speaker[:] = 0
        mel[:] = 0
    time = np.array([0.42], dtype="float32")
    with torch.no_grad():
        condition = reference.input_embed.spk_encoder(torch.from_numpy(mel))[0].numpy()
        expected = reference(
            hidden_states=torch.from_numpy(noise[None]),
            condition_vector=torch.from_numpy(mel),
            speaker_embedding=torch.from_numpy(speaker)[None, None].expand(1, len(noise), -1),
            quantized_code=torch.from_numpy(code[None]),
            time_step=torch.from_numpy(time),
            apply_cfg=False,
        )[0].numpy()
    np.testing.assert_allclose(
        actual(noise, time, code, condition, speaker), expected, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize(
    "kind", ["code_count", "negative_code", "time_order", "nonfinite", "null_shape"]
)
def test_solver_rejects_invalid_inputs(kind):
    noise = np.zeros((4, 2), dtype="float32")
    code = np.ones(2, dtype="int32")
    ref = np.zeros(3, dtype="float32")
    null = ref.copy()
    speaker = ref.copy()
    times = np.array([0, 1], dtype="float32")
    if kind == "code_count":
        code = code[:1]
    if kind == "negative_code":
        code[0] = -1
    if kind == "time_order":
        times = times[::-1]
    if kind == "nonfinite":
        noise[0, 0] = np.nan
    if kind == "null_shape":
        null = null[:1]

    def never(*args):
        raise AssertionError("Invalid inputs reached predictor")

    with pytest.raises(ValueError):
        sample_flow(never, noise, code, ref, null, speaker, times)


@pytest.mark.parametrize("bad_call", [0, 1])
@pytest.mark.parametrize("kind", ["broadcast", "dtype", "nonfinite"])
def test_solver_validates_both_velocities_before_guidance(bad_call, kind):
    noise = np.zeros((4, 2), dtype="float32")
    code = np.ones(2, dtype="int32")
    ref = np.zeros(3, dtype="float32")
    times = np.array([0, 1], dtype="float32")
    calls = 0

    def predictor(*args):
        nonlocal calls
        value = noise.copy()
        if calls == bad_call:
            if kind == "broadcast":
                value = value[:1]
            elif kind == "dtype":
                value = value.astype("float64")
            else:
                value[0, 0] = np.nan
        calls += 1
        return value

    with pytest.raises(ValueError, match="velocity"):
        sample_flow(predictor, noise, code, ref, ref, ref, times)
    assert calls == bad_call + 1
