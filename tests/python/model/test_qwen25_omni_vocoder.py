"""Waveform decoder parity against the pinned Transformers BigVGAN implementation."""

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.qwen2_5_omni.vocoder_loader import huggingface
from mlc_llm.model.qwen2_5_omni.vocoder_model import (
    Qwen25OmniVocoder,
    Qwen25OmniVocoderConfig,
    vocoder_filter,
)


@pytest.fixture(scope="module", params=[((2, 2), (4, 4)), ((5, 3, 2), (11, 7, 4))])
def vocoder_models(request):
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniVocoderConfig(
        mel_dim=4,
        upsample_initial_channel=16,
        upsample_rates=request.param[0],
        upsample_kernel_sizes=request.param[1],
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    torch.manual_seed(29)
    hf_model = reference.Qwen2_5OmniToken2WavBigVGANModel(
        reference.Qwen2_5OmniBigVGANConfig(**vars(config))
    ).eval()
    with torch.no_grad():
        for name, value in hf_model.named_parameters():
            value.uniform_(-0.15, 0.15)
    model = Qwen25OmniVocoder(config)
    mod, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    state = {
        "token2wav.code2wav_bigvgan_model." + name: value.detach().numpy()
        for name, value in hf_model.state_dict().items()
    }
    mapping = huggingface(config)
    assert {name for names in mapping.param_map.values() for name in names} == set(state)
    assert set(mapping.param_map) == {name for name, _ in named}
    params = [
        tvm_tirx.runtime.tensor(
            mapping.map_func[name](*[state[key] for key in mapping.param_map[name]])
        )
        for name, _ in named
    ]
    return torch, hf_model, vm, params, config


@pytest.mark.parametrize("frames", [1, 2, 3, 8, 17])
def test_compiled_waveform_parity(vocoder_models, frames):
    torch, reference, vm, params, config = vocoder_models
    mel = np.random.default_rng(frames).uniform(-12, 2, (config.mel_dim, frames)).astype("float32")
    with torch.no_grad():
        expected = reference(torch.from_numpy(mel[None])).numpy()
    actual = vm["forward"](tvm_tirx.runtime.tensor(mel), params).numpy()
    assert actual.shape == (frames * config.samples_per_frame,)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_mel_floor_and_clipping(vocoder_models):
    torch, reference, vm, params, config = vocoder_models
    for level in [-30.0, 0.0, 30.0]:
        mel = np.full((config.mel_dim, 3), level, "float32")
        with torch.no_grad():
            expected = reference(torch.from_numpy(mel[None])).numpy()
        actual = vm["forward"](tvm_tirx.runtime.tensor(mel), params).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_filter_matches_reference():
    reference = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    np.testing.assert_allclose(
        vocoder_filter(),
        reference.kaiser_sinc_filter1d(0.25, 0.3, 12).numpy(),
        atol=2e-8,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"mel_dim": 0},
        {"upsample_initial_channel": 7},
        {"upsample_rates": ()},
        {"upsample_kernel_sizes": (12, 7, 4, 4, 4, 4)},
        {"resblock_dilation_sizes": ((1, 3),)},
        {"use_bias_at_final": True},
    ],
)
def test_reject_unsupported_vocoder_config(changes):
    with pytest.raises(ValueError):
        Qwen25OmniVocoderConfig(**changes)


def test_float32_only():
    with pytest.raises(ValueError, match="float32"):
        Qwen25OmniVocoder(Qwen25OmniVocoderConfig()).to("float16")
