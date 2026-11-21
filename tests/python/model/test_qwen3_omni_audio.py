"""Check the standalone Qwen3 audio schema against the HF model."""

import dataclasses

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.qwen2_5_omni.audio_model import audio_positions
from mlc_llm.model.qwen3_omni.audio_model import (
    Qwen3OmniAudioConfig,
    Qwen3OmniAudioEncoder,
    stem_length,
)


def test_audio_schema():
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    config = Qwen3OmniAudioConfig(
        num_mel_bins=8,
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=1,
        output_dim=8,
        downsample_hidden_size=4,
    )
    reference = hf.Qwen3OmniMoeAudioEncoder(hf.Qwen3OmniMoeAudioEncoderConfig(**vars(config)))
    model = Qwen3OmniAudioEncoder(config)
    _, parameters, _ = model.export_tvm(model.get_default_spec(), allow_extern=True)
    actual = {name: list(value.shape) for name, value in parameters if name != "positions"}
    expected = {name: list(value.shape) for name, value in reference.state_dict().items()}
    assert actual == expected
    assert dict(parameters)["positions"].shape == [13, 16]


@pytest.mark.parametrize("frames, expected", [(1, 1), (8, 1), (9, 2), (99, 13), (100, 13)])
def test_cnn_length(frames, expected):
    assert stem_length(frames) == expected


@pytest.mark.parametrize(
    "updates", [{"n_window": 51}, {"n_window_infer": 1600}, {"downsample_hidden_size": 0}]
)
def test_unsupported_audio_layout(updates):
    with pytest.raises(ValueError):
        dataclasses.replace(Qwen3OmniAudioConfig(), **updates)


@pytest.fixture(scope="module")
def compiled_audio():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    config = Qwen3OmniAudioConfig(
        num_mel_bins=16,
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=1,
        output_dim=12,
        downsample_hidden_size=4,
    )
    hf_config = hf.Qwen3OmniMoeAudioEncoderConfig(**dataclasses.asdict(config))
    hf_config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(55)
        reference = hf.Qwen3OmniMoeAudioEncoder(hf_config).eval()
    model = Qwen3OmniAudioEncoder(config)
    mod, named, _ = model.export_tvm(model.get_default_spec(), allow_extern=True)
    state = {name: value.detach().numpy() for name, value in reference.state_dict().items()}
    state["positions"] = audio_positions(config)[: stem_length(2 * config.n_window)]
    params = [tvm_tirx.runtime.tensor(state[name]) for name, _ in named]
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield torch, reference, config, vm, params
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("frames", [101, 801])
def test_compiled_audio_boundaries(compiled_audio, frames):
    torch, reference, config, vm, params = compiled_audio
    features = np.random.default_rng(frames).normal(size=(16, frames)).astype("float32")
    with torch.inference_mode():
        expected = reference(
            torch.from_numpy(features), feature_lens=torch.tensor([frames])
        ).last_hidden_state.numpy()

    # Apply the host chunk/window contract; all encoder math runs in compiled TVM.
    chunk = 2 * config.n_window
    states = []
    for start in range(0, frames, chunk):
        valid = min(chunk, frames - start)
        padded = np.zeros((1, 1, config.num_mel_bins, chunk), dtype="float32")
        padded[0, 0, :, :valid] = features[:, start : start + valid]
        result = vm["stem"](tvm_tirx.runtime.tensor(padded), params).numpy()
        states.append(result[: stem_length(valid)])
    states = np.concatenate(states)
    window = stem_length(chunk) * (config.n_window_infer // chunk)
    outputs = []
    for start in range(0, len(states), window):
        hidden = states[start : start + window]
        valid = tvm_tirx.runtime.tensor(np.array([len(hidden) * 2], dtype="int32"))
        outputs.append(vm["encode_window"](tvm_tirx.runtime.tensor(hidden), valid, params).numpy())
    actual = np.concatenate(outputs)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
