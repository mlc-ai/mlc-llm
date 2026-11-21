"""Compiled audio tower parity against the independent Transformers implementation."""

import numpy as np
import pytest
import tvm_tirx
from tvm import relax

from mlc_llm.model.qwen2_5_omni.audio_model import (
    Qwen25OmniAudioConfig,
    Qwen25OmniAudioEncoder,
    audio_positions,
)


@pytest.fixture(scope="module")
def audio_models():
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    config = Qwen25OmniAudioConfig(
        num_mel_bins=4,
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=2,
        output_dim=8,
    )
    torch.manual_seed(17)
    hf_config = reference.Qwen2_5OmniAudioEncoderConfig(**vars(config))
    hf_config._attn_implementation = "eager"
    hf_model = reference.Qwen2_5OmniAudioEncoder(hf_config).eval()
    model = Qwen25OmniAudioEncoder(config)
    mod, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    state = hf_model.state_dict()
    parameters = [
        tvm_tirx.runtime.tensor(
            audio_positions(config) if name == "positions" else state[name].detach().numpy()
        )
        for name, _ in named
    ]
    return torch, hf_model, vm, parameters, config


@pytest.mark.parametrize(
    "lengths", [[1], [2], [3], [199], [200], [201], [399], [400], [401], [199, 201]]
)
def test_encoder_parity(audio_models, lengths):
    torch, hf_model, vm, parameters, config = audio_models
    features = (
        np.random.default_rng(sum(lengths))
        .normal(size=(config.num_mel_bins, sum(lengths)))
        .astype("float32")
    )
    with torch.no_grad():
        expected = hf_model(
            torch.from_numpy(features), feature_lens=torch.tensor(lengths)
        ).last_hidden_state.numpy()
    outputs = []
    offset = 0
    for length in lengths:
        chunks = []
        for start in range(0, length, 2 * config.n_window):
            end = min(start + 2 * config.n_window, length)
            chunk = np.zeros((1, config.num_mel_bins, 2 * config.n_window), dtype="float32")
            chunk[0, :, : end - start] = features[:, offset + start : offset + end]
            valid = tvm_tirx.runtime.tensor(np.array([end - start], dtype="int32"))
            chunks.append(
                vm["encode_chunk"](tvm_tirx.runtime.tensor(chunk), valid, parameters).numpy()[
                    : (end - start + 1) // 2
                ]
            )
        hidden = np.concatenate(chunks)
        if len(hidden) >= 2:
            pairs = hidden[: len(hidden) // 2 * 2].reshape(-1, 2, config.d_model)
            outputs.append(vm["project"](tvm_tirx.runtime.tensor(pairs), parameters).numpy())
        offset += length
    actual = (
        np.concatenate(outputs) if outputs else np.empty((0, config.output_dim), dtype="float32")
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "changes",
    [
        {"d_model": 2},
        {"d_model": 15},
        {"encoder_attention_heads": 3},
        {"n_window": 1501},
        {"activation_function": "relu"},
        {"encoder_layers": 0},
    ],
)
def test_reject_invalid_audio_config(changes):
    with pytest.raises(ValueError):
        Qwen25OmniAudioConfig(**changes)


def test_pcm_frontend_matches_whisper():
    from transformers import WhisperFeatureExtractor

    from mlc_llm.model.qwen2_5_omni.audio_processing import (
        Qwen25OmniAudioFeatures,
        whisper_frontend_parameters,
    )

    reference = WhisperFeatureExtractor(feature_size=128)
    model = Qwen25OmniAudioFeatures(num_samples=3200)
    mod, named, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    values = whisper_frontend_parameters(reference.mel_filters)
    params = [tvm_tirx.runtime.tensor(values[name]) for name, _ in named]
    samples = np.random.default_rng(42).normal(size=3200).astype("float32") * 0.05
    for waveform in [samples, np.zeros(3200, dtype="float32"), samples * 1e-5]:
        expected = reference._torch_extract_fbank_features(waveform[None])[0]
        actual = vm["forward"](tvm_tirx.runtime.tensor(waveform), params).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
