"""The speech builder exposes aligned host ports without changing the default ABI."""

import json
import runpy
import sys
from pathlib import Path

import pytest


@pytest.fixture
def source(tmp_path):
    # Reduced layer/FFN counts make schema tests cheap; no weights are synthesized.
    config = dict(
        hidden_act="silu",
        hidden_size=2048,
        intermediate_size=256,
        num_attention_heads=16,
        num_hidden_layers=1,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        vocab_size=151936,
        context_window_size=32768,
        prefill_chunk_size=256,
    )
    path = tmp_path / "source"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            dict(
                thinker_config=dict(text_config=config),
                talker_config=dict(tts_text_end_token_id=151861, tts_text_pad_token_id=151859),
            )
        )
    )
    return path


def build(monkeypatch, source, out, *options):
    script = Path(__file__).resolve().parents[3] / "examples/python/omni_audio/build_thinker.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--source", str(source), "--out", str(out), "--export-only", *options],
    )
    runpy.run_path(str(script), run_name="__main__")
    return json.loads((out / "metadata.json").read_text())


def test_speech_and_default_contracts(monkeypatch, source, tmp_path):
    plain = build(monkeypatch, source, tmp_path / "plain", "--quantization", "q0f32")
    speech = build(monkeypatch, source, tmp_path / "speech", "--quantization", "q0f32", "--talker")
    assert plain["params"] == speech["params"]
    p = plain["artifact"]["components"]["thinker"]
    s = speech["artifact"]["components"]["thinker"]
    assert p["spec"]["contract"] == "qwen2_5_omni.thinker_audio.v1"
    assert set(p["spec"]["outputs"]) == {"token_ids"}
    assert p["exports"]["prefill"] == "prefill"
    assert s["spec"]["contract"] == "qwen2_5_omni.thinker_audio_talker.v1"
    assert set(s["spec"]["outputs"]) == {
        "token_ids",
        "prefix_embeddings",
        "reply_embeddings",
        "finish_reason",
    }
    assert s["exports"]["prefill"] == "prefill_with_hidden"
    assert s["exports"]["decode"] == "decode_with_hidden"
    assert s["spec"]["outputs"]["prefix_embeddings"]["shape"] == ["prefix", 2048]
    assert plain["thinker"]["active_vocab_size"] == 151665
    assert speech["thinker"]["active_vocab_size"] == 151936
    assert speech["context_window_size"] == 32768
    assert speech["thinker"]["speech_profile"] == dict(
        speaker_bos_token_ids=[151870, 151872],
        text_eos_token_id=151861,
        text_pad_token_id=151859,
    )
    assert not list((tmp_path / "speech").glob("*.wasm"))


@pytest.mark.parametrize("device, suffix", [("webgpu", "wasm"), ("metal", "dylib"), ("cuda", "so")])
def test_builder_library_name_matches_manifest(monkeypatch, source, tmp_path, device, suffix):
    from mlc_llm.support import constants

    monkeypatch.setattr(constants, "MLC_DSO_SUFFIX", suffix)
    metadata = build(
        monkeypatch, source, tmp_path / device, "--device", device, "--quantization", "q0f32"
    )
    assert metadata["artifact"]["components"]["thinker"]["module"] == f"thinker.{suffix}"


@pytest.mark.parametrize(
    "options", [("--talker",), ("--talker", "--tiny", "--quantization", "q0f32")]
)
def test_reject_unverified_speech_profiles(monkeypatch, source, tmp_path, options):
    with pytest.raises(SystemExit, match="2"):
        build(monkeypatch, source, tmp_path / "invalid", *options)


def test_reject_speech_context_mismatch(monkeypatch, source, tmp_path):
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["thinker_config"]["text_config"]["context_window_size"] = 4096
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="2"):
        build(monkeypatch, source, tmp_path / "invalid", "--quantization", "q0f32", "--talker")
