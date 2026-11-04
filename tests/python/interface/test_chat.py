import types

import pytest

from mlc_llm.cli import chat as chat_cli
from mlc_llm.interface.chat import (
    ChatState,
    ModelConfigOverride,
    _get_vocab_size,
    _parse_random_tokens_command,
)
from mlc_llm.protocol.openai_api_protocol import CompletionUsage


class FakeTokenizer:
    def encode(self, text):
        if text == "":
            return []
        return [int(token) for token in text.split()]

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeEngine:
    def __init__(self, model_path=None):
        self.engine_config = types.SimpleNamespace(model=model_path)
        self.tokenizer = FakeTokenizer()


def test_cli_forwards_random_token_options(monkeypatch):
    captured_kwargs = {}

    def fake_chat(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(chat_cli, "chat", fake_chat)

    chat_cli.main(
        [
            "dist/model",
            "--device",
            "vulkan",
            "--model-lib",
            "model.so",
            "--random-tokens",
            "512",
            "--max-decode-tokens",
            "128",
        ]
    )

    assert captured_kwargs == {
        "model": "dist/model",
        "device": "vulkan",
        "model_lib": "model.so",
        "overrides": ModelConfigOverride(),
        "random_tokens": 512,
        "max_decode_tokens": 128,
    }


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("/random-tokens 512", (512, None)),
        ("/random-tokens 512 128", (512, 128)),
    ],
)
def test_parse_random_tokens_command(prompt, expected):
    assert _parse_random_tokens_command(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "/random-tokens",
        "/random-tokens 0",
        "/random-tokens abc",
        "/random-tokens 512 0",
        "/random-tokens 512 128 extra",
    ],
)
def test_parse_random_tokens_command_rejects_invalid_input(prompt):
    with pytest.raises(ValueError):
        _parse_random_tokens_command(prompt)


def test_get_vocab_size_reads_tokenizer_json(tmp_path):
    (tmp_path / "tokenizer.json").write_text(
        '{"model": {"vocab": {"<s>": 0, "hello": 1, "world": 2}}}',
        encoding="utf-8",
    )

    assert _get_vocab_size(FakeEngine(model_path=str(tmp_path))) == 3


def test_get_vocab_size_uses_common_fallback_without_tokenizer_json(tmp_path):
    assert _get_vocab_size(FakeEngine(model_path=str(tmp_path))) == 32000


def test_stats_prints_speed_and_exact_token_counts(capsys):
    chat_state = ChatState(FakeEngine())
    chat_state.last_finished_request_usage = CompletionUsage(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        extra={"prefill_tokens_per_s": 123.45, "decode_tokens_per_s": 67.89},
    )

    chat_state.stats()

    assert capsys.readouterr().out == (
        "prefill: 123.5 tok/s, decode: 67.9 tok/s\n"
        "prompt tokens: 11, completion tokens: 7, total tokens: 18\n"
    )


def test_generate_random_tokens_applies_temporary_decode_limit(monkeypatch, capsys):
    chat_state = ChatState(FakeEngine())
    chat_state.overrides.max_tokens = 7
    generated_prompts = []

    monkeypatch.setattr("mlc_llm.interface.chat._get_vocab_size", lambda _engine: 100)
    monkeypatch.setattr(chat_state, "_get_template_overhead", lambda: 2)

    def fake_generate(prompt):
        assert chat_state.overrides.max_tokens == 13
        generated_prompts.append(prompt)

    monkeypatch.setattr(chat_state, "generate", fake_generate)

    chat_state.generate_random_tokens(5, max_decode_tokens=13)

    assert chat_state.overrides.max_tokens == 7
    assert len(chat_state.engine.tokenizer.encode(generated_prompts[0])) == 3
    assert "[random-tokens] target=5, template_overhead=2" in capsys.readouterr().out
