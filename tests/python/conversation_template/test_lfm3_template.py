"""LFM3's limited text-chat rendering, including historical reasoning normalization."""

import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.conversation_protocol import _normalize_lfm3_history


@pytest.mark.parametrize(
    "text,expected",
    [
        ("answer", "<think></think>answer"),
        ("  answer  ", "<think></think>  answer  "),
        ("<think>secret</think> \tanswer\n ", "<think></think>answer"),
        ("x</think>y</think>\u2003answer\u00a0", "<think></think>answer"),
        ("<think>x</think>\t\n", "<think></think>"),
        ("", "<think></think>"),
    ],
)
def test_history(text, expected):
    messages = [("user", "first"), ("assistant", text), ("user", "next"), ("assistant", None)]
    normalized = _normalize_lfm3_history(messages)
    assert normalized[1] == ("assistant", expected)
    assert normalized[-1] == ("assistant", None)
    assert messages[1] == ("assistant", text)


@pytest.mark.parametrize("name,suffix", [("lfm3", "<think>"), ("lfm3_nothink", "<think></think>")])
@pytest.mark.parametrize("system", ["", "Be brief."])
def test_prompt(name, suffix, system):
    conv = ConvTemplateRegistry.get_conv_template(name)
    conv.system_message = system
    conv.messages = [
        ("user", "first"),
        ("assistant", "<think>private</think> final "),
        ("user", "next"),
        ("assistant", None),
    ]
    expected = ("<|im_start|>system\n" + system + "<|im_end|>\n") if system else ""
    expected += (
        "<|im_start|>user\nfirst<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>final<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n" + suffix
    )
    assert "".join(conv.as_prompt()) == expected
    assert conv.system_prefix_token_ids == [124894]
    assert conv.stop_token_ids == [124900]
    assert conv.to_json_dict()["reasoning_history_style"] == "lfm3"


@pytest.mark.parametrize(
    "messages",
    [
        [("assistant", "continuation")],
        [("tool", "result"), ("user", "next")],
        [("user", [{"type": "text", "text": "parts"}])],
    ],
)
def test_unsupported_messages(messages):
    with pytest.raises(ValueError, match="LFM3 templates"):
        _normalize_lfm3_history(messages)
