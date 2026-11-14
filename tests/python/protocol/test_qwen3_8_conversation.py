"""Qwen3.8 prompt parity against the pinned HF template, without model inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.serve.engine_base import process_chat_completion_request

FIXTURE = json.loads((Path(__file__).parent / "fixtures/qwen3_8/hf.json").read_text())


def render_request(request, template):
    pieces = []

    def encode(text):
        pieces.append(text)
        return [1]

    conv = ConvTemplateRegistry.get_conv_template(template).model_copy(deep=True)
    result = process_chat_completion_request(
        request,
        "test",
        SimpleNamespace(record_event=lambda *args, **kwargs: None),
        {},
        encode,
        100000,
        conv,
    )
    return "".join(pieces), conv, result


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["template"] + ":" + c["name"])
def test_pinned_hf_rendering_and_request_bridge(case):
    rendered, conv, result = render_request(
        ChatCompletionRequest(messages=case["messages"]), case["template"]
    )
    assert rendered == case["rendered"]
    assert conv.stop_token_ids == FIXTURE["stop_token_ids"]
    assert "reasoning_contents" not in conv.to_json_dict()
    assert not result[2]


@pytest.mark.parametrize("field", ["enable_thinking", "reasoning_effort", "preserve_thinking"])
@pytest.mark.parametrize("extra", [False, True])
def test_dynamic_template_options_rejected(field, extra):
    options = {"extra_body": {field: None}} if extra else {field: None}
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], **options)
    with pytest.raises(ValueError, match="static template"):
        render_request(request, "qwen3_8")


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "partial"}],
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        [{"role": "user", "content": "<tool_response>result</tool_response>"}],
        [{"role": "user", "content": "hi", "name": None}],
        [
            {"role": "assistant", "content": "hi", "reasoning_content": None},
            {"role": "user", "content": "next"},
        ],
        [{"role": "user", "content": "hi", "reasoning_content": "reason"}],
    ],
)
def test_unsupported_message_scope(messages):
    with pytest.raises(ValueError):
        render_request(ChatCompletionRequest(messages=messages), "qwen3_8")


def test_existing_qwen35_template_unchanged():
    conv = ConvTemplateRegistry.get_conv_template("qwen3_5")
    assert conv.system_message == "You are a helpful assistant."
    assert conv.roles["assistant"] == "<|im_start|>assistant\n<think>"
    assert "qwen3_8_reasoning_effort" not in conv.to_json_dict()


def test_serialized_templates_match_shared_native_fixture():
    templates = json.loads(
        (Path(__file__).parent / "fixtures/qwen3_8/mlc_templates.json").read_text()
    )
    for name, serialized in templates.items():
        assert ConvTemplateRegistry.get_conv_template(name).to_json_dict() == serialized


@pytest.mark.parametrize("choice", ["none", "auto"])
def test_explicit_tool_choice_rejected(choice):
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], tool_choice=choice
    )
    with pytest.raises(ValueError, match="plain text"):
        render_request(request, "qwen3_8")


def test_null_extra_body_allowed():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], extra_body=None
    )
    assert "hello" in render_request(request, "qwen3_8")[0]
