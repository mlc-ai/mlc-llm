"""Qwen3.6 and Qwen3.8 prompt parity against their reference templates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.serve.engine_base import process_chat_completion_request

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURES = {
    family: json.loads((FIXTURE_DIR / family / "hf.json").read_text())
    for family in ("qwen3_6", "qwen3_8")
}
CASES = [
    pytest.param(family, case, id=f"{case['template']}:{case['name']}")
    for family, fixture in FIXTURES.items()
    for case in fixture["cases"]
]


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


@pytest.mark.parametrize("family,case", CASES)
def test_hf_rendering_and_request_bridge(family, case):
    rendered, conv, result = render_request(
        ChatCompletionRequest(messages=case["messages"]), case["template"]
    )
    assert rendered == case["rendered"]
    assert conv.stop_token_ids == FIXTURES[family]["stop_token_ids"]
    assert "reasoning_contents" not in conv.to_json_dict()
    assert not result[2]


@pytest.mark.parametrize("template", FIXTURES)
@pytest.mark.parametrize("field", ["enable_thinking", "reasoning_effort", "preserve_thinking"])
@pytest.mark.parametrize("extra", [False, True])
def test_dynamic_template_options_rejected(template, field, extra):
    options = {"extra_body": {field: None}} if extra else {field: None}
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], **options)
    with pytest.raises(ValueError, match="static template"):
        render_request(request, template)


@pytest.mark.parametrize("template", FIXTURES)
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
def test_unsupported_message_scope(template, messages):
    with pytest.raises(ValueError):
        render_request(ChatCompletionRequest(messages=messages), template)


def test_existing_qwen35_template_unchanged():
    conv = ConvTemplateRegistry.get_conv_template("qwen3_5")
    assert conv.system_message == "You are a helpful assistant."
    assert conv.roles["assistant"] == "<|im_start|>assistant\n<think>"
    assert "qwen3_8_reasoning_effort" not in conv.to_json_dict()


@pytest.mark.parametrize("family", FIXTURES)
def test_serialized_templates_match_shared_native_fixture(family):
    templates = json.loads((FIXTURE_DIR / family / "mlc_templates.json").read_text())
    for name, serialized in templates.items():
        assert ConvTemplateRegistry.get_conv_template(name).to_json_dict() == serialized


@pytest.mark.parametrize("template", FIXTURES)
@pytest.mark.parametrize("choice", ["none", "auto"])
def test_explicit_tool_choice_rejected(template, choice):
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], tool_choice=choice
    )
    with pytest.raises(ValueError, match="plain text"):
        render_request(request, template)


@pytest.mark.parametrize("template", FIXTURES)
def test_null_extra_body_allowed(template):
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], extra_body=None
    )
    assert "hello" in render_request(request, template)[0]
