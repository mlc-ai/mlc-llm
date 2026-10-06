from unittest.mock import Mock

import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.serve import data

pytestmark = [pytest.mark.unittest]


@pytest.mark.parametrize(
    "template_name",
    ["ministral3", "ministral3_reasoning", "gorilla", "gorilla-openfunctions-v2", "chatml"],
)
@pytest.mark.parametrize(
    "parts", [["Hello world"], ["Hello ", "world"], ["", "Hello ", "world", ""]]
)
def test_text_parts_apply_role_template_once(template_name, parts):
    content = [{"type": "text", "text": text} for text in parts]
    request = ChatCompletionRequest(messages=[{"role": "user", "content": content}])
    request.check_message_validity()
    conversation = ConvTemplateRegistry.get_conv_template(template_name).model_copy(deep=True)
    conversation.function_string = '[{"name": "get_weather"}]'
    conversation.messages = [("user", request.messages[0].content), ("assistant", None)]

    expected = conversation.model_copy(deep=True)
    expected.messages = [("user", "".join(parts)), ("assistant", None)]

    assert conversation.as_prompt() == expected.as_prompt()
    assert conversation.messages[0][1] == content
    assert request.messages[0].content == content


@pytest.mark.parametrize("template_name", ["ministral3", "ministral3_reasoning"])
def test_ministral_text_parts_have_one_instruction_pair(template_name):
    conversation = ConvTemplateRegistry.get_conv_template(template_name).model_copy(deep=True)
    conversation.messages = [
        ("user", [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]),
        ("assistant", None),
    ]

    prompt = conversation.as_prompt()[0]

    assert prompt.endswith("[INST]Hello world[/INST]")
    assert prompt.count("[INST]") == 1
    assert prompt.count("[/INST]") == 1


def test_empty_content_parts_keep_existing_format():
    conversation = ConvTemplateRegistry.get_conv_template("ministral3").model_copy(deep=True)
    conversation.system_template = ""
    conversation.messages = [("user", []), ("assistant", None)]

    assert conversation.as_prompt() == [""]


@pytest.mark.parametrize(
    "content, error_type, message",
    [
        ([{"type": "audio"}], ValueError, "Unsupported content type: audio"),
        ([{"text": "hello"}], AssertionError, "Content item should have a type field"),
        ([{"type": "text"}], KeyError, "text"),
    ],
)
def test_invalid_content_parts_keep_existing_errors(content, error_type, message):
    conversation = ConvTemplateRegistry.get_conv_template("chatml").model_copy(deep=True)
    conversation.messages = [("user", content), ("assistant", None)]

    with pytest.raises(error_type, match=message):
        conversation.as_prompt()


def test_mixed_content_parts_keep_image_order_and_separators(monkeypatch):
    image = Mock(spec=data.ImageData)
    from_url = Mock(return_value=image)
    monkeypatch.setattr(data.ImageData, "from_url", from_url)
    config = {"model_type": "llava"}
    content = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "after"},
    ]
    conversation = ConvTemplateRegistry.get_conv_template("chatml").model_copy(deep=True)
    conversation.system_template = ""
    conversation.messages = [("user", content), ("assistant", None)]

    assert conversation.as_prompt(config) == [
        "<|im_start|>user\nbefore",
        image,
        "\nafter<|im_end|>\n<|im_start|>assistant\n",
    ]
    from_url.assert_called_once_with("https://example.com/image.png", config)
    assert conversation.messages[0][1] == content
