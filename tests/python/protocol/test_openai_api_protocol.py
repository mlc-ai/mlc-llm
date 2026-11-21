import pytest

from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def _make_request(tool_choice):
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="test",
        tools=[
            {
                "type": "function",
                "function": {"name": "foo", "description": "d", "parameters": {}},
            }
        ],
        tool_choice=tool_choice,
    )


def _make_conv():
    return Conversation(name="test", roles={"user": "user", "assistant": "assistant"}, seps=[" "])


def test_check_function_call_usage_missing_function_name_raises_bad_request():
    """tool_choice is an unstructured Dict, so a request can omit the required
    "name" field. This must raise the documented BadRequestError, not a raw
    KeyError."""
    request = _make_request({"type": "function", "function": {}})
    with pytest.raises(BadRequestError):
        request.check_function_call_usage(_make_conv())


def test_check_function_call_usage_missing_type_raises_bad_request():
    """Same class of issue for a missing "type" key."""
    request = _make_request({"function": {"name": "foo"}})
    with pytest.raises(BadRequestError):
        request.check_function_call_usage(_make_conv())


def test_check_function_call_usage_selects_tool_on_valid_choice():
    request = _make_request({"type": "function", "function": {"name": "foo"}})
    conv_template = _make_conv()
    request.check_function_call_usage(conv_template)
    assert conv_template.use_function_calling is True
