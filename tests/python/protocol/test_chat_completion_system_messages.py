"""System-message handling in the shared chat-completion preprocessing path."""

import pytest

from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.serve.engine_base import EngineState, process_chat_completion_request


@pytest.mark.parametrize(
    "messages,expected_system,expected_messages,expected_prompt",
    [
        ([("user", "hello")], "DEFAULT", [("user", "hello")], "[S]DEFAULTU:hello|A"),
        (
            [("system", "FIRST"), ("user", "hello")],
            "FIRST",
            [("user", "hello")],
            "[S]FIRSTU:hello|A",
        ),
        ([("system", ""), ("user", "hello")], "", [("user", "hello")], "[S]U:hello|A"),
        (
            [("system", "FIRST"), ("system", "SECOND"), ("user", "hello")],
            "FIRSTSECOND",
            [("user", "hello")],
            "[S]FIRSTSECONDU:hello|A",
        ),
        (
            [("user", "hello"), ("system", "LATER")],
            "LATER",
            [("user", "hello")],
            "[S]LATERU:hello|A",
        ),
        (
            [("system", ""), ("system", ""), ("user", "hello")],
            "",
            [("user", "hello")],
            "[S]U:hello|A",
        ),
        (
            [("system", ""), ("system", "SECOND"), ("system", ""), ("user", "hello")],
            "SECOND",
            [("user", "hello")],
            "[S]SECONDU:hello|A",
        ),
        (
            [
                ("system", "first "),
                ("user", "hello"),
                ("assistant", "answer"),
                ("system", "第二"),
                ("user", "next"),
                ("system", "!"),
            ],
            "first 第二!",
            [("user", "hello"), ("assistant", "answer"), ("user", "next")],
            "[S]first 第二!U:hello|A:answer|U:next|A",
        ),
    ],
)
def test_process_chat_completion_system_messages(
    messages, expected_system, expected_messages, expected_prompt
):
    request = ChatCompletionRequest(
        messages=[{"role": role, "content": content} for role, content in messages]
    )
    request.check_message_validity()
    conversation = Conversation(
        roles={"user": "U", "assistant": "A", "tool": "T"},
        system_template="[S]{system_message}",
        system_message="DEFAULT",
        seps=["|"],
        role_content_sep=":",
    )

    prompts, _, use_function_calling, prompt_length = process_chat_completion_request(
        request,
        "system-messages-test",
        EngineState(enable_tracing=False),
        {},
        lambda text: list(map(ord, text)),
        4096,
        conversation,
    )

    assert conversation.system_message == expected_system
    assert conversation.messages == [*expected_messages, ("assistant", None)]
    assert prompts == [list(map(ord, expected_prompt))]
    assert prompt_length == len(expected_prompt)
    assert not use_function_calling


@pytest.mark.parametrize(
    "message,error",
    [
        (
            {"role": "system", "content": "bad", "tool_call_id": "tool-id"},
            "Non-tool message having `tool_call_id` is invalid.",
        ),
        (
            {"role": "system", "content": [{"type": "text", "text": "bad"}]},
            "Non-user message having a list of content is invalid.",
        ),
        (
            {"role": "user", "content": "bad", "tool_calls": []},
            "Non-assistant message having `tool_calls` is invalid.",
        ),
        (
            {"role": "assistant", "content": "bad", "tool_calls": []},
            "Assistant message having `tool_calls` is not supported yet.",
        ),
    ],
)
def test_system_messages_preserve_other_message_validation(message, error):
    request = ChatCompletionRequest(messages=[{"role": "system", "content": "first"}, message])

    with pytest.raises(BadRequestError) as exc_info:
        request.check_message_validity()

    assert str(exc_info.value) == error
