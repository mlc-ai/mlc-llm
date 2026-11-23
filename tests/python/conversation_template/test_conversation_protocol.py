import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders


def get_conv_templates():
    return [
        "llama-3",
        "llama-2",
        "mistral_default",
        "gorilla",
        "gorilla-openfunctions-v2",
        "chatml",
        "phi-2",
        "codellama_completion",
        "codellama_instruct",
        "rwkv_world",
    ]


@pytest.mark.parametrize("conv_template_name", get_conv_templates())
def test_json(conv_template_name):
    template = ConvTemplateRegistry.get_conv_template(conv_template_name)
    j = template.to_json_dict()
    template_parsed = Conversation.from_json_dict(j)
    assert template == template_parsed


@pytest.mark.parametrize("conv_template_name", get_conv_templates())
def test_prompt(conv_template_name):
    conversation = ConvTemplateRegistry.get_conv_template(conv_template_name)
    user_msg = "test1"
    assistant_msg = "test2"
    prompt = "test3"

    expected_user_msg = (
        conversation.role_templates["user"]
        .replace(MessagePlaceholders.USER.value, user_msg)
        .replace(MessagePlaceholders.FUNCTION.value, "")
    )

    expected_prompt = (
        conversation.role_templates["user"]
        .replace(MessagePlaceholders.USER.value, prompt)
        .replace(MessagePlaceholders.FUNCTION.value, "")
    )

    conversation.messages.append(("user", user_msg))
    conversation.messages.append(("assistant", assistant_msg))
    conversation.messages.append(("user", prompt))
    conversation.messages.append(("assistant", None))
    res = conversation.as_prompt()

    system_msg = conversation.system_template.replace(
        MessagePlaceholders.SYSTEM.value, conversation.system_message
    )
    expected_final_prompt = (
        system_msg
        + (conversation.seps[0] if system_msg != "" else "")
        + (
            conversation.roles["user"] + conversation.role_content_sep
            if conversation.add_role_after_system_message
            else ""
        )
        + expected_user_msg
        + conversation.seps[0 % len(conversation.seps)]
        + conversation.roles["assistant"]
        + conversation.role_content_sep
        + assistant_msg
        + conversation.seps[1 % len(conversation.seps)]
        + conversation.roles["user"]
        + conversation.role_content_sep
        + expected_prompt
        + conversation.seps[0 % len(conversation.seps)]
        + conversation.roles["assistant"]
        + conversation.role_empty_sep
    )
    assert res == expected_final_prompt


# Qwen3.5 reasoning handling, see mlc-ai/mlc-llm#3482.
QWEN3_5_MULTI_TURN = [
    ("user", "What is 2+2?"),
    ("assistant", "Let me compute. 2 plus 2 is 4.\n</think>\n\n4"),
    ("user", "And 3+3?"),
    ("assistant", None),
]


def _qwen3_5_prompt(conv_template_name, messages):
    conversation = ConvTemplateRegistry.get_conv_template(conv_template_name)
    conversation.messages = list(messages)
    return conversation.as_prompt()[0]


@pytest.mark.parametrize("conv_template_name", ["qwen3_5", "qwen3_5_nothink"])
def test_qwen3_5_json(conv_template_name):
    template = ConvTemplateRegistry.get_conv_template(conv_template_name)
    j = template.to_json_dict()
    assert j["strip_reasoning_in_history"] is True
    assert template == Conversation.from_json_dict(j)


@pytest.mark.parametrize(
    "conv_template_name,generation_prompt",
    [
        ("qwen3_5", "<|im_start|>assistant\n<think>\n"),
        ("qwen3_5_nothink", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ],
)
def test_qwen3_5_generation_prompt(conv_template_name, generation_prompt):
    prompt = _qwen3_5_prompt(conv_template_name, [("user", "hi"), ("assistant", None)])
    assert prompt == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n" + generation_prompt
    )


@pytest.mark.parametrize(
    "conv_template_name,generation_prompt",
    [
        ("qwen3_5", "<|im_start|>assistant\n<think>\n"),
        ("qwen3_5_nothink", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ],
)
def test_qwen3_5_strips_reasoning_in_history(conv_template_name, generation_prompt):
    prompt = _qwen3_5_prompt(conv_template_name, QWEN3_5_MULTI_TURN)
    assert "Let me compute. 2 plus 2 is 4." not in prompt
    assert "<|im_start|>assistant\n4<|im_end|>\n" in prompt
    assert prompt.endswith(generation_prompt)
    assert prompt.count("<think>") == generation_prompt.count("<think>")


@pytest.mark.parametrize("conv_template_name", ["qwen3_5", "qwen3_5_nothink"])
def test_qwen3_5_keeps_reasoning_on_last_assistant_turn(conv_template_name):
    prompt = _qwen3_5_prompt(
        conv_template_name, [("user", "hi"), ("assistant", "reasoning\n</think>\n\nanswer")]
    )
    assert "reasoning" in prompt


if __name__ == "__main__":
    test_json("llama-3")
