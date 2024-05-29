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


if __name__ == "__main__":
    test_json("llama-3")
