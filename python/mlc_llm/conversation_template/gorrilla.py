"""Gorrilla default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Gorilla
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gorilla",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant provides helpful, detailed, and "
            "polite responses to the user's inquiries."
        ),
        role_templates={
            "user": (
                f"<<question>> {MessagePlaceholders.USER.value} <<function>> "
                f"{MessagePlaceholders.FUNCTION.value}"
            ),
        },
        roles={"user": "USER", "assistant": "ASSISTANT", "tool": "USER"},
        seps=["\n", "</s>"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)

# Gorilla-openfunctions-v2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gorilla-openfunctions-v2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "You are an AI programming assistant, utilizing the Gorilla LLM model, "
            "developed by Gorilla LLM, and you only answer questions related to computer "
            "science. For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer."
        ),
        role_templates={
            "user": (
                f"<<function>>{MessagePlaceholders.FUNCTION.value}\n<<question>>"
                f"{MessagePlaceholders.USER.value}"
            ),
        },
        roles={"user": "### Instruction", "assistant": "### Response", "tool": "### Instruction"},
        seps=["\n", "<|EOT|>"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["<|EOT|>"],
        stop_token_ids=[100015],
        system_prefix_token_ids=[100000],
    )
)
