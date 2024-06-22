"""Cohere default templates"""
# Referred from: https://huggingface.co/CohereForAI/aya-23-8B/blob/main/tokenizer_config.json

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Aya-23
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="aya-23",
        system_template=f"<|system|>\n{MessagePlaceholders.SYSTEM.value}",
        system_message="You are a helpful digital assistant. Please provide safe, "
        "ethical and accurate information to the user.",
        roles={"user": "<|USER_TOKEN|>", "assistant": "<|CHATBOT_TOKEN|>"},
        seps=["<|SEP|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        system_prefix_token_ids=[255008],
        stop_str=["<|END_OF_TURN_TOKEN|>"],
        stop_token_ids=[255001],
    )
)
