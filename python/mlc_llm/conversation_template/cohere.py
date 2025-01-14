"""Cohere default templates"""

# pylint: disable=line-too-long

# Referred from: https://huggingface.co/CohereForAI/aya-23-8B/blob/main/tokenizer_config.json

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Aya-23
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="aya-23",
        system_template=f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{MessagePlaceholders.SYSTEM.value}<|END_OF_TURN_TOKEN|>",
        system_message="You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses.",
        roles={
            "user": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "assistant": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        },
        seps=["<|END_OF_TURN_TOKEN|>"],
        role_content_sep="",
        role_empty_sep="",
        system_prefix_token_ids=[5],
        stop_str=["<|END_OF_TURN_TOKEN|>"],
        stop_token_ids=[6, 255001],
    )
)
