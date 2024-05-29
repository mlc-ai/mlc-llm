"""Orion default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Orion
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="orion",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "Human: ", "assistant": "Assistant: "},
        seps=["\n\n", "</s>"],
        role_content_sep="",
        role_empty_sep="</s>",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)
