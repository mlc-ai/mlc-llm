"""Deepseek default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Deepseek
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        system_prefix_token_ids=[100000],
        roles={"user": "User", "assistant": "Assistant"},
        seps=["\n\n", "<｜end▁of▁sentence｜>"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<｜end▁of▁sentence｜>"],
        stop_token_ids=[100001],
    )
)