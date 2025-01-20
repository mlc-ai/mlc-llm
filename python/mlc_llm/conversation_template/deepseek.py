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

# Deepseek V2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_v2",
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

# Deepseek V3
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_v3",
        system_template=f"<｜begin▁of▁sentence｜>{MessagePlaceholders.SYSTEM.value}",
        system_message="You are a helpful assistant.",
        roles={"user": "<｜User｜>", "assistant": "<｜Assistant｜>"},
        seps=["", "<｜end▁of▁sentence｜>"],
        role_content_sep="",
        role_empty_sep="",
        stop_token_ids=[151643],
    )
)
