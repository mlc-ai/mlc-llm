"""Llava default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Llava
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llava",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="\n",
        roles={"user": "USER", "assistant": "ASSISTANT"},
        seps=[" "],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
        add_role_after_system_message=False,
    )
)
