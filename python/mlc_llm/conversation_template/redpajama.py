"""RedPajama default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# RedPajama Chat
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="redpajama_chat",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<human>", "assistant": "<bot>"},
        seps=["\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<human>"],
        stop_token_ids=[0],
    )
)
