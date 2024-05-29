"""GLM default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# GLM
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="glm",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={
            "user": "问",
            "assistant": "答",
            "tool": "问",
        },
        seps=["\n\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[64790, 64792],
    )
)
