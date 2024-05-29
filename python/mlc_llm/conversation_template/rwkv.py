"""RWKV default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# RWKV World
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="rwkv_world",
        system_template=f"User: hi\n\nAssistant: {MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "Hi. I am your assistant and I will provide expert full response "
            "in full details. Please feel free to ask any question and I will "
            "always answer it."
        ),
        roles={"user": "User", "assistant": "Assistant"},
        seps=["\n\n"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["\n\n"],
        stop_token_ids=[0],
    )
)
