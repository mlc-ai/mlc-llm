"""Oasst default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Oasst
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="oasst",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|prompter|>", "assistant": "<|assistant|>"},
        seps=["<|endoftext|>"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[2],
    )
)
