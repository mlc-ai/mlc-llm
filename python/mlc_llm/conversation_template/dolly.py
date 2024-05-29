"""Dolly default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Dolly
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="dolly",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "Below is an instruction that describes a task. Write "
            "a response that appropriately completes the request."
        ),
        roles={"user": "### Instruction", "assistant": "### Response"},
        seps=["\n\n", "### End\n"],
        role_content_sep=":\n",
        role_empty_sep=":\n",
        stop_str=["### End"],
        stop_token_ids=[50256],
    )
)
