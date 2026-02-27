"""Gemma default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Gemma Instruction
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"},
        seps=["<end_of_turn>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<end_of_turn>"],
        stop_token_ids=[1, 107],
        system_prefix_token_ids=[2],
    )
)

# Gemma 3 Instruction. Same as gemma_instruction but with different stop token id
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma3_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"},
        seps=["<end_of_turn>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<end_of_turn>"],
        stop_token_ids=[1, 106],
        system_prefix_token_ids=[2],
    )
)
