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

# Gemma 4 Instruction
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma4_instruction",
        system_template=f"<|turn>system\n{MessagePlaceholders.SYSTEM.value}<turn|>\n",
        system_message="",
        render_empty_system_message=False,
        roles={"user": "<|turn>user", "assistant": "<|turn>model"},
        seps=["<turn|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<turn|>"],
        stop_token_ids=[1, 106, 50],
        system_prefix_token_ids=[2],
    )
)

# Pinned Gemma 4 large text profile; existing E2B behavior stays unchanged.
for _thinking in (False, True):
    ConvTemplateRegistry.register_conv_template(
        Conversation(
            name="gemma4_large_thinking" if _thinking else "gemma4_large",
            system_template=(
                "<|turn>system\n"
                + ("<|think|>\n" if _thinking else "")
                + MessagePlaceholders.SYSTEM.value
                + "<turn|>\n"
            ),
            system_message="",
            render_empty_system_message=_thinking,
            roles={"user": "<|turn>user", "assistant": "<|turn>model"},
            seps=["<turn|>\n"],
            role_content_sep="\n",
            role_empty_sep="\n" if _thinking else "\n<|channel>thought\n<channel|>",
            stop_str=["<turn|>"],
            stop_token_ids=[1, 106, 50],
            system_prefix_token_ids=[2],
            reasoning_history_style="gemma4",
        )
    )
