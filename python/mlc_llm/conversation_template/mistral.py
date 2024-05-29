"""Mistral default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Mistral default
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="mistral_default",
        system_template=f"[INST] {MessagePlaceholders.SYSTEM.value}",
        system_message="Always assist with care, respect, and truth. Respond with utmost "
        "utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. "
        "Ensure replies promote fairness and positivity.",
        roles={"user": "[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
        add_role_after_system_message=False,
    )
)
