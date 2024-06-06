"""Tiny Llama default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# TinyLlama v1.0
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="tinyllama_v1_0",
        system_template=f"<|system|>\n{MessagePlaceholders.SYSTEM.value}</s>",
        system_message="You are a helpful chatbot.",
        roles={"user": "<|user|>", "assistant": "<|assistant|>"},
        seps=["</s>"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["</s>"],
        stop_token_ids=[2],
    )
)
