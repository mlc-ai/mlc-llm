"""Qwen2 default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Same as chatml except system message, stop token, and stop string
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen2",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
    )
)
