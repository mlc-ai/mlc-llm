"""OLMo2 default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# OLMo-2 Instruct (Tulu format)
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="olmo2",
        system_template=f"<|system|>\n{MessagePlaceholders.SYSTEM.value}\n",
        system_message="",
        roles={
            "user": "<|user|>",
            "assistant": "<|assistant|>",
        },
        seps=["<|endoftext|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[100257],
        system_prefix_token_ids=[100257],
    )
)
