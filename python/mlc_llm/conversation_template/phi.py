"""Phi default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Phi-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="phi-2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "Instruct", "assistant": "Output"},
        seps=["\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[50256],
    )
)

# Phi-3
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="phi-3",
        system_template=f"<|system|>\n{MessagePlaceholders.SYSTEM.value}",
        system_message="You are a helpful digital assistant. Please provide safe, "
        "ethical and accurate information to the user.",
        roles={"user": "<|user|>", "assistant": "<|assistant|>"},
        seps=["<|end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        system_prefix_token_ids=[1],
        stop_str=["<|endoftext|>"],
        stop_token_ids=[2, 32000, 32001, 32007],
    )
)

# Phi-3-vision
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="phi-3-vision",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|user|>", "assistant": "<|assistant|>"},
        seps=["<|end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        system_prefix_token_ids=[1],
        stop_str=["<|endoftext|>"],
        stop_token_ids=[2, 32000, 32001, 32007],
    )
)
