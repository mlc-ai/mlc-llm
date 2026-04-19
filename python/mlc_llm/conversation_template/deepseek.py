"""Deepseek default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Deepseek
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        system_prefix_token_ids=[100000],
        roles={"user": "User", "assistant": "Assistant"},
        seps=["\n\n", "<｜end▁of▁sentence｜>"],  # noqa: RUF001
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<｜end▁of▁sentence｜>"],  # noqa: RUF001
        stop_token_ids=[100001],
    )
)

# Deepseek V2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_v2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        system_prefix_token_ids=[100000],
        roles={"user": "User", "assistant": "Assistant"},
        seps=["\n\n", "<｜end▁of▁sentence｜>"],  # noqa: RUF001
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<｜end▁of▁sentence｜>"],  # noqa: RUF001
        stop_token_ids=[100001],
    )
)

# DeepSeek-V3
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_v3",
        system_template=f"<｜begin▁of▁sentence｜>{MessagePlaceholders.SYSTEM.value}",  # noqa: RUF001
        system_message="You are Deepseek-V3, an AI assistant created exclusively by the Chinese "
        "Company DeepSeek. You'll provide helpful, harmless, and detailed responses to all "
        "user inquiries.",
        roles={"user": "<｜User｜>", "assistant": "<｜Assistant｜>"},  # noqa: RUF001
        seps=["", "<｜end▁of▁sentence｜>"],  # noqa: RUF001
        role_content_sep="",
        role_empty_sep="",
        stop_token_ids=[1],
    )
)

# DeepSeek-R1-Distill-Qwen
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_r1_qwen",
        system_template=f"<｜begin▁of▁sentence｜>{MessagePlaceholders.SYSTEM.value}",  # noqa: RUF001
        system_message="You are Deepseek-R1, an AI assistant created exclusively by the Chinese "
        "Company DeepSeek. You'll provide helpful, harmless, and detailed responses to all "
        "user inquiries.",
        roles={"user": "<｜User｜>", "assistant": "<｜Assistant｜>"},  # noqa: RUF001
        seps=["", "<｜end▁of▁sentence｜>"],  # noqa: RUF001
        role_content_sep="",
        role_empty_sep="",
        stop_token_ids=[151643],
    )
)

# DeepSeek-R1-Distill-Llama, exactly the same as DeepSeek-R1-Distill-Qwen, but different stop token
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="deepseek_r1_llama",
        system_template=f"<｜begin▁of▁sentence｜>{MessagePlaceholders.SYSTEM.value}",  # noqa: RUF001
        system_message="You are Deepseek-R1, an AI assistant created exclusively by the Chinese "
        "Company DeepSeek. You'll provide helpful, harmless, and detailed responses to all"
        " user inquiries.",
        roles={"user": "<｜User｜>", "assistant": "<｜Assistant｜>"},  # noqa: RUF001
        seps=["", "<｜end▁of▁sentence｜>"],  # noqa: RUF001
        role_content_sep="",
        role_empty_sep="",
        stop_token_ids=[128001],
    )
)
