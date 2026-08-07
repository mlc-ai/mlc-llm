"""GPT-2 and GPT bigcode default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# GPT-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gpt2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[50256],
    )
)

# GPTBigCode
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gpt_bigcode",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[0],
    )
)

# GPT_OSS
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gpt_oss",
        system_template=f"<|start|>system<|message|>{MessagePlaceholders.SYSTEM.value}<|end|>",
        system_message="You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.",
        roles={"user": "<|start|>user<|message|>", "assistant": "<|start|>assistant"},
        seps=["<|end|>"],
        role_content_sep="<|end|>",
        role_empty_sep="<|end|>",
        stop_str=["<|return|>", "<|endoftext|>"],
        stop_token_ids=[200002, 199999],
    )
)
