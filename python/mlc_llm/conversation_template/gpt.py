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
