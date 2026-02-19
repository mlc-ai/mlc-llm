"""GLM default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# GLM (ChatGLM3)
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="glm",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={
            "user": "问",
            "assistant": "答",
            "tool": "问",
        },
        seps=["\n\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[64790, 64792],
    )
)

# GLM-4.5 (GLM-4.5-Air, GLM-4.5V)
# Chat format: [gMASK]<sop><|system|>\n{system}<|user|>\n{user}<|assistant|>\n{assistant}
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="glm4",
        system_template=f"<|system|>\n{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "tool": "<|observation|>",
        },
        seps=[""],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|user|>", "<|observation|>"],
        stop_token_ids=[151329, 151336, 151338],
        # [gMASK] (151331) and <sop> (151333) are prefix tokens
        system_prefix_token_ids=[151331, 151333],
    )
)
