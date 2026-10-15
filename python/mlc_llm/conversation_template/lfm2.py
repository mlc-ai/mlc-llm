"""LFM2 conversation template: ChatML with a leading BOS.

The token ids are those of the 65,536-token vocabulary that every LFM2 and LFM2.5
checkpoint uses except LFM2.5-2.6B (128,000 tokens), which needs other ids.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="lfm2",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="",
        render_empty_system_message=False,
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[7],  # "<|im_end|>"
        system_prefix_token_ids=[1],  # "<|startoftext|>"
    )
)
