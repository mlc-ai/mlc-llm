"""LFM3 plain-text chat templates, with default historical reasoning normalization.

Tools, assistant continuation and preserve_thinking overrides are not supported.
Use lfm3_nothink for the exact non-thinking generation prefix.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

for name, suffix in (("lfm3", "<think>"), ("lfm3_nothink", "<think></think>")):
    ConvTemplateRegistry.register_conv_template(
        Conversation(
            name=name,
            system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
            system_message="",
            render_empty_system_message=False,
            roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
            seps=["<|im_end|>\n"],
            role_content_sep="\n",
            role_empty_sep="\n" + suffix,
            stop_str=["<|im_end|>"],
            stop_token_ids=[124900],
            system_prefix_token_ids=[124894],
            reasoning_history_style="lfm3",
        )
    )
