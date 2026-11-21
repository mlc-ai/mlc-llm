"""Qwen3.6 plain-text templates with default historical reasoning removal.

Source: Qwen/Qwen3.6-27B at 8f33b98937c6031bb49e713dee3e8e185aafd26f.
Tools, media, assistant continuation and dynamic thinking options are unsupported.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

for name, thinking in (("qwen3_6", True), ("qwen3_6_nothink", False)):
    ConvTemplateRegistry.register_conv_template(
        Conversation(
            name=name,
            system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
            system_message="",
            render_empty_system_message=False,
            roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
            seps=["<|im_end|>\n"],
            role_content_sep="\n",
            role_empty_sep="\n<think>\n" if thinking else "\n<think>\n\n</think>\n\n",
            stop_str=["<|endoftext|>", "<|im_end|>"],
            stop_token_ids=[248046, 248044],
            reasoning_history_style="qwen3_6",
        )
    )
