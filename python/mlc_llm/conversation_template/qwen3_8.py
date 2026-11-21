"""Pinned Qwen3.8 plain-text templates with static thinking-effort selection.

Tool use, multimodal content, assistant continuation, and dynamic template options
are unsupported. Structured assistant reasoning is preserved in historical turns.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

for name, effort in (
    ("qwen3_8", "xhigh"),
    ("qwen3_8_medium", "medium"),
    ("qwen3_8_low", "low"),
    ("qwen3_8_nothink", "none"),
):
    ConvTemplateRegistry.register_conv_template(
        Conversation(
            name=name,
            system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
            system_message="",
            render_empty_system_message=False,
            roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
            seps=["<|im_end|>\n"],
            role_content_sep="\n",
            role_empty_sep="\n<think>\n\n</think>\n\n" if effort == "none" else "\n<think>\n",
            stop_str=["<|endoftext|>", "<|im_end|>"],
            stop_token_ids=[248046, 248044],
            reasoning_history_style="qwen3_8",
            qwen3_8_reasoning_effort=effort,
        )
    )
