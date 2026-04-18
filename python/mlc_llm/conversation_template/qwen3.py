"""Qwen3 conversation template.

Matches Qwen2's ChatML structure but strips `<think>...</think>` blocks from
historical assistant turns, mirroring Qwen3's official HF chat template. Small
Qwen3 variants (e.g. 0.6B) otherwise emit `<|im_end|>` prematurely when their
own thinking traces are echoed back in multi-turn context (see mlc-ai/mlc-llm#3482).
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
        strip_reasoning_in_history=True,
    )
)
