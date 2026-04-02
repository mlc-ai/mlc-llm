"""Qwen3.5 conversation templates.

qwen3_5: Thinking enabled — assistant prefix opens a <think> block for the model
         to reason in before responding.
qwen3_5_nothink: Thinking disabled — assistant prefix includes a closed empty
                 <think> block so the model skips straight to responding.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
    )
)

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5_nothink",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>\n\n</think>\n",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
    )
)
