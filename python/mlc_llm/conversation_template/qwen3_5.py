"""Qwen3.5 conversation templates.

qwen3_5: Thinking enabled - the `<think>` opener is emitted on the generation
         prompt only, and `<think>...</think>` blocks are stripped from
         historical assistant turns.
qwen3_5_nothink: Thinking disabled - the generation prompt carries a closed
                 empty `<think>` block so the model skips straight to
                 responding, and history is stripped the same way.

Both mirror Qwen's official HF chat template. Echoing thinking traces back into
multi-turn context makes small variants emit `<|im_end|>` prematurely, which is
the failure fixed for Qwen3 in mlc-ai/mlc-llm#3482.
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
            "assistant": "<|im_start|>assistant",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n<think>\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
        strip_reasoning_in_history=True,
    )
)

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5_nothink",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n<think>\n\n</think>\n\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
        strip_reasoning_in_history=True,
    )
)
