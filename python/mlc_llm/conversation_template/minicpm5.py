"""MiniCPM5 conversation templates.

ChatML with a leading BOS. MiniCPM5's HF chat template keeps the reasoning of earlier
assistant turns and only inserts an empty `<think>\n\n</think>\n\n` block into turns that
carry none, so history is passed through as generated.

minicpm5: thinking on (the HF default).
minicpm5_nothink: thinking off; the assistant turn opens with a closed empty think block.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="minicpm5",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="",
        render_empty_system_message=False,
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[1, 130073],  # "</s>", "<|im_end|>"
        system_prefix_token_ids=[0],  # "<s>"
    )
)

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="minicpm5_nothink",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="",
        render_empty_system_message=False,
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>\n\n</think>\n",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[1, 130073],  # "</s>", "<|im_end|>"
        system_prefix_token_ids=[0],  # "<s>"
    )
)
