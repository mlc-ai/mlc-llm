"""llama default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Llama3.1 -- same as Llama3 except stop token ids and stop str
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llama-3_1",
        system_template=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{MessagePlaceholders.SYSTEM.value}<|eot_id|>"
        ),
        system_message="You are a helpful, respectful and honest assistant.",
        roles={
            "user": "<|start_header_id|>user",
            "assistant": "<|start_header_id|>assistant",
            "tool": "<|start_header_id|>ipython",
        },
        seps=["<|eot_id|>"],
        role_content_sep="<|end_header_id|>\n\n",
        role_empty_sep="<|end_header_id|>\n\n",
        stop_str=[],
        stop_token_ids=[128001, 128008, 128009],  # "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"
        system_prefix_token_ids=[128000],  # "<|begin_of_text|>"
        add_role_after_system_message=True,
    )
)

# Llama3
# See https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models
# and https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llama-3",
        system_template=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{MessagePlaceholders.SYSTEM.value}<|eot_id|>"
        ),
        system_message="You are a helpful, respectful and honest assistant.",
        roles={"user": "<|start_header_id|>user", "assistant": "<|start_header_id|>assistant"},
        seps=["<|eot_id|>"],
        role_content_sep="<|end_header_id|>\n\n",
        role_empty_sep="<|end_header_id|>\n\n",
        stop_str=["<|end_of_text|>", "<|eot_id|>"],
        stop_token_ids=[128001, 128009],  # "<|end_of_text|>", "<|eot_id|>"
        system_prefix_token_ids=[128000],  # "<|begin_of_text|>"
        add_role_after_system_message=True,
    )
)

# Llama2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llama-2",
        system_template=f"[INST] <<SYS>>\n{MessagePlaceholders.SYSTEM.value}\n<</SYS>>\n\n",
        system_message="You are a helpful, respectful and honest assistant.",
        roles={"user": "<s>[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" ", " </s>"],
        role_content_sep=" ",
        role_empty_sep=" ",
        stop_str=["[INST]"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
        add_role_after_system_message=False,
    )
)

# CodeLlama Completion
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="codellama_completion",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)

# CodeLlama Instruct
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="codellama_instruct",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "[INST]", "assistant": "[/INST]"},
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep=" ",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)
