"""nemotron default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Nemotron template
# https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct/blob/6a417790c444fd65a3da6a5c8821de6afc9654a6/tokenizer_config.json#L8030
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="nemotron",
        system_template=("<extra_id_0>System\n" f"{MessagePlaceholders.SYSTEM.value}\n\n"),
        system_message="",
        roles={
            "user": "<extra_id_1>User",
            "assistant": "<extra_id_1>Assistant",
            "tool": "<extra_id_1>Tool",
        },
        seps=["\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["</s>"],
        stop_token_ids=[3],
        system_prefix_token_ids=[2],
        add_role_after_system_message=True,
    )
)
