"""LLM-jp default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# LLM-jp instruct
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llm-jp",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
        roles={
            "user": "\n\n### 指示:",
            "assistant": "\n\n### 応答:",
        },
        seps=["", "</s>"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=[],
        stop_token_ids=[2],  # eos_token_id
        system_prefix_token_ids=[1],  # bos_token_id (<s>)
        add_role_after_system_message=True,
    )
)
