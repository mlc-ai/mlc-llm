"""OLMo default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Note that eos_token id is "50279" both in Allenai and AMD version.
# So use the number instead of text.
# Allenai version chat_template and eos_token:
# https://huggingface.co/allenai/OLMo-7B-Instruct/blob/main/tokenizer_config.json
# AMD version chat_template and eos_token:
# https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO/blob/main/tokenizer_config.json
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="olmo",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        system_prefix_token_ids=[50279],
        roles={
            "user": "<|user|>",
            "assistant": "<|assistant|>",
        },
        seps=["\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_token_ids=[50279],
    )
)
