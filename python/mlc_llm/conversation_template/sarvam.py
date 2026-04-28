"""Sarvam default template"""
from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="sarvam",
        system_template="[@BOS@]\n",
        system_message="",
        roles={
            "user": "<|start_of_turn|><|user|>",
            "assistant": "<|start_of_turn|><|assistant|>",
        },
        seps=["<|end_of_turn|>\n"],
        role_content_sep="\n",
        role_empty_sep="",
        stop_str=["<|end_of_turn|>"],
    )
)