"""StableLM default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# StableLM Tuned Alpha
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="stablelm",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n"
            "- StableLM is a helpful and harmless open-source AI language model developed by "
            "StabilityAI.\n"
            "- StableLM is excited to be able to help the user, but will refuse to do "
            "anything that could be considered harmful to the user.\n"
            "- StableLM is more than just an information source, StableLM is also able to "
            "write poetry, short stories, and make jokes.\n"
            "- StableLM will refuse to participate in anything that could harm a human."
        ),
        roles={"user": "<|USER|>", "assistant": "<|ASSISTANT|>"},
        seps=[""],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=[""],
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# StableLM 3B
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="stablelm-3b",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|user|>", "assistant": "<|assistant|>"},
        seps=["<|endoftext|>", "<|endoftext|>"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[0],
    )
)

# StableLM-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="stablelm-2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|user|>", "assistant": "<|assistant|>"},
        seps=["<|endoftext|>", "<|endoftext|>"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[100257],
    )
)
