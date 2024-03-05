"""The conversation template registry and presets in MLC LLM"""

from typing import Dict, Optional

from .protocol.conversation_protocol import Conversation, MessagePlaceholders


class ConvTemplateRegistry:
    """Global conversation template registry for preset templates."""

    _conv_templates: Dict[str, Conversation] = {}

    @staticmethod
    def register_conv_template(conv_template: Conversation, override: bool = False) -> None:
        """Register a new conversation template in the global registry.
        Using `override = True` to override the previously registered
        template with the same name.
        """
        name = conv_template.name
        if name is None:
            raise ValueError("The template to register should have non-None name.")
        if name in ConvTemplateRegistry._conv_templates and not override:
            raise ValueError(
                "The name of the template has been registered "
                f"for {ConvTemplateRegistry._conv_templates[name].model_dump_json()}"
            )
        ConvTemplateRegistry._conv_templates[name] = conv_template

    @staticmethod
    def get_conv_template(name: str) -> Optional[Conversation]:
        """Return the conversation template specified by the given name,
        or None if the template is not registered.
        """
        return ConvTemplateRegistry._conv_templates.get(name, None)


############## Preset Conversation Templates ##############

# Llama2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llama-2",
        system_template=f"[INST] <<SYS>>\n{MessagePlaceholders.SYSTEM.value}\n<</SYS>>\n\n ",
        system_message="You are a helpful, respectful and honest assistant.",
        roles={"user": "[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep=" ",
        stop_str=["[INST]"],
        stop_token_ids=[2],
    )
)

# Mistral default
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="mistral_default",
        system_template=f"[INST] {MessagePlaceholders.SYSTEM.value}\n\n ",
        system_message="Always assist with care, respect, and truth. Respond with utmost "
        "utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. "
        "Ensure replies promote fairness and positivity.",
        roles={"user": "[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[2],
    )
)

# Gorilla
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gorilla",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant provides helpful, detailed, and "
            "polite responses to the user's inquiries."
        ),
        role_templates={
            "user": (
                f"<<question>> {MessagePlaceholders.USER.value} <<function>> "
                f"{MessagePlaceholders.FUNCTION.value}"
            ),
        },
        roles={"user": "USER", "assistant": "ASSISTANT", "tool": "USER"},
        seps=["\n", "</s>"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
    )
)

# ChatML
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="chatml",
        system_template=f"<|im_start|>{MessagePlaceholders.SYSTEM.value}<|im_end|> ",
        system_message=(
            "system A conversation between a user and an LLM-based AI assistant. The "
            "assistant gives helpful and honest answers."
        ),
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant",
            "tool": "<|im_start|>user",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2],
    )
)

# Phi-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="phi-2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={
            "user": "Instruct",
            "assistant": "Output",
            "tool": "Instruct",
        },
        seps=["\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[50256],
    )
)
