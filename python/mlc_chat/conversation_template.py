"""The conversation template registry and presets in MLC LLM"""

from typing import Dict, Optional

from .protocol.conversation_protocol import SYSTEM_MESSAGE_PLACEHOLDER, Conversation


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
        system_template=f"[INST] <<SYS>>\n\n{SYSTEM_MESSAGE_PLACEHOLDER}\n<</SYS>>\n\n ",
        system_message="You are a helpful, respectful and honest assistant.",
        roles=("[INST]", "[/INST]"),
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep=" ",
        stop_str=["[INST]"],
        stop_token_ids=[2],
    )
)
