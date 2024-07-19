"""The conversation template registry and presets in MLC LLM"""

from typing import Dict, Optional

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders


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
                f"for {ConvTemplateRegistry._conv_templates[name].model_dump_json(by_alias=True)}"
            )
        ConvTemplateRegistry._conv_templates[name] = conv_template

    @staticmethod
    def get_conv_template(name: str) -> Optional[Conversation]:
        """Return the conversation template specified by the given name,
        or None if the template is not registered.
        """
        return ConvTemplateRegistry._conv_templates.get(name, None)


# ChatML
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="chatml",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message=(
            "A conversation between a user and an LLM-based AI assistant. The "
            "assistant gives helpful and honest answers."
        ),
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2],
    )
)

# ChatML without a system prompt
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="chatml_nosystem",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2],
    )
)


# Vanilla LM
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="LM",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=[],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)
