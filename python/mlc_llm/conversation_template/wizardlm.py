"""WiazrdLM and Coder default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Wizard LM 7B
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="wizardlm_7b",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "User", "assistant": "Response"},
        seps=["###"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["###"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)

# WizardCoder or WizardMath
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="wizard_coder_or_math",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request."
        ),
        roles={"user": "Instruction", "assistant": "Response"},
        seps=["\n\n### ", "\n\n### "],
        role_content_sep=":\n",
        role_empty_sep=":\n",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)
