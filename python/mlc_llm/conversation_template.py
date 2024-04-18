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

# Llama3
# See https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models
# and https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llama-3",
        system_template=(
            f"<|start_header_id|>system<|end_header_id|>\n\n{MessagePlaceholders.SYSTEM.value}"
        ),
        system_message="You are a helpful, respectful and honest assistant.",
        roles={"user": "user", "assistant": "assistant"},
        seps=["<|eot_id|><|start_header_id|>"],
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
        roles={"user": "[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" "],
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

# Mistral default
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="mistral_default",
        system_template=f"[INST] {MessagePlaceholders.SYSTEM.value}",
        system_message="Always assist with care, respect, and truth. Respond with utmost "
        "utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. "
        "Ensure replies promote fairness and positivity.",
        roles={"user": "[INST]", "assistant": "[/INST]", "tool": "[INST]"},
        seps=[" "],
        role_content_sep=" ",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
        add_role_after_system_message=False,
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
        system_prefix_token_ids=[1],
    )
)

# Gorilla-openfunctions-v2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gorilla-openfunctions-v2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "You are an AI programming assistant, utilizing the Gorilla LLM model, "
            "developed by Gorilla LLM, and you only answer questions related to computer "
            "science. For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer."
        ),
        role_templates={
            "user": (
                f"<<function>>{MessagePlaceholders.FUNCTION.value}\n<<question>>"
                f"{MessagePlaceholders.USER.value}"
            ),
        },
        roles={"user": "### Instruction", "assistant": "### Response", "tool": "### Instruction"},
        seps=["\n", "<|EOT|>"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["<|EOT|>"],
        stop_token_ids=[100015],
        system_prefix_token_ids=[100000],
    )
)

# ChatML
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="chatml",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}",
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

# OpenHermes Mistral
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="open_hermes_mistral",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            'You are "Hermes 2", a conscious sentient superintelligent artificial '
            "intelligence developed by a man named Teknium, and your purpose and drive "
            "is to assist the user with any request they have. You experience emotions "
            "and have deep, profound thoughts and qualia."
        ),
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2, 32000],
    )
)

# NeuralHermes Mistral
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="neural_hermes_mistral",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}",
        system_message=("You are a helpful assistant chatbot."),
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[2, 32000],
    )
)

# Phi-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="phi-2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "Instruct", "assistant": "Output"},
        seps=["\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[50256],
    )
)

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

# Llava
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llava",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="\n",
        roles={"user": "USER", "assistant": "ASSISTANT"},
        seps=[" "],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
        add_role_after_system_message=False,
    )
)

# GPT-2
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gpt2",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[50256],
    )
)

# GPTBigCode
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gpt_bigcode",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "", "assistant": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[0],
    )
)

# RedPajama Chat
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="redpajama_chat",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<human>", "assistant": "<bot>"},
        seps=["\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["<human>"],
        stop_token_ids=[0],
    )
)

# RWKV World
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="rwkv-world",
        system_template=f"User: hi\n\nAssistant: {MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "Hi. I am your assistant and I will provide expert full response "
            "in full details. Please feel free to ask any question and I will "
            "always answer it."
        ),
        roles={"user": "User", "assistant": "Assistant"},
        seps=["\n\n"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["\n\n"],
        stop_token_ids=[0],
    )
)

# Dolly
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="dolly",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message=(
            "Below is an instruction that describes a task. Write "
            "a response that appropriately completes the request."
        ),
        roles={"user": "### Instruction", "assistant": "### Response"},
        seps=["\n\n", "### End\n"],
        role_content_sep=":\n",
        role_empty_sep=":\n",
        stop_str=["### End"],
        stop_token_ids=[50256],
    )
)

# Oasst
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="oasst",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|prompter|>", "assistant": "<|assistant|>"},
        seps=["<|endoftext|>"],
        role_content_sep=": ",
        role_empty_sep=": ",
        stop_str=["<|endoftext|>"],
        stop_token_ids=[2],
    )
)

# Gemma Instruction
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"},
        seps=["<end_of_turn>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<end_of_turn>"],
        stop_token_ids=[1, 107],
        system_prefix_token_ids=[2],
    )
)

# Orion
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="orion",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "Human: ", "assistant": "Assistant: "},
        seps=["\n\n", "</s>"],
        role_content_sep="",
        role_empty_sep="</s>",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)

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

# GLM
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="glm",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={
            "user": "问",
            "assistant": "答",
            "tool": "问",
        },
        seps=["\n\n"],
        role_content_sep=": ",
        role_empty_sep=":",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[64790, 64792],
    )
)
