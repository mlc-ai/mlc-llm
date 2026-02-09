"""Ministral3 reasoning templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Ministral-3-XB-Reasoning-2512
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="ministral3_reasoning",
        system_template=(
            f"[SYSTEM_PROMPT]{MessagePlaceholders.SYSTEM.value}[/SYSTEM_PROMPT]"
            f"{MessagePlaceholders.FUNCTION.value}"
        ),
        system_message=(
            "# HOW YOU SHOULD THINK AND ANSWER\n\n"
            "First draft your thinking process (inner monologue) until you arrive at a response. "
            "Format your response using Markdown, and use LaTeX for any mathematical equations. "
            "Write both your thoughts and the response in the same language as the input.\n\n"
            "Your thinking process must follow the template below:"
            "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
            "Be as casual and as long as you want until you are confident to generate the response "
            "to the user.[/THINK]Here, provide a self-contained response."
        ),
        role_templates={
            "user": f"[INST]{MessagePlaceholders.USER.value}[/INST]",
            "assistant": f"{MessagePlaceholders.ASSISTANT.value}</s>",
            "tool": f"[TOOL_RESULTS]{MessagePlaceholders.TOOL.value}[/TOOL_RESULTS]",
        },
        roles={"user": "", "assistant": "", "tool": ""},
        seps=[""],
        role_content_sep="",
        role_empty_sep="",
        stop_str=["</s>"],
        stop_token_ids=[2],
        system_prefix_token_ids=[1],
    )
)
