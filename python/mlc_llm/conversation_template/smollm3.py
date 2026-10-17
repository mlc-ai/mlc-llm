"""SmolLM3 conversation templates.

The HF template writes a metadata header into the system turn. Its "Today Date" line is
generated at render time; a static template cannot reproduce it, so it is left out. The HF
template closes the system turn with <|im_end|> only when tools are given or the system message
asks for /system_override (chat_template.jinja lines 67 and 28). Otherwise the system text runs
straight into the first user turn, as it does here.

smollm3: reasoning on (the HF default), with the HF default system message for that mode.
smollm3_nothink: reasoning off; the assistant turn opens with an empty think block.
"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="smollm3",
        system_template=(
            "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\n"
            "Reasoning Mode: /think\n\n## Custom Instructions\n\n"
            f"{MessagePlaceholders.SYSTEM.value}\n\n"
        ),
        system_message=(
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role "
            "as an assistant involves thoroughly exploring questions through a systematic "
            "thinking process before providing the final precise and accurate solutions. This "
            "requires engaging in a comprehensive cycle of analysis, summarizing, "
            "exploration, reassessment, reflection, backtracking, and iteration to develop "
            "well-considered thinking process. Please structure your response into two main "
            "sections: Thought and Solution using the specified format: <think> Thought "
            "section </think> Solution section. In the Thought section, detail your reasoning "
            "process in steps. Each step should include detailed considerations such as "
            "analysing questions, summarizing relevant findings, brainstorming new ideas, "
            "verifying the accuracy of the current steps, refining any errors, and revisiting "
            "previous steps. In the Solution section, based on various attempts, "
            "explorations, and reflections from the Thought section, systematically present "
            "the final solution that you deem correct. The Solution section should be "
            "logical, accurate, and concise and detail necessary steps needed to reach the "
            "conclusion."
        ),
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[128012],  # "<|im_end|>"
    )
)

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="smollm3_nothink",
        system_template=(
            "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\n"
            "Reasoning Mode: /no_think\n\n## Custom Instructions\n\n"
            f"{MessagePlaceholders.SYSTEM.value}\n\n"
        ),
        system_message="You are a helpful AI assistant named SmolLM, trained by Hugging Face.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>\n\n</think>",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|im_end|>"],
        stop_token_ids=[128012],  # "<|im_end|>"
    )
)
