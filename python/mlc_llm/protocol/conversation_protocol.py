"""The standard conversation protocol in MLC LLM"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# The message placeholders in the message prompts according to roles.
class MessagePlaceholders(Enum):
    """The message placeholders in the message prompts according to roles."""

    SYSTEM = "{system_message}"
    USER = "{user_message}"
    ASSISTANT = "{assistant_message}"
    TOOL = "{tool_message}"
    FUNCTION = "{function_string}"


class Conversation(BaseModel):
    """Class that specifies the convention template of conversation
    and contains the conversation history.

    Given a conversation template, the corresponding prompt generated out
    from it is usually in the following format:

      <<system>><<messages[0][0]>><<role_content_sep>><<messages[0][1]>><<seps[0]>>
                <<messages[1][0]>><<role_content_sep>><<messages[1][1]>><<seps[1]>>
                ...
                <<messages[2][0]>><<role_content_sep>><<messages[2][1]>><<seps[0]>>
                <<roles[1]>><<role_empty_sep>>
    """

    # Optional name of the template.
    name: Optional[str] = None
    # The system prompt template, it optionally contains the system
    # message placeholder, and the placeholder will be replaced with
    # the system message below.
    system_template: str = MessagePlaceholders.SYSTEM.value
    # The content of the system prompt (without the template format).
    system_message: str = ""
    # The system token ids to be prepended at the beginning of tokenized
    # generated prompt.
    system_prefix_token_ids: Optional[List[int]] = None

    # The conversation roles
    roles: Dict[str, str]

    # The roles prompt template, it optionally contains the defaults
    # message placeholders and will be replaced by actual content
    role_templates: Dict[str, str]

    # The conversation history messages.
    # Each message is a pair of strings, denoting "(role, content)".
    # The content can be None.
    messages: List[Tuple[str, Optional[str]]] = Field(default_factory=lambda: [])

    # The separators between messages when concatenating into a single prompt.
    # List size should be either 1 or 2.
    # - When size is 1, the separator will be used between adjacent messages.
    # - When size is 2, seps[0] is used after user message, and
    #   seps[1] is used after assistant message.
    seps: List[str]

    # The separator between the role and the content in a message.
    role_content_sep: str = ""
    # The separator between the role and empty contents.
    role_empty_sep: str = ""

    # The stop criteria
    stop_str: List[str] = Field(default_factory=lambda: [])
    stop_token_ids: List[int] = Field(default_factory=lambda: [])

    # Function call fields
    function_string: str = ""
    # whether using function calling or not, helps check for output message format in API call
    use_function_calling: bool = False

    def __init__(self, role_templates: Optional[Dict[str, str]] = None, **kwargs):
        # Defaults templates which would be overridden by model specific templates
        _role_templates: Dict[str, str] = {
            "user": MessagePlaceholders.USER.value,
            "assistant": MessagePlaceholders.ASSISTANT.value,
            "tool": MessagePlaceholders.TOOL.value,
        }
        if role_templates is not None:
            _role_templates.update(role_templates)
        super().__init__(role_templates=_role_templates, **kwargs)

    @field_validator("seps")
    @classmethod
    def check_message_seps(cls, seps: List[str]) -> List[str]:
        """Check if the input message separators has size 1 or 2."""
        if len(seps) == 0 or len(seps) > 2:
            raise ValueError("seps should have size 1 or 2.")
        return seps

    def as_prompt(self) -> str:
        """Convert the conversation template and history messages to
        a single prompt.
        """
        # - Get the system message.
        system_msg = self.system_template.replace(
            MessagePlaceholders.SYSTEM.value, self.system_message
        )

        # - Get the message strings.
        message_list: List[str] = []
        separators = list(self.seps)
        if len(separators) == 1:
            separators.append(separators[0])
        for role, content in self.messages:  # pylint: disable=not-an-iterable
            if role not in self.roles.keys():
                raise ValueError(f'Role "{role}" is not a supported role in {self.roles.keys()}')
            separator = separators[role == "assistant"]  # check assistant role
            if content is not None:
                message_string = (
                    self.roles[role]
                    + self.role_content_sep
                    + self.role_templates[role].replace(
                        MessagePlaceholders[role.upper()].value, content
                    )
                    + separator
                )
            else:
                message_string = self.roles[role] + self.role_empty_sep
            message_list.append(message_string)

        prompt = system_msg + separators[0] + "".join(message_list)

        # Replace the last function string placeholder with actual function string
        prompt = self.function_string.join(prompt.rsplit(MessagePlaceholders.FUNCTION.value, 1))
        # Replace with remaining function string placeholders with empty string
        prompt = prompt.replace(MessagePlaceholders.FUNCTION.value, "")

        return prompt
