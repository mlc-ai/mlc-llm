"""The standard conversation protocol in MLC LLM"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

# The system message placeholder in the system message prompt.
SYSTEM_MESSAGE_PLACEHOLDER = "{system_message}"


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
    system_template: str = SYSTEM_MESSAGE_PLACEHOLDER
    # The content of the system prompt (without the template format).
    system_message: str = ""
    # The system token ids to be prepended at the beginning of tokenized
    # generated prompt.
    system_prefix_token_ids: Optional[List[int]] = None

    # The conversation roles.
    roles: Tuple[str, str]
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
        system_msg = self.system_template.replace(SYSTEM_MESSAGE_PLACEHOLDER, self.system_message)

        # - Get the message strings.
        message_list: List[str] = []
        separators = list(self.seps)
        if len(separators) == 1:
            separators.append(separators[0])
        for role, content in self.messages:  # pylint: disable=not-an-iterable
            if role not in self.roles:
                raise ValueError(f'Role "{role}" is not a supported role in {self.roles}')
            separator = separators[role == self.roles[1]]
            if content is not None:
                message_string = role + self.role_content_sep + content + separator
            else:
                message_string = role + self.role_empty_sep
            message_list.append(message_string)

        return system_msg + separators[0] + "".join(message_list)
