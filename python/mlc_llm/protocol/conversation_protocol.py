"""The standard conversation protocol in MLC LLM"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field, field_validator


# The message placeholders in the message prompts according to roles.
class MessagePlaceholders(Enum):
    """The message placeholders in the message prompts according to roles."""

    SYSTEM = "{system_message}"
    USER = "{user_message}"
    ASSISTANT = "{assistant_message}"
    TOOL = "{tool_message}"
    FUNCTION = "{function_string}"


T = TypeVar("T", bound="BaseModel")


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
    # Whether or not to append user role and separator after the system message.
    # This is mainly for [INST] [/INST] style prompt format
    add_role_after_system_message: bool = True

    # The conversation roles
    roles: Dict[str, str]

    # The roles prompt template, it optionally contains the defaults
    # message placeholders and will be replaced by actual content
    role_templates: Dict[str, str]

    # The conversation history messages.
    # Each message is a pair of strings, denoting "(role, content)".
    # The content can be None.
    messages: List[Tuple[str, Optional[Union[str, List[Dict]]]]] = Field(default_factory=lambda: [])

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

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to a json dictionary"""
        return self.model_dump(by_alias=True, exclude_none=True)

    @classmethod
    def from_json_dict(cls: Type[T], json_dict: Dict[str, Any]) -> T:
        """Convert from a json dictionary"""
        return Conversation.model_validate(json_dict)

    # pylint: disable=too-many-branches
    def as_prompt(self, config=None) -> List[Any]:
        """Convert the conversation template and history messages to
        a single prompt.

        Returns
        -------
        prompts : List[Union[str, "mlc_llm.serve.data.Data"]]
            The prompts converted from the conversation messages.
            We use Any in the signature to avoid cyclic import.
        """
        from ..serve import data  # pylint: disable=import-outside-toplevel

        # - Get the system message.
        system_msg = self.system_template.replace(
            MessagePlaceholders.SYSTEM.value, self.system_message
        )

        # - Get the message strings.
        message_list: List[Union[str, data.Data]] = []
        separators = list(self.seps)
        if len(separators) == 1:
            separators.append(separators[0])

        if system_msg != "":
            message_list.append(system_msg)

        for i, (role, content) in enumerate(self.messages):  # pylint: disable=not-an-iterable
            if role not in self.roles.keys():
                raise ValueError(f'Role "{role}" is not a supported role in {self.roles.keys()}')
            separator = separators[role == "assistant"]  # check assistant role

            if content is None:
                message_list.append(self.roles[role] + self.role_empty_sep)
                continue

            role_prefix = (
                ""
                # Do not append role prefix if this is the first message and there
                # is already a system message
                if (not self.add_role_after_system_message and system_msg != "" and i == 0)
                else self.roles[role] + self.role_content_sep
            )
            if isinstance(content, str):
                message_list.append(
                    role_prefix
                    + self.role_templates[role].replace(
                        MessagePlaceholders[role.upper()].value, content
                    )
                    + separator
                )
                continue

            message_list.append(role_prefix)

            for item in content:
                assert isinstance(item, dict), "Content should be a string or a list of dicts"
                assert "type" in item, "Content item should have a type field"
                if item["type"] == "text":
                    message = self.role_templates[role].replace(
                        MessagePlaceholders[role.upper()].value, item["text"]
                    )
                    message_list.append(message)
                elif item["type"] == "image_url":
                    assert config is not None, "Model config is required"
                    image_url = _get_url_from_item(item)
                    message_list.append(data.ImageData.from_url(image_url, config))
                    message_list.append("\n")
                else:
                    raise ValueError(f"Unsupported content type: {item['type']}")

            message_list.append(separator)

        prompt = _combine_consecutive_messages(message_list)

        if not any(isinstance(item, data.ImageData) for item in message_list):
            # Replace the last function string placeholder with actual function string
            prompt[0] = self.function_string.join(
                prompt[0].rsplit(MessagePlaceholders.FUNCTION.value, 1)
            )
            # Replace with remaining function string placeholders with empty string
            prompt[0] = prompt[0].replace(MessagePlaceholders.FUNCTION.value, "")

        return prompt


def _get_url_from_item(item: Dict) -> str:
    image_url: str
    assert "image_url" in item, "Content item should have an image_url field"
    if isinstance(item["image_url"], str):
        image_url = item["image_url"]
    elif isinstance(item["image_url"], dict):
        assert (
            "url" in item["image_url"]
        ), "Content image_url item should be a string or a dict with a url field"  # pylint: disable=line-too-long
        image_url = item["image_url"]["url"]
    else:
        raise ValueError(
            "Content image_url item type not supported. "
            "Should be a string or a dict with a url field."
        )
    return image_url


def _combine_consecutive_messages(messages: List[Any]) -> List[Any]:
    """Combining consecutive strings into one.

    Parameters
    ----------
    messages : List[Union[str, "mlc_llm.serve.data.Data"]]
        The input messages to be combined.
        We use Any in the signature to avoid cyclic import.

    Returns
    -------
    updated_messages : List[Union[str, "mlc_llm.serve.data.Data"]]
        The combined messages
    """
    if len(messages) == 0:
        return []

    combined_messages = [messages[0]]
    for message in messages[1:]:
        if isinstance(message, str) and isinstance(combined_messages[-1], str):
            combined_messages[-1] += message
        else:
            combined_messages.append(message)
    return combined_messages
