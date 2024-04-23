"""Protocols in MLC LLM for OpenAI API.
Adapted from FastChat's OpenAI protocol:
https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""

# pylint: disable=missing-class-docstring

import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import shortuuid
from pydantic import BaseModel, Field, field_validator, model_validator

from .conversation_protocol import Conversation
from .error_protocol import BadRequestError

################ Commons ################


class ListResponse(BaseModel):
    object: str = "list"
    data: List[Any]


class TopLogProbs(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]


class LogProbsContent(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[TopLogProbs] = []


class LogProbs(BaseModel):
    content: List[LogProbsContent]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        super().__init__(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


################ v1/models ################


class ModelResponse(BaseModel):
    """OpenAI "v1/models" response protocol.
    API reference: https://platform.openai.com/docs/api-reference/models/object
    """

    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str = "model"
    owned_by: str = "MLC-LLM"


################ v1/completions ################


class RequestResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"
    json_schema: Optional[str] = Field(default=None, alias="schema")
    """This field is named json_schema instead of schema because BaseModel defines a method called
    schema. During construction of RequestResponseFormat, key "schema" still should be used:
    `RequestResponseFormat(type="json_object", schema="{}")`
    """


class CompletionRequest(BaseModel):
    """OpenAI completion request protocol.
    API reference: https://platform.openai.com/docs/api-reference/completions/create
    """

    model: Optional[str] = None
    prompt: Union[str, List[int]]
    best_of: int = 1
    echo: bool = False
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: int = 16
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    ignore_eos: bool = False
    response_format: Optional[RequestResponseFormat] = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def check_penalty_range(cls, penalty_value: float) -> float:
        """Check if the penalty value is in range [-2, 2]."""
        if penalty_value < -2 or penalty_value > 2:
            raise ValueError("Penalty value should be in range [-2, 2].")
        return penalty_value

    @field_validator("logit_bias")
    @classmethod
    def check_logit_bias(
        cls, logit_bias_value: Optional[Dict[int, float]]
    ) -> Optional[Dict[int, float]]:
        """Check if the logit bias key is given as an integer."""
        if logit_bias_value is None:
            return None
        for token_id, bias in logit_bias_value.items():
            if abs(bias) > 100:
                raise ValueError(
                    "Logit bias value should be in range [-100, 100], while value "
                    f"{bias} is given for token id {token_id}"
                )
        return logit_bias_value

    @model_validator(mode="after")
    def check_logprobs(self) -> "CompletionRequest":
        """Check if the logprobs requirements are valid."""
        if self.top_logprobs < 0 or self.top_logprobs > 5:
            raise ValueError('"top_logprobs" must be in range [0, 5]')
        if not self.logprobs and self.top_logprobs > 0:
            raise ValueError('"logprobs" must be True to support "top_logprobs"')
        return self


class CompletionResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length"]] = None
    index: int = 0
    logprobs: Optional[LogProbs] = None
    text: str


class CompletionResponse(BaseModel):
    """OpenAI completion response protocol.
    API reference: https://platform.openai.com/docs/api-reference/completions/object
    """

    id: str
    choices: List[CompletionResponseChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    object: str = "text_completion"
    usage: UsageInfo = Field(
        default_factory=lambda: UsageInfo()  # pylint: disable=unnecessary-lambda
    )


################ v1/chat/completions ################


class ChatFunction(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: Dict


class ChatTool(BaseModel):
    type: Literal["function"]
    function: ChatFunction


class ChatFunctionCall(BaseModel):
    name: str
    arguments: Union[None, Dict[str, Any]] = None


class ChatToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{shortuuid.random()}")
    type: Literal["function"]
    function: ChatFunctionCall


class ChatCompletionMessage(BaseModel):
    content: Optional[Union[str, List[Dict]]] = None
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = None
    tool_calls: Optional[List[ChatToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request protocol.
    API reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: List[ChatCompletionMessage]
    model: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: Optional[int] = None
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ChatTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None
    user: Optional[str] = None
    ignore_eos: bool = False
    response_format: Optional[RequestResponseFormat] = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def check_penalty_range(cls, penalty_value: float) -> float:
        """Check if the penalty value is in range [-2, 2]."""
        if penalty_value < -2 or penalty_value > 2:
            raise ValueError("Penalty value should be in range [-2, 2].")
        return penalty_value

    @field_validator("logit_bias")
    @classmethod
    def check_logit_bias(
        cls, logit_bias_value: Optional[Dict[int, float]]
    ) -> Optional[Dict[int, float]]:
        """Check if the logit bias key is given as an integer."""
        if logit_bias_value is None:
            return None
        for token_id, bias in logit_bias_value.items():
            if abs(bias) > 100:
                raise ValueError(
                    "Logit bias value should be in range [-100, 100], while value "
                    f"{bias} is given for token id {token_id}"
                )
        return logit_bias_value

    @model_validator(mode="after")
    def check_logprobs(self) -> "ChatCompletionRequest":
        """Check if the logprobs requirements are valid."""
        if self.top_logprobs < 0 or self.top_logprobs > 5:
            raise ValueError('"top_logprobs" must be in range [0, 5]')
        if not self.logprobs and self.top_logprobs > 0:
            raise ValueError('"logprobs" must be True to support "top_logprobs"')
        return self

    def check_message_validity(self) -> None:
        """Check if the given chat messages are valid. Return error message if invalid."""
        for i, message in enumerate(self.messages):
            if message.role == "system" and i != 0:
                raise BadRequestError(
                    f"System prompt at position {i} in the message list is invalid."
                )
            if message.role == "tool":
                raise BadRequestError("Tool as the message author is not supported yet.")
            if message.tool_call_id is not None:
                if message.role != "tool":
                    raise BadRequestError("Non-tool message having `tool_call_id` is invalid.")
            if isinstance(message.content, list):
                if message.role != "user":
                    raise BadRequestError("Non-user message having a list of content is invalid.")
            if message.tool_calls is not None:
                if message.role != "assistant":
                    raise BadRequestError("Non-assistant message having `tool_calls` is invalid.")
                raise BadRequestError("Assistant message having `tool_calls` is not supported yet.")

    def check_function_call_usage(self, conv_template: Conversation) -> None:
        """Check if function calling is used and update the conversation template.
        Return error message if invalid request format for function calling.
        """

        # return if no tools are provided or tool_choice is set to none
        if self.tools is None or (isinstance(self.tool_choice, str) and self.tool_choice == "none"):
            conv_template.use_function_calling = False
            return

        # select the tool based on the tool_choice if specified
        if isinstance(self.tool_choice, dict):
            if self.tool_choice["type"] != "function":  # pylint: disable=unsubscriptable-object
                raise BadRequestError("Only 'function' tool choice is supported")

            if len(self.tool_choice["function"]) > 1:  # pylint: disable=unsubscriptable-object
                raise BadRequestError("Only one tool is supported when tool_choice is specified")

            for tool in self.tools:  # pylint: disable=not-an-iterable
                if (
                    tool.function.name
                    == self.tool_choice["function"][  # pylint: disable=unsubscriptable-object
                        "name"
                    ]
                ):
                    conv_template.use_function_calling = True
                    conv_template.function_string = tool.function.model_dump_json()
                    return

            # pylint: disable=unsubscriptable-object
            raise BadRequestError(
                f"The tool_choice function {self.tool_choice['function']['name']}"
                " is not found in the tools list"
            )
            # pylint: enable=unsubscriptable-object

        if isinstance(self.tool_choice, str) and self.tool_choice != "auto":
            raise BadRequestError(f"Invalid tool_choice value: {self.tool_choice}")

        function_list = []
        for tool in self.tools:  # pylint: disable=not-an-iterable
            if tool.type != "function":
                raise BadRequestError("Only 'function' tool type is supported")
            function_list.append(tool.function.model_dump())

        conv_template.use_function_calling = True
        conv_template.function_string = json.dumps(function_list)


class ChatCompletionResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None
    index: int = 0
    message: ChatCompletionMessage
    logprobs: Optional[LogProbs] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None
    index: int = 0
    delta: ChatCompletionMessage
    logprobs: Optional[LogProbs] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI completion response protocol.
    API reference: https://platform.openai.com/docs/api-reference/chat/object
    """

    id: str
    choices: List[ChatCompletionResponseChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    system_fingerprint: str
    object: Literal["chat.completion"] = "chat.completion"
    usage: UsageInfo = Field(
        default_factory=lambda: UsageInfo()  # pylint: disable=unnecessary-lambda
    )


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI completion stream response protocol.
    API reference: https://platform.openai.com/docs/api-reference/chat/streaming
    """

    id: str
    choices: List[ChatCompletionStreamResponseChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    system_fingerprint: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: UsageInfo = Field(
        default_factory=lambda: UsageInfo()  # pylint: disable=unnecessary-lambda
    )


################################################


def openai_api_get_unsupported_fields(
    request: Union[CompletionRequest, ChatCompletionRequest]
) -> List[str]:
    """Get the unsupported fields in the request."""
    unsupported_field_default_values: List[Tuple[str, Any]] = [
        ("best_of", 1),
    ]

    unsupported_fields: List[str] = []
    for field, value in unsupported_field_default_values:
        if hasattr(request, field) and getattr(request, field) != value:
            unsupported_fields.append(field)
    return unsupported_fields


def openai_api_get_generation_config(
    request: Union[CompletionRequest, ChatCompletionRequest], model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create the generation config from the given request."""
    from ..serve.config import ResponseFormat  # pylint: disable=import-outside-toplevel

    kwargs: Dict[str, Any] = {}
    arg_names = [
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "seed",
        "ignore_eos",
    ]
    for arg_name in arg_names:
        kwargs[arg_name] = getattr(request, arg_name)

    # If per-request generation config values are missing, try loading from model config.
    # If still not found, then use the default OpenAI API value
    if kwargs["temperature"] is None:
        kwargs["temperature"] = model_config.get("temperature", 1.0)
    if kwargs["top_p"] is None:
        kwargs["top_p"] = model_config.get("top_p", 1.0)
    if kwargs["frequency_penalty"] is None:
        kwargs["frequency_penalty"] = model_config.get("frequency_penalty", 0.0)
    if kwargs["presence_penalty"] is None:
        kwargs["presence_penalty"] = model_config.get("presence_penalty", 0.0)
    if kwargs["max_tokens"] is None:
        # Setting to -1 means the generation will not stop until
        # exceeding model capability or hit any stop criteria.
        kwargs["max_tokens"] = -1
    if request.stop is not None:
        kwargs["stop_strs"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.response_format is not None:
        kwargs["response_format"] = ResponseFormat(
            **request.response_format.model_dump(by_alias=True)
        )
    return kwargs
