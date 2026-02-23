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
from .debug_protocol import DebugConfig
from .error_protocol import BadRequestError

################ Commons ################


# OPenAI API compatible limits
CHAT_COMPLETION_MAX_TOP_LOGPROBS = 20
COMPLETION_MAX_TOP_LOGPROBS = 5


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


class CompletionLogProbs(BaseModel):
    # The position of the token in the concatenated str: prompt + completion_text
    # TODO(vvchernov): skip optional after support
    text_offset: Optional[List[int]]
    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    extra: Optional[Dict[str, Any]] = None
    """Extra metrics and info that may be returned by debug_config
    """


class StreamOptions(BaseModel):
    include_usage: Optional[bool]


################ v1/embeddings ################


class EmbeddingRequest(BaseModel):
    """OpenAI "v1/embeddings" request protocol.
    API reference: https://platform.openai.com/docs/api-reference/embeddings/create
    """

    input: Union[str, List[str], List[int], List[List[int]]]
    model: Optional[str] = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        """Check that the input is not an empty list.

        Note: empty strings are allowed — encoder models produce valid
        embeddings from [CLS]+[SEP] tokens alone.
        """
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Input list must not be empty.")
        return v


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI "v1/embeddings" response protocol.
    API reference: https://platform.openai.com/docs/api-reference/embeddings/object
    """

    object: str = "list"
    data: List[EmbeddingObject]
    model: Optional[str] = None
    usage: EmbeddingUsage


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
    logprobs: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: Optional[int] = None
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    response_format: Optional[RequestResponseFormat] = None
    debug_config: Optional[DebugConfig] = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def check_penalty_range(cls, penalty_value: Optional[float]) -> Optional[float]:
        """Check if the penalty value is in range [-2, 2]."""
        if penalty_value and (penalty_value < -2 or penalty_value > 2):
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
        if self.logprobs is not None and (
            self.logprobs < 0 or self.logprobs > COMPLETION_MAX_TOP_LOGPROBS
        ):
            raise ValueError(f'"logprobs" must be in range [0, {COMPLETION_MAX_TOP_LOGPROBS}]')
        return self


class CompletionResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "preempt"]] = None
    index: int = 0
    logprobs: Optional[CompletionLogProbs] = None
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
    usage: Optional[CompletionUsage] = None


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
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ChatTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None
    user: Optional[str] = None
    response_format: Optional[RequestResponseFormat] = None
    # NOTE: debug_config is not part of OpenAI protocol
    # we add it to enable extra debug options
    debug_config: Optional[DebugConfig] = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def check_penalty_range(cls, penalty_value: Optional[float]) -> Optional[float]:
        """Check if the penalty value is in range [-2, 2]."""
        if penalty_value and (penalty_value < -2 or penalty_value > 2):
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
        if self.top_logprobs < 0 or self.top_logprobs > CHAT_COMPLETION_MAX_TOP_LOGPROBS:
            raise ValueError(
                f'"top_logprobs" must be in range [0, {CHAT_COMPLETION_MAX_TOP_LOGPROBS}]'
            )
        if not self.logprobs and self.top_logprobs > 0:
            raise ValueError('"logprobs" must be True to support "top_logprobs"')
        return self

    @model_validator(mode="after")
    def check_stream_options(self) -> "ChatCompletionRequest":
        """Check stream options"""
        if self.stream_options is None:
            return self
        if not self.stream:
            raise ValueError("stream must be set to True when stream_options is present")
        return self

    @model_validator(mode="after")
    def check_debug_config(self) -> "ChatCompletionRequest":
        """Check debug config"""
        if self.debug_config is None:
            return self

        if self.debug_config.special_request is None:
            return self

        if not self.stream:
            raise ValueError("DebugConfig.special_request requires stream=True")

        if self.stream_options is None or not self.stream_options.include_usage:
            raise ValueError("DebugConfig.special_request requires include_usage in stream_options")

        return self

    def check_message_validity(self) -> None:
        """Check if the given chat messages are valid. Return error message if invalid."""
        for i, message in enumerate(self.messages):
            if message.role == "system" and i != 0:
                raise BadRequestError(
                    f"System prompt at position {i} in the message list is invalid."
                )
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
                    conv_template.function_string = tool.function.model_dump_json(by_alias=True)
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
            function_list.append(tool.function.model_dump(by_alias=True))

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
    usage: Optional[CompletionUsage] = None


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
    usage: Optional[CompletionUsage] = None


################################################


def openai_api_get_unsupported_fields(
    request: Union[CompletionRequest, ChatCompletionRequest],
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
