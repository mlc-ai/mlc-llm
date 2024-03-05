"""Protocols in MLC LLM for OpenAI API.
Adapted from FastChat's OpenAI protocol:
https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""

# pylint: disable=missing-class-docstring
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import shortuuid
from pydantic import BaseModel, Field, field_validator, model_validator

from mlc_chat.serve.config import ResponseFormat

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
    json_schema: Optional[str] = None


class CompletionRequest(BaseModel):
    """OpenAI completion request protocol.
    API reference: https://platform.openai.com/docs/api-reference/completions/create
    """

    model: str
    prompt: Union[str, List[int], List[Union[str, List[int]]]]
    best_of: int = 1
    echo: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: int = 16
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None
    ignore_eos: bool = False
    response_format: RequestResponseFormat = Field(default_factory=RequestResponseFormat)

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
    model: str
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
    content: Optional[Union[str, List[Dict[str, str]]]] = None
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = None
    tool_calls: Optional[List[ChatToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request protocol.
    API reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: List[ChatCompletionMessage]
    model: str
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: Optional[int] = None
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    tools: Optional[List[ChatTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None
    user: Optional[str] = None
    ignore_eos: bool = False
    response_format: RequestResponseFormat = Field(default_factory=RequestResponseFormat)

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


class ChatCompletionResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None
    index: int = 0
    message: ChatCompletionMessage
    logprobs: Optional[LogProbs] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
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
    model: str
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
    model: str
    system_fingerprint: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"


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
    request: Union[CompletionRequest, ChatCompletionRequest]
) -> Dict[str, Any]:
    """Create the generation config from the given request."""
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
    if kwargs["max_tokens"] is None:
        # Setting to -1 means the generation will not stop until
        # exceeding model capability or hit any stop criteria.
        kwargs["max_tokens"] = -1
    if request.stop is not None:
        kwargs["stop_strs"] = [request.stop] if isinstance(request.stop, str) else request.stop
    kwargs["response_format"] = ResponseFormat(**request.response_format.model_dump())
    return kwargs
