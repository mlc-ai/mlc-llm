# pylint: disable=missing-docstring,fixme,import-error,too-few-public-methods
"""
Adapted from FastChat's OpenAI protocol:
https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


class ToolCalls(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{shortuuid.random()}")
    type: str = "function"
    function: object


class ChatMessage(BaseModel):
    role: str
    content: Union[str, None]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCalls]] = None


class Function(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: object


class Tools(BaseModel):
    type: Literal["function"]
    function: Dict[str, Any]


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: float = None
    top_p: float = None
    # TODO: replace by presence_penalty and frequency_penalty
    repetition_penalty: float = None
    mean_gen_len: int = None
    # TODO: replace by max_tokens
    max_gen_len: int = None
    presence_penalty: float = None
    frequency_penalty: float = None
    n: int = None
    stop: Union[str, List[str]] = None
    tools: Optional[List[Tools]] = None
    tool_choice: Union[Literal["none", "auto"], ToolChoice] = "auto"
    # TODO: Implement support for the OpenAI API parameters
    # stop: Optional[Union[str, List[str]]] = None
    # max_tokens: Optional[int]
    # logit_bias
    # user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    # TODO: Implement support for the following fields
    usage: Optional[UsageInfo] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseStreamChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: Optional[bool] = False
    temperature: float = None
    repetition_penalty: float = None
    top_p: float = None
    mean_gen_len: int = None
    # TODO: replace by max_tokens
    max_gen_len: int = None
    presence_penalty: float = None
    frequency_penalty: float = None
    n: int = None
    stop: Union[str, List[str]] = None
    # TODO: Implement support for the OpenAI API parameters
    # suffix
    # logprobs
    # echo
    # best_of
    # logit_bias
    # user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = None
    # TODO: logprobs support
    logprobs: Optional[int] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseStreamChoice]


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: Optional[str] = None
    usage: UsageInfo


class VisualStudioCodeCompletionParameters(BaseModel):
    temperature: float = None
    top_p: float = None
    max_new_tokens: int = None


class VisualStudioCodeCompletionRequest(BaseModel):
    inputs: str
    parameters: VisualStudioCodeCompletionParameters


class VisualStudioCodeCompletionResponse(BaseModel):
    generated_text: str
