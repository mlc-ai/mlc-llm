"""
Adapted from FastChat's OpenAI protocol: https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool | None = False
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
    # TODO: Implement support for the OpenAI API parameters
    # function []
    # function_call
    # stop: Optional[Union[str, List[str]]] = None
    # max_tokens: Optional[int]
    # logit_bias
    # user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list[ChatCompletionResponseChoice]
    # TODO: Implement support for the following fields
    usage: UsageInfo | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list[ChatCompletionResponseStreamChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    stream: bool | None = False
    temperature: float = None
    repetition_penalty: float = None
    top_p: float = None
    mean_gen_len: int = None
    # TODO: replace by max_tokens
    max_gen_len: int = None
    presence_penalty: float = None
    frequency_penalty: float = None
    # TODO: Implement support for the OpenAI API parameters
    # suffix
    # max_tokens: Optional[int]
    # n: Optional[int] = 1
    # logprobs
    # echo
    # stop: Optional[Union[str, List[str]]] = None
    # best_of
    # logit_bias
    # user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: int | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
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
