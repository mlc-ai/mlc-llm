"""
Adapted from FastChat's OpenAI protocol: https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""

from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import shortuuid
import time


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: Optional[bool] = False
    # TODO: Implement support for the following fields
    # temperature: Optional[float] = 1.0
    # top_p: Optional[float] = 1.0
    # n: Optional[int] = 1
    # stop: Optional[Union[str, List[str]]] = None
    # max_tokens: Optional[int] = None
    # presence_penalty: Optional[float] = 0.0
    # frequency_penalty: Optional[float] = 0.0
    # user: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    # TODO: Implement support for the following fields
    usage: Optional[UsageInfo]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseStreamChoice]

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]]

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: Optional[str] = None
    usage: UsageInfo
