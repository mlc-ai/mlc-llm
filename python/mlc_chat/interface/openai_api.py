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
    name: str | None = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool | None = False
    temperature: float = None
    top_p: float = None
    repetition_penalty: float = None
    mean_gen_len: int = None
    max_gen_len: int = None
    # TODO: Implement support for the following fields
    # n: Optional[int] = 1
    # stop: Optional[Union[str, List[str]]] = None
    # presence_penalty: Optional[float] = 0.0
    # frequency_penalty: Optional[float] = 0.0
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
    temperature: float = None
    repetition_penalty: float = None
    top_p: float = None
    mean_gen_len: int = None
    max_gen_len: int = None
    system_prompt: str = None
    chat_roles: List[str] = None
    messages: List[List[str]] = None
    offset: str = None
    separator_style: int = None
    seps: List[str] = None
    role_msg_sep: str = None
    role_empty_sep: str = None
    stop_str: str = None
    stop_tokens: List[int] = None
    add_bos: bool = None

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

class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: Optional[str] = None
    usage: UsageInfo
