"""Protocols in MLC LLM for OpenAI API.
Adapted from FastChat's OpenAI protocol:
https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py
"""
# pylint: disable=missing-class-docstring
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..serve.config import GenerationConfig

################ Commons ################


class ListResponse(BaseModel):
    object: str = "list"
    data: List[Any]


class LogProbs(BaseModel):
    pass


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
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None


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


# Right now we only have one request protocol.
# When we have more protocols, change it to `Union[CompletionRequest, ...]`.
OpenAIRequestProtocol = CompletionRequest


def openai_api_get_unsupported_fields(request: OpenAIRequestProtocol) -> List[str]:
    """Get the unsupported fields in the request."""
    unsupport_field_values: List[Tuple[str, Any]] = [
        ("best_of", 1),
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
        ("logit_bias", None),
        ("logprobs", None),
        ("n", 1),
        ("seed", None),
    ]

    unsupported_fields: List[str] = []
    for field, value in unsupport_field_values:
        if hasattr(request, field) and getattr(request, field) != value:
            unsupported_fields.append(field)
    return unsupported_fields


def openai_api_get_generation_config(request: OpenAIRequestProtocol) -> GenerationConfig:
    """Create the generation config from the given request."""
    kwargs = {}
    arg_name_mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "max_tokens": "max_new_tokens",
    }
    for openai_name, mlc_name in arg_name_mapping.items():
        kwargs[mlc_name] = getattr(request, openai_name)
    if request.stop is not None:
        kwargs["stop_strs"] = [request.stop] if isinstance(request.stop, str) else request.stop
    return GenerationConfig(**kwargs)
