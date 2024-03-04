"""Utility functions for request protocols"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..serve.config import GenerationConfig
from . import RequestProtocol
from .openai_api_protocol import ChatCompletionRequest as OpenAIChatCompletionRequest
from .openai_api_protocol import CompletionRequest as OpenAICompletionRequest
from .openai_api_protocol import (
    openai_api_get_generation_config,
    openai_api_get_unsupported_fields,
)


class ErrorResponse(BaseModel):
    """The class of error response."""

    object: str = "error"
    message: str
    code: int = None


def get_unsupported_fields(request: RequestProtocol) -> List[str]:
    """Get the unsupported fields of the request.
    Return the list of unsupported field names.
    """
    if isinstance(request, (OpenAICompletionRequest, OpenAIChatCompletionRequest)):
        return openai_api_get_unsupported_fields(request)
    raise RuntimeError("Cannot reach here")


def get_generation_config(
    request: RequestProtocol,
    extra_stop_token_ids: Optional[List[int]] = None,
    extra_stop_str: Optional[List[str]] = None,
) -> GenerationConfig:
    """Create the generation config in MLC LLM out from the input request protocol."""
    kwargs: Dict[str, Any]
    if isinstance(request, (OpenAICompletionRequest, OpenAIChatCompletionRequest)):
        kwargs = openai_api_get_generation_config(request)
    else:
        raise RuntimeError("Cannot reach here")

    if extra_stop_token_ids is not None:
        stop_token_ids = kwargs.get("stop_token_ids", [])
        assert isinstance(stop_token_ids, list)
        stop_token_ids += extra_stop_token_ids
        kwargs["stop_token_ids"] = stop_token_ids

    if extra_stop_str is not None:
        stop_strs = kwargs.get("stop_strs", [])
        assert isinstance(stop_strs, list)
        stop_strs += extra_stop_str
        kwargs["stop_strs"] = stop_strs

    return GenerationConfig(**kwargs)
