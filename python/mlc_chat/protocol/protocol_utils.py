"""Utility functions for request protocols"""

from typing import List

from pydantic import BaseModel

from ..serve.config import GenerationConfig
from . import RequestProtocol
from .openai_api_protocol import (
    OpenAIRequestProtocol,
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
    if isinstance(request, OpenAIRequestProtocol):
        return openai_api_get_unsupported_fields(request)
    raise RuntimeError("Cannot reach here")


def get_generation_config(request: RequestProtocol) -> GenerationConfig:
    """Create the generation config in MLC LLM out from the input request protocol."""
    if isinstance(request, OpenAIRequestProtocol):
        return openai_api_get_generation_config(request)
    raise RuntimeError("Cannot reach here")
