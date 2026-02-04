"""Error protocols in MLC LLM"""

from http import HTTPStatus
from typing import Optional

import fastapi
from pydantic import BaseModel


class BadRequestError(ValueError):
    """The exception for bad requests in engines."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ErrorResponse(BaseModel):
    """The class of error response."""

    object: str = "error"
    message: str
    code: Optional[int] = None


def create_error_response(status_code: HTTPStatus, message: str) -> fastapi.responses.JSONResponse:
    """Create a JSON response that reports error with regarding the input message."""
    return fastapi.responses.JSONResponse(
        ErrorResponse(message=message, code=status_code.value).model_dump_json(by_alias=True),
        status_code=status_code.value,
    )


async def bad_request_error_handler(_request: fastapi.Request, e: BadRequestError):
    """The handler of BadRequestError that converts an exception into error response."""
    return create_error_response(status_code=HTTPStatus.BAD_REQUEST, message=e.args[0])
