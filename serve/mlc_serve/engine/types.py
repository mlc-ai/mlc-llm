from dataclasses import dataclass
from typing import Dict, Literal, Optional, Protocol


@dataclass
class Request:
    prompt: str
    temperature: float
    max_tokens: Optional[int] = None


RequestId = int


@dataclass
class TextGenerationOutput:
    request_id: int
    delta: str
    finish_reason: Optional[Literal["stop", "length"]] = None


@dataclass
class TextGenerationError:
    request_id: int
    type: str
    message: str


@dataclass
class InferenceStepResult:
    outputs: list[TextGenerationOutput]
    errors: list[TextGenerationError]


class LMInferenceEngine(Protocol):
    def add(self, requests: list[Request]) -> list[int]:
        """
        Add requests to the queue of InferenceEngine.

        Returns a list of request id, that should be used to relate the output
        of `step` back to each request
        """

    def cancel(self, request_id: int):
        """
        Cancel the generation of a request.
        """

    def step(self) -> InferenceStepResult:
        """
        InferenceResult contains the next token for processed results,
        and indicates whether the generation for a request is finished.

        It's up to the InferenceEngine to choose which requests
        to work on, while it should be guaranteed that all requests will be
        processed eventually.
        """
