from dataclasses import dataclass
from typing import Dict, Literal, Optional, Protocol


@dataclass
class SamplingParams:
    temperature: float


@dataclass
class StoppingCriteria:
    max_tokens: Optional[int]


@dataclass
class Request:
    prompt: str
    sampling_params: SamplingParams
    stopping_criteria: StoppingCriteria


RequestId = str


@dataclass
class TextGenerationOutput:
    request_id: RequestId
    delta: str
    # TODO: make this enum
    finish_reason: Optional[Literal["stop", "length", "cancelled"]] = None


@dataclass
class TextGenerationError:
    request_id: RequestId
    type: str
    message: str


@dataclass
class InferenceStepResult:
    outputs: list[TextGenerationOutput]
    errors: list[TextGenerationError]


class InferenceEngine(Protocol):
    def add(self, requests: list[Request]) -> list[RequestId]:
        """
        Add requests to the queue of InferenceEngine.

        Returns a list of request id, that should be used to relate the output
        of `step` back to each request
        """

    def cancel(self, request_id: RequestId):
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

        If the engine has no requests in the queue, `step` will block until there is
        request coming in.
        """


@dataclass
class SequenceGenerationRequest:
    request_id: RequestId
    # prompts or the last token(s)
    token_ids: list[int]
    start_position: int
    sampling_params: SamplingParams

    # Extension for multi-modal model
    # class ImagePayload:
    #   pre_image_tokens: List[int]
    #   image: bytes
    #   post_image_tokens: List[int]


@dataclass
class SequenceGenerationResponse:
    request_id: RequestId
    # for most cases, there should be only one token returned
    # making this a list of token ids to leave room for speculative decoding
    token_ids: list[int]
    error: Optional[str]


class ModelExecutor(Protocol):
    def generate(
        self, requests: list[SequenceGenerationRequest]
    ) -> list[SequenceGenerationResponse]:
        """
        A unified entrypoint for inference.

        `requests` could contain both new requests to prefill and existing request to decode.
        It's up to the ModelExecutor to decide how the actual inference is performed.
        It could do one prefill pass and one decode pass, or do one single pass internally.

        ModelExecutor could evict keys and values from KVCache for requests that's not in
        the `requests`.
        """

    def allocate(self, request_id: RequestId, num_tokens: int):
        """
        Allocate cache space for request, raise error if there is no space.
        """

    # TODO: Can this be merged with `allocate`?
    def extend(self, request_id: RequestId, new_tokens: int):
        """
        Extend cache space for request, raise error if there is no space.
        """

    def free(self, request_id: RequestId):
        """
        Free cache space for request.
        """

    def get_kv_cache_size(self) -> int:
        """
        Return the size of KV cache in number of tokens.
        """

    def get_free_space(self) -> int:
        """
        Query available space in the cache.
        Return number of tokens that can be allocated for new request.

        For paged KV cache, this ignores the remaining tokens in pages allocated
        for existing requests, since they cannot be used for the new request.
        """

    def get_max_new_tokens(self) -> int:
        """
        Query the maximum number of new tokens that can be extended for all requests.

        The returned number indicates the max number of forward passes that can be run,
        assuming no requests finish during this process.

        It should return cache size if there is no requests in the cache.
        """


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        pass
