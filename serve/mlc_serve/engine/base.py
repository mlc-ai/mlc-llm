from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from typing import List, Callable, Any, Optional, Dict
import inspect

from .sampling_params import SamplingParams, SamplingType

RequestId = str

# TODO(@sunggg): consider transition to something like Pydantic
@dataclass
class MLCServeEngineConfig:
    # The maximum number of tokens in the batch.
    # TODO(@sunggg): figure out better defaults
    use_staging_engine: bool = True
    # we don't generally expect users to set `max_num_batched_tokens` directly
    # since it is less intuitive.
    # instead, we expose `max_input_len` and `max_num_sequences`
    # so that `max_num_batched_tokens` can be deduced by the following equation.
    # -> `max_num_batched_tokens` = `max_input_len`*`max_num_sequences`
    max_input_len: int = 512
    max_num_sequences: int = 8
    max_num_batched_tokens: int = -1
    min_decode_steps: int = 32
    max_decode_steps: int = 48
    prompt_allocate_ratio: float = 2.0

    @classmethod
    def _from_json(config_cls, json_obj: Dict[Any, Any]):
        return config_cls(
            **{
                k: v
                for k, v in json_obj.items()
                if k in inspect.signature(config_cls).parameters
            }
        )

def get_engine_config(dict_config, enable_check = True):
    engine_config = MLCServeEngineConfig._from_json(dict_config)
    # Checks to make sure engine configs are set correctly
    # since engine config is critical to the performance
    if enable_check:
        assert isinstance(engine_config.use_staging_engine, bool)
        assert isinstance(engine_config.max_num_batched_tokens, int)
        assert isinstance(engine_config.max_input_len, int)
        assert isinstance(engine_config.max_num_sequences, int)
        assert isinstance(engine_config.max_decode_steps, int)
        assert isinstance(engine_config.min_decode_steps, int)
        assert isinstance(engine_config.prompt_allocate_ratio, float)

        # TODO(@sunggg): engine allows -1 for these params. figure out the behavior and enable checks properly
        assert engine_config.max_num_batched_tokens == -1, \
            "`max_num_batched_tokens` is not supposed to be configured directly. \
              Use `max_num_sequences` and `max_input_len` instead."
        assert engine_config.max_input_len > 0
        assert engine_config.max_num_sequences > 0
        engine_config.max_num_batched_tokens = engine_config.max_num_sequences * engine_config.max_input_len

        assert (engine_config.min_decode_steps > 0) and (engine_config.max_decode_steps > 0)
        assert engine_config.max_decode_steps > engine_config.min_decode_steps
        assert engine_config.prompt_allocate_ratio > 0

    return engine_config

@dataclass
class StoppingCriteria:
    """
    Parameters about when to stop text generation.
    """
    max_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None

@dataclass
class ChatMessage:
    role: str
    content: Optional[str]


@dataclass
class DebugOptions:
    ignore_eos: bool = False
    # Override messages with a single prompt, skipping conversation template
    prompt: Optional[str] = None


class FinishReason(Enum):
    Stop = "stop"
    Length = "length"
    Cancelled = "cancelled"

# A single token.
Token = List[int]

@dataclass
class ValidationError:
    msg: str

# The type signature of the token validation callback.
ValidateTokensCallback = Callable[["Request", List[Token]], ValidationError]

@dataclass
class Request:
    request_id: RequestId
    messages: List[ChatMessage]
    # Number of sequences to generate
    num_sequences: int = 1
    # TODO: should `best_of` be handled in the serving layer?
    best_of: Optional[int] = None
    # Options for sampling.
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    # Options for stopping.
    stopping_criteria: StoppingCriteria = field(default_factory=lambda: StoppingCriteria())
    # Options for debugging.
    debug_options: DebugOptions = field(default_factory=DebugOptions)
    # Perform request validation post-tokenization, used by the HTTP layer to control validation.
    validate_tokens: Optional[ValidateTokensCallback] = None
    # Context variables to attach to logging.
    contextvars: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.best_of is None:
            self.best_of = self.num_sequences
        if self.num_sequences < 1:
            raise ValueError(
                f"num_sequences must be at least 1, got {self.num_sequences}."
            )
        if self.best_of < self.num_sequences:
            raise ValueError(
                f"best_of must be greater than or equal to num_sequences, "
                f"got n={self.num_sequences} and best_of={self.best_of}."
            )
        if (
            self.best_of > 1
            and self.sampling_params.sampling_type == SamplingType.GREEDY
        ):
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )


@dataclass
class SequenceOutput:
    index: int
    delta: Optional[str] = None
    # If finish_reason is not None, delta should be None.
    finish_reason: Optional[FinishReason] = None
    # Number of generated tokens so far
    num_generated_tokens: int = 0

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    request_id: RequestId
    sequences: list[SequenceOutput]
    # TODO: reconsider the place to put this number
    # Only set for outputs with valid sequence otuputs
    num_prompt_tokens: Optional[int] = None
    # TODO(@jroesch): We should generalize the type here so we are allowed to return more structured information
    # for logging/user output.
    #
    # Right now I am abusing dynamic typing by putting the ValidationError in here.
    # I would prefer to unblock ourselves then figure this one out right now
    error: Optional[str] = None

    @property
    def is_finished(self) -> bool:
        return self.error is not None or all(seq.is_finished for seq in self.sequences)


@dataclass
class InferenceStepResult:
    outputs: list[RequestOutput]


class InferenceEngine(ABC):
    @abstractmethod
    def add(self, requests: list[Request]) -> None:
        """
        Add requests to the InferenceEngine.

        Requests will be processed when `step` is called, if there is capacity.
        Requests will be handled on a first-in, first-out basis.
        """
        ...

    @abstractmethod
    def cancel(self, request_id: RequestId) -> None:
        """
        Cancel the generation of a request.

        The next call to `step` will return TextGenerationOutput for cancelled requests.
        The output will contain empty delta and finish reason `cancelled`.
        """
        ...

    @abstractmethod
    def has_pending_requests(self) -> bool:
        """
        Check if there is pending requests in the engine.
        """
        ...

    @abstractmethod
    def wait_for_request(self, timeout_seconds=None) -> bool:
        """
        Block until there is request to process.
        It can also be given a timeout_seconds parameter, allowing it to return even if
        no requests are coming in. The return value is a boolean that indicates whether
        there are requests when it's returned.
        """
        ...

    @abstractmethod
    def step(self) -> InferenceStepResult:
        """
        Perform a single inference step. In general it will generates one token for each
        requests that are being processed, but it could generate more if speculative decoding
        is used.

        If the engine has no requests in the queue, `step` will return immediately with
        an empty `InferenceStepResult.outputs`.
        """
        ...


class ScopedInferenceEngine(InferenceEngine):
    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


@dataclass
class RequestState:
    """
    The internal state of request in the InferenceEngine.
    """

    request_id: RequestId
    token_ids: list[int]
    output_text: str
    prompt_len: int
    next_start_position: int
    sampling_params: SamplingParams
    stopping_criteria: StoppingCriteria
    debug_options: DebugOptions
    is_ended: bool = False
    validation_err: Optional[ValidationError] = None

def check_stopping_sequences(stopping_criteria, output_text, delta, is_ended):
    if stopping_criteria.stop_sequences:
        for t in stopping_criteria.stop_sequences:
            if t in output_text:
                # since search pattern can include only part of the new generated token,
                # we need to trim generated string
                # for example, we have "I " in the stopping criteria, previously existed
                # output_text had "I" and new coming token "am" would add space before the word
                # thus final output_text would have "I am" before verification on stop sequence
                # While eventually we need to return "I "
                if not output_text.endswith(t):
                    sub_index = output_text.find(t)
                    delta = delta[:-(len(output_text) - sub_index - len(t))]
                    output_text = output_text[:output_text.find(t) + len(t)]
                is_ended = True
                break
    return output_text, delta, is_ended
