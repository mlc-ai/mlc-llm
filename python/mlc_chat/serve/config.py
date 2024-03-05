"""Configuration dataclasses used in MLC LLM serving"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class ResponseFormat:
    """The response format dataclass.

    Parameters
    ----------
    type : Literal["text", "json_object"]
        The type of response format. Default: "text".

    json_schema : Optional[str]
        The JSON schema string for the JSON response format. If None, a legal json string without
        special restrictions will be generated.

        Could be specified when the response format is "json_object". Default: None.
    """

    type: Literal["text", "json_object"] = "text"
    json_schema: Optional[str] = None

    def __post_init__(self):
        if self.json_schema is not None and self.type != "json_object":
            raise ValueError("JSON json_schema is only supported in JSON response format")


@dataclass
class GenerationConfig:  # pylint: disable=too-many-instance-attributes
    """The generation configuration dataclass.

    Parameters
    ----------
    n : int
        How many chat completion choices to generate for each input message.

    temperature : float
        The value that applies to logits and modulates the next token probabilities.

    top_p : float
        In sampling, only the most probable tokens with probabilities summed up to
        `top_p` are kept for sampling.

    frequency_penalty : float
        Positive values penalize new tokens based on their existing frequency
        in the text so far, decreasing the model's likelihood to repeat the same
        line verbatim.

    presence_penalty : float
        Positive values penalize new tokens based on whether they appear in the text
        so far, increasing the model's likelihood to talk about new topics.

    repetition_penalty : float
        The penalty term that applies to logits to control token repetition in generation.
        It will be suppressed when any of frequency_penalty and presence_penalty is
        non-zero.

    logprobs : bool
        Whether to return log probabilities of the output tokens or not.
        If true, the log probabilities of each output token will be returned.

    top_logprobs : int
        An integer between 0 and 5 specifying the number of most likely
        tokens to return at each token position, each with an associated
        log probability.
        `logprobs` must be set to True if this parameter is used.

    logit_bias : Optional[Dict[int, float]]
        The bias logit value added to selected tokens prior to sampling.

    max_tokens : Optional[int]
        The maximum number of generated tokens,
        or None, in which case the generation will not stop
        until exceeding model capability or hit any stop criteria.

    seed : Optional[int]
        The random seed of the generation.
        The seed will be a random value if not specified.

    stop_strs : List[str]
        The list of strings that mark the end of generation.

    stop_token_ids : List[int]
        The list of token ids that mark the end of generation.

    ignore_eos: bool
        When it is true, ignore the eos token and generate tokens until `max_tokens`.
        Default is set to False.

    response_format : ResponseFormat
        The response format of the generation output.
    """

    n: int = 1
    temperature: float = 0.8
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = field(default_factory=dict)

    max_tokens: Optional[int] = 128
    seed: Optional[int] = None
    stop_strs: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False

    response_format: ResponseFormat = field(default_factory=ResponseFormat)

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "GenerationConfig":
        """Construct a config from JSON string."""
        return GenerationConfig(**json.loads(json_str))


@dataclass
class KVCacheConfig:
    """The KV cache initialization configuration.

    Parameters
    ----------
    page_size : int
        The number of consecutive tokens handled in each page in paged KV cache.

    max_num_sequence : int
        The maximum number of sequences that are allowed to processed by the KV
        cache at any time.

    max_total_sequence_length : Optional[int]
        The maximum total number of tokens whose KV data are allowed to exist
        in the KV cache at any time.
        Set it to None to enable automatic computation of the max total
        sequence length.

    prefill_chunk_size : Optional[int]
        The maximum total sequence length in a prefill.
        If not specified, it will be automatically inferred from model config.
    """

    page_size: int = 16
    max_num_sequence: int = 32
    max_total_sequence_length: Optional[int] = None
    prefill_chunk_size: Optional[int] = None

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "KVCacheConfig":
        """Construct a config from JSON string."""
        return KVCacheConfig(**json.loads(json_str))


@dataclass
class EngineMode:
    """The Engine execution mode.

    Parameters
    ----------
    enable_speculative : bool
        Whether the speculative decoding mode is enabled, default False.

    spec_draft_length : int
        The number of tokens to generate in speculative proposal (draft), default 4.
    """

    enable_speculative: bool = False
    spec_draft_length: int = 4

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "EngineMode":
        """Construct a config from JSON string."""
        return EngineMode(**json.loads(json_str))
