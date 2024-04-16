"""Configuration dataclasses used in MLC LLM serving"""

import argparse
import enum
import json
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Dict, List, Literal, Optional


@dataclass
class ResponseFormat:
    """The response format dataclass.

    Parameters
    ----------
    type : Literal["text", "json_object"]
        The type of response format. Default: "text".

    schema : Optional[str]
        The JSON schema string for the JSON response format. If None, a legal json string without
        special restrictions will be generated.

        Could be specified when the response format is "json_object". Default: None.
    """

    type: Literal["text", "json_object"] = "text"
    schema: Optional[str] = None

    def __post_init__(self):
        if self.schema is not None and self.type != "json_object":
            raise ValueError("JSON schema is only supported in JSON response format")


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


class SpeculativeMode(enum.IntEnum):
    """The speculative mode."""

    DISABLE = 0
    SMALL_DRAFT = 1
    EAGLE = 2


@dataclass
class EngineConfig:
    """The class of Engine execution configuration.

    Parameters
    ----------
    spec_draft_length : int
        The number of tokens to generate in speculative proposal (draft), default 4.

    speculative_mode: SpeculativeMode
        The speculative mode.
    """

    spec_draft_length: int = 4
    speculative_mode: SpeculativeMode = SpeculativeMode.DISABLE

    def __repr__(self) -> str:
        out = StringIO()
        print(f"spec_draft_length={self.spec_draft_length}", file=out, end="")
        print(f";speculative_mode={self.speculative_mode.name}", file=out, end="")
        return out.getvalue().rstrip()

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        dt = asdict(self)
        dt["speculative_mode"] = int(self.speculative_mode)
        return json.dumps(dt)

    @staticmethod
    def from_json(json_str: str) -> "EngineConfig":
        """Construct a config from JSON string."""
        return EngineConfig(**json.loads(json_str))

    @staticmethod
    def from_str(source: str) -> "EngineConfig":
        """Parse engine config from a string."""

        parser = argparse.ArgumentParser(description="optimization flags")
        parser.add_argument("--spec_draft_length", type=int, default=4)
        parser.add_argument(
            "--speculative_mode",
            type=str,
            choices=["DISABLE", "SMALL_DRAFT", "EAGLE"],
            default="DISABLE",
        )
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return EngineConfig(
            spec_draft_length=results.spec_draft_length,
            speculative_mode=SpeculativeMode[results.speculative_mode],
        )
