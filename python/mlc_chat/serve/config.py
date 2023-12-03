"""Configuration dataclasses used in MLC LLM serving"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """The generation configuration dataclass.

    Parameters
    ----------
    temperature : float
        The value that applies to logits and modulates the next token probabilities.

    top_p : float
        In sampling, only the most probable tokens with probabilities summed up to
        `top_k` are kept for sampling.

    repetition_penalty : float
        The penalty term that applies to logits to control token repetition in generation.

    max_tokens : Optional[int]
        The maximum number of generated tokens,
        or None, in which case the generation will not stop
        until exceeding model capability or hit any stop criteria.

    stop_strs : List[str]
        The list of strings that mark the end of generation.

    stop_token_ids : List[int]
        The list of token ids that mark the end of generation.
    """

    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0

    max_tokens: Optional[int] = 128
    stop_strs: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)

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

    max_total_sequence_length : int
        The maximum total number of tokens whose KV data are allowed to exist
        in the KV cache at any time.
    """

    page_size: int = 16
    max_num_sequence: int = 32
    max_total_sequence_length: int = 16384

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "KVCacheConfig":
        """Construct a config from JSON string."""
        return KVCacheConfig(**json.loads(json_str))
