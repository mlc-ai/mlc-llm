"""Configuration dataclasses used in MLC LLM serving"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union


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
class DebugConfig:
    """The debug configuration dataclass.Parameters
    ----------

    pinned_system_prompt : bool
        Whether the input and generated data pinned in engine. Default is set to False.
        This can be used for system prompt or other purpose, if the data is aimed to be
        kept all the time.
    """

    pinned_system_prompt: bool = False


@dataclass
class GenerationConfig:  # pylint: disable=too-many-instance-attributes
    """The generation configuration dataclass.

    Parameters
    ----------
    n : int
        How many chat completion choices to generate for each input message.

    temperature : Optional[float]
        The value that applies to logits and modulates the next token probabilities.

    top_p : Optional[float]
        In sampling, only the most probable tokens with probabilities summed up to
        `top_p` are kept for sampling.

    frequency_penalty : Optional[float]
        Positive values penalize new tokens based on their existing frequency
        in the text so far, decreasing the model's likelihood to repeat the same
        line verbatim.

    presence_penalty : Optional[float]
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

    response_format : ResponseFormat
        The response format of the generation output.

    debug_config : Optional[DebugConfig]
        The optional debug configuration.
    """

    n: int = 1
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: float = 1.0
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: Optional[Dict[int, float]] = field(default_factory=dict)

    max_tokens: Optional[int] = 128
    seed: Optional[int] = None
    stop_strs: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)

    response_format: ResponseFormat = field(default_factory=ResponseFormat)

    debug_config: Optional[DebugConfig] = field(default_factory=DebugConfig)

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "GenerationConfig":
        """Construct a config from JSON string."""
        return GenerationConfig(**json.loads(json_str))


@dataclass
class EngineConfig:  # pylint: disable=too-many-instance-attributes
    """The class of MLCEngine execution configuration.

    Parameters
    ----------
    model : str
        The path to the model directory.

    model_lib : str
        The path to the model library.

    additional_models : List[Union[str, Tuple[str, str]]]
        The paths to the additional models' directories (and model libraries).
        Each element is a single string (denoting the model directory)
        or a tuple of two strings (denoting the model directory and model lib path).

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_num_sequence", "max_total_sequence_length"
        and "prefill_chunk_size" when they are not explicitly specified.
        1. Mode "local" refers to the local server deployment which has low
        request concurrency. So the max batch size will be set to 4, and max
        total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        2. Mode "interactive" refers to the interactive use of server, which
        has at most 1 concurrent request. So the max batch size will be set to 1,
        and max total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        3. Mode "server" refers to the large server use case which may handle
        many concurrent request and want to use GPU memory as much as possible.
        In this mode, we will automatically infer the largest possible max batch
        size and max total sequence length.

        You can manually specify arguments "max_num_sequence", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    gpu_memory_utilization : float
        A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
        It is used to infer to maximum possible KV cache capacity.
        When it is unspecified, it defaults to 0.85.
        Under mode "local" or "interactive", the actual memory usage may be
        significantly smaller than this number. Under mode "server", the actual
        memory usage may be slightly larger than this number.

    kv_cache_page_size : int
        The number of consecutive tokens handled in each page in paged KV cache.

    max_num_sequence : Optional[int]
        The maximum number of sequences that are allowed to be
        processed by the KV cache at any time.

    max_total_sequence_length : Optional[int]
        The maximum total number of tokens whose KV data are allowed
        to exist in the KV cache at any time.

    max_single_sequence_length : Optional[int]
        The maximum length allowed for a single sequence in the engine.

    prefill_chunk_size : Optional[int]
        The maximum total sequence length in a prefill.

    max_history_size: Optional[int]
        The maximum history size for RNN state to rool back.

    kv_state_kind: Optional[Literal["kv_cache", "rnn_state"]]
        The kind of cache.

    speculative_mode : Literal["disable", "small_draft", "eagle", "medusa"]
        The speculative mode.
        "disable" means speculative decoding is disabled.
        "small_draft" means the normal speculative decoding (small draft) mode.
        "eagle" means the eagle-style speculative decoding.
        "medusa" means the medusa-style speculative decoding.

    spec_draft_length : int
        The number of tokens to generate in speculative proposal (draft).

    prefix_cache_mode : Literal["disable", "radix"]
        The prefix cache mode.
        "disable" means no prefix cache is disabled.
        "radix" means the paged radix tree based prefix cache mode.

    prefix_cache_max_num_recycling_seqs: Optional[int]
        The maximum number of recycling sequences in prefix cache, default as max_num_sequence.
        And set 0 to disable prefix cache, set -1 to have infinite capacity prefix cache.

    verbose : bool
        A boolean indicating whether to print logging info in engine.
    """

    model: Optional[str] = None
    model_lib: Optional[str] = None
    additional_models: List[Union[str, Tuple[str, str]]] = field(default_factory=list)
    mode: Optional[Literal["local", "interactive", "server"]] = None
    gpu_memory_utilization: Optional[float] = None
    kv_cache_page_size: int = 16
    max_num_sequence: Optional[int] = None
    max_total_sequence_length: Optional[int] = None
    max_single_sequence_length: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    max_history_size: Optional[int] = None
    kv_state_kind: Optional[Literal["kv_cache", "rnn_state"]] = None
    speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"] = "disable"
    spec_draft_length: int = 4
    prefix_cache_mode: Literal["disable", "radix"] = "radix"
    prefix_cache_max_num_recycling_seqs: Optional[int] = None
    verbose: bool = True

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "EngineConfig":
        """Construct a config from JSON string."""
        return EngineConfig(**json.loads(json_str))
