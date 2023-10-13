from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Protocol

from .sampling_params import SamplingParams


@dataclass
class StoppingCriteria:
    max_tokens: Optional[int]


RequestId = str


@dataclass
class Request:
    request_id: RequestId
    prompt: str
    sampling_params: SamplingParams
    stopping_criteria: StoppingCriteria


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
    def add(self, requests: list[Request]):
        """
        Add requests to the queue of InferenceEngine.

        Returns a list of request id, that should be used to relate the output
        of `step` back to each request
        """

    def cancel(self, request_id: RequestId):
        """
        Cancel the generation of a request.
        """

    def wait_for_request(self, timeout_seconds=None):
        """
        Block until there is request to process
        """

    def step(self) -> InferenceStepResult:
        """
        InferenceResult contains the next token for processed results,
        and indicates whether the generation for a request is finished.

        It's up to the InferenceEngine to choose which requests
        to work on, while it should be guaranteed that all requests will be
        processed eventually.

        If the engine has no requests in the queue, `step` will return immediately.
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


_GB = 1 << 30


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
    """

    def __init__(
        self,
        model: str,
        lib_path: str,
        device: str,
        # tokenizer: str,
        # tokenizer_mode: str,
        # trust_remote_code: bool,
        # download_dir: Optional[str],
        # load_format: str,
        # dtype: str,
        random_seed: int,
        # revision: Optional[str] = None,
        # tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
    ) -> None:
        self.model = model
        self.lib_path = lib_path
        self.device = device
        # self.tokenizer = tokenizer
        # self.tokenizer_mode = tokenizer_mode
        # self.trust_remote_code = trust_remote_code
        # self.download_dir = download_dir
        # self.load_format = load_format
        self.random_seed = random_seed
        # self.revision = revision
        # self.tokenizer_revision = tokenizer_revision
        self.max_model_len = max_model_len


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.sliding_window = sliding_window


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[List[Dict[int, float]]],
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason})"
        )


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished})"
        )
