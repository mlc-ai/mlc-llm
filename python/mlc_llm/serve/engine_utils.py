"""Utility functions for MLC Serve engine"""

import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from mlc_llm.protocol import error_protocol, openai_api_protocol
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import data

RequestProtocol = Union[
    openai_api_protocol.CompletionRequest, openai_api_protocol.ChatCompletionRequest
]


def get_unsupported_fields(request: RequestProtocol) -> List[str]:
    """Get the unsupported fields of the request.
    Return the list of unsupported field names.
    """
    if isinstance(
        request,
        (
            openai_api_protocol.CompletionRequest,
            openai_api_protocol.ChatCompletionRequest,
        ),
    ):
        return openai_api_protocol.openai_api_get_unsupported_fields(request)
    raise RuntimeError("Cannot reach here")


def openai_api_get_generation_config(request: RequestProtocol) -> Dict[str, Any]:
    """Create the generation config from the given request."""
    kwargs: Dict[str, Any] = {}
    arg_names = [
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "seed",
        "response_format",
        "debug_config",
    ]
    for arg_name in arg_names:
        kwargs[arg_name] = getattr(request, arg_name)
    if kwargs["max_tokens"] is None:
        # Setting to -1 means the generation will not stop until
        # exceeding model capability or hit any stop criteria.
        kwargs["max_tokens"] = -1
    if request.stop is not None:
        kwargs["stop_strs"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if isinstance(request, openai_api_protocol.ChatCompletionRequest):
        kwargs["logprobs"] = request.logprobs
        kwargs["top_logprobs"] = request.top_logprobs
    else:
        logprobs = request.logprobs is not None
        kwargs["logprobs"] = logprobs
        kwargs["top_logprobs"] = request.logprobs if logprobs else 0
    return kwargs


def get_generation_config(
    request: RequestProtocol,
    extra_stop_token_ids: Optional[List[int]] = None,
    extra_stop_str: Optional[List[str]] = None,
) -> GenerationConfig:
    """Create the generation config in MLC LLM out from the input request protocol."""
    kwargs: Dict[str, Any]
    if isinstance(
        request,
        (
            openai_api_protocol.CompletionRequest,
            openai_api_protocol.ChatCompletionRequest,
        ),
    ):
        kwargs = openai_api_get_generation_config(request)
    else:
        raise RuntimeError("Cannot reach here")

    if extra_stop_token_ids is not None:
        stop_token_ids = kwargs.get("stop_token_ids", [])
        assert isinstance(stop_token_ids, list)
        stop_token_ids += extra_stop_token_ids
        kwargs["stop_token_ids"] = stop_token_ids

    if extra_stop_str is not None:
        stop_strs = kwargs.get("stop_strs", [])
        assert isinstance(stop_strs, list)
        stop_strs += extra_stop_str
        kwargs["stop_strs"] = stop_strs

    return GenerationConfig(**kwargs)


def random_uuid() -> str:
    """Generate a random id in hexadecimal string."""
    return uuid.uuid4().hex


def check_unsupported_fields(request: RequestProtocol) -> None:
    """Check if the request has unsupported fields. Raise BadRequestError if so."""
    unsupported_fields = get_unsupported_fields(request)
    if len(unsupported_fields) != 0:
        unsupported_fields = [f'"{field}"' for field in unsupported_fields]
        raise error_protocol.BadRequestError(
            f"Request fields {', '.join(unsupported_fields)} are not supported right now.",
        )


def check_and_get_prompts_length(
    prompts: List[Union[List[int], data.ImageData]], max_input_sequence_length: int
) -> int:
    """Check if the total prompt length exceeds the max single sequence
    sequence length allowed by the served model. Raise BadRequestError if so.
    Return the total prompt length.
    """
    total_length: int = 0
    for prompt in prompts:
        total_length += len(prompt)
    if total_length > max_input_sequence_length:
        raise error_protocol.BadRequestError(
            f"Request prompt has {total_length} tokens in total,"
            f" larger than the model input length limit {max_input_sequence_length}.",
        )
    return total_length


def process_prompts(
    input_prompts: Union[str, List[int], List[Union[str, List[int], data.ImageData]]],
    ftokenize: Callable[[str], List[int]],
) -> List[Union[List[int], data.ImageData]]:
    """Convert all input tokens to list of token ids with regard to the
    given tokenization function.
    For each input prompt, return the list of token ids after tokenization.
    """
    error_msg = f"Invalid request prompt {input_prompts}"

    # Case 1. The prompt is a single string.
    if isinstance(input_prompts, str):
        return [ftokenize(input_prompts)]

    assert isinstance(input_prompts, list)
    if len(input_prompts) == 0:
        raise error_protocol.BadRequestError(error_msg)

    # Case 2. The prompt is a list of token ids.
    if isinstance(input_prompts[0], int):
        assert isinstance(input_prompts, list)
        if not all(isinstance(token_id, int) for token_id in input_prompts):
            raise error_protocol.BadRequestError(error_msg)
        return [input_prompts]  # type: ignore

    # Case 3. A list of prompts.
    output_prompts: List[Union[List[int], data.ImageData]] = []
    for input_prompt in input_prompts:
        if isinstance(input_prompt, str):
            output_prompts.append(ftokenize(input_prompt))
        elif isinstance(input_prompt, list) and all(
            isinstance(token_id, int) for token_id in input_prompt
        ):
            output_prompts.append(input_prompt)
        elif isinstance(input_prompt, data.ImageData):
            output_prompts.append(input_prompt)
        else:
            raise error_protocol.BadRequestError(error_msg)
    return output_prompts


def convert_prompts_to_data(
    prompts: Union[str, List[int], List[Union[str, List[int], data.Data]]],
) -> List[data.Data]:
    """Convert the given prompts in the combination of token id lists
    and/or data to all data."""
    if isinstance(prompts, data.Data):
        return [prompts]
    if isinstance(prompts, str):
        return [data.TextData(prompts)]
    if isinstance(prompts[0], int):
        assert isinstance(prompts, list) and all(isinstance(token_id, int) for token_id in prompts)
        return [data.TokenData(prompts)]  # type: ignore
    return [convert_prompts_to_data(x)[0] for x in prompts]  # type: ignore


class ErrorCleanupScope:
    """Scope to call cleanup when an error is thrown.

    This class provides an important pattern properly cleanup
    when async scope CancelledError or other exception happens.

    Parameters
    ----------
    cleanup : Callable
        A callable function to trigger at scope exit during an exception.

    Note
    ----
    This helper is motivated by the need to properly
    abort an async generator and trigger corresponding
    cleanup functions. Naively use the try except
    pattern will results in bug when we chain up
    async generators.

    .. code:: python

        class EngineNotSafe:
            async def _inner_gen(self, request):
                request_id = self.get_request_id()
                self.add_request(request)
                try:
                    async for res in await producer_stream:
                        yield res
                except asyncio.CancelledError:
                    self.abort(request_id)

            async def generate(self, request):
                async for res in await self._inner_gen(request):
                    # async error can he raised in here
                    # this will cause
                    res = await process(res)
                    yield res

    The above except pattern is not safe.
    This is because CancelledError may also be raised
    outside _inner_gen during the process of generate
    function in between iterations.

    Instead, we use ErrorCleanupScope to safeguard the
    generation process. The scope will always properly
    cleanup in exit function when the exception is raised

     .. code:: python

        class EngineSafe:
            async def _inner_gen(self, request):
                request_id = self.get_request_id()
                self.add_request(request)
                with ErrorCleanupScope(lambda: self.abort(request_id))
                    async for res in await producer_stream:
                        yield res

            async def generate(self, request):
                async for res in await self._inner_gen(request):
                    # even if async error is raised here
                    # it will cleanup the ErrorCleanupScope
                    # properly during function exit
                    res = await process(res)
                    yield res
    """

    cleanup: Callable

    def __init__(self, cleanup: Callable):
        self.cleanup = cleanup

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # only cleanup when exc type is not none
        if exc_type is not None:
            self.cleanup()


# ====== Embedding Engine Utilities ======


def load_embedding_params(model_weight_path, device, model_metadata) -> list:
    """Load embedding model parameters from weight directory.

    Parameters
    ----------
    model_weight_path : str
        Path to the model weight directory.
    device : tvm.runtime.Device
        The target device.
    model_metadata : dict
        The model metadata dictionary containing param info.

    Returns
    -------
    params : list
        List of tvm.runtime.Tensor parameters in metadata order.
    """
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_tensor_cache(model_weight_path, device)
    param_names = [param["name"] for param in model_metadata["params"]]
    assert len(param_names) == meta["ParamSize"]
    return [params[name] for name in param_names]


def detect_embedding_model_type(mod) -> Literal["encoder", "decoder"]:
    """Detect embedding model type from compiled TVM module functions.

    Parameters
    ----------
    mod : tvm.runtime.Module
        The VM module with model functions.

    Returns
    -------
    model_type : str
        "encoder" for BERT-style models, "decoder" for Qwen3-Embeddings style.
    """
    has_embed = mod.implements_function("embed")
    has_prefill_to_hidden = mod.implements_function("prefill_to_last_hidden_states")
    has_prefill = mod.implements_function("prefill")

    if has_embed and has_prefill_to_hidden:
        return "decoder"
    if has_prefill:
        return "encoder"
    raise ValueError(
        "Model does not support embedding inference. "
        "Expected 'embed' + 'prefill_to_last_hidden_states' (decoder) "
        "or 'prefill' (encoder)."
    )
