"""Utility functions for MLC Serve engine"""

import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from mlc_llm.protocol import RequestProtocol, error_protocol, openai_api_protocol
from mlc_llm.serve import data

from .config import DebugConfig, GenerationConfig, ResponseFormat


def get_unsupported_fields(request: RequestProtocol) -> List[str]:
    """Get the unsupported fields of the request.
    Return the list of unsupported field names.
    """
    if isinstance(
        request, (openai_api_protocol.CompletionRequest, openai_api_protocol.ChatCompletionRequest)
    ):
        return openai_api_protocol.openai_api_get_unsupported_fields(request)
    raise RuntimeError("Cannot reach here")


def openai_api_get_generation_config(
    request: Union[openai_api_protocol.CompletionRequest, openai_api_protocol.ChatCompletionRequest]
) -> Dict[str, Any]:
    """Create the generation config from the given request."""
    kwargs: Dict[str, Any] = {}
    arg_names = [
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "seed",
    ]
    for arg_name in arg_names:
        kwargs[arg_name] = getattr(request, arg_name)
    if kwargs["max_tokens"] is None:
        # Setting to -1 means the generation will not stop until
        # exceeding model capability or hit any stop criteria.
        kwargs["max_tokens"] = -1
    if request.stop is not None:
        kwargs["stop_strs"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.response_format is not None:
        kwargs["response_format"] = ResponseFormat(
            **request.response_format.model_dump(by_alias=True)
        )
    if request.debug_config is not None:
        kwargs["debug_config"] = DebugConfig(**request.debug_config.model_dump())
    return kwargs


def get_generation_config(
    request: RequestProtocol,
    extra_stop_token_ids: Optional[List[int]] = None,
    extra_stop_str: Optional[List[str]] = None,
) -> GenerationConfig:
    """Create the generation config in MLC LLM out from the input request protocol."""
    kwargs: Dict[str, Any]
    if isinstance(
        request, (openai_api_protocol.CompletionRequest, openai_api_protocol.ChatCompletionRequest)
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
            f'Request fields {", ".join(unsupported_fields)} are not supported right now.',
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
    prompts: Union[str, List[int], List[Union[str, List[int], data.Data]]]
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
