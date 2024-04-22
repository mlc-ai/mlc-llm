"""Utility functions for MLC Serve engine"""

import uuid
from typing import Callable, List, Union

from mlc_llm.serve import data

from ..protocol import RequestProtocol, error_protocol, protocol_utils


def random_uuid() -> str:
    """Generate a random id in hexadecimal string."""
    return uuid.uuid4().hex


def check_unsupported_fields(request: RequestProtocol) -> None:
    """Check if the request has unsupported fields. Raise BadRequestError if so."""
    unsupported_fields = protocol_utils.get_unsupported_fields(request)
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
