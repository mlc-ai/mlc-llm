"""Utility functions for server entrypoints"""

import uuid
from http import HTTPStatus
from typing import Callable, List, Optional, Union
import sys
import subprocess
import json

import fastapi

from ...protocol import RequestProtocol
from ...protocol.protocol_utils import ErrorResponse, get_unsupported_fields


def random_uuid() -> str:
    """Generate a random id in hexadecimal string."""
    return uuid.uuid4().hex


def create_error_response(status_code: HTTPStatus, message: str) -> fastapi.responses.JSONResponse:
    """Create a JSON response that reports error with regarding the input message."""
    return fastapi.responses.JSONResponse(
        ErrorResponse(message=message, code=status_code.value).model_dump_json(),
        status_code=status_code.value,
    )


def check_unsupported_fields(
    request: RequestProtocol,
) -> Optional[fastapi.responses.JSONResponse]:
    """Check if the request has unsupported fields. Return an error if so."""
    unsupported_fields = get_unsupported_fields(request)
    if len(unsupported_fields) != 0:
        unsupported_fields = [f'"{field}"' for field in unsupported_fields]
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            message=f'Request fields {", ".join(unsupported_fields)} are not supported right now.',
        )
    return None


def check_prompts_length(
    prompts: List[List[int]], max_single_sequence_length: int
) -> Optional[fastapi.responses.JSONResponse]:
    """Check if the total prompt length exceeds the max single sequence
    sequence length allowed by the served model. Return an error if so.
    """
    total_length = 0
    for prompt in prompts:
        total_length += len(prompt)
    if total_length > max_single_sequence_length:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            message=f"Request prompt has {total_length} tokens in total,"
            f" larger than the model capacity {max_single_sequence_length}.",
        )
    return None


def process_prompts(
    input_prompts: Union[str, List[int], List[Union[str, List[int]]]],
    ftokenize: Callable[[str], List[int]],
) -> Union[List[List[int]], fastapi.responses.JSONResponse]:
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
        return create_error_response(HTTPStatus.BAD_REQUEST, message=error_msg)

    # Case 2. The prompt is a list of token ids.
    if isinstance(input_prompts[0], int):
        if not all(isinstance(token_id, int) for token_id in input_prompts):
            return create_error_response(HTTPStatus.BAD_REQUEST, message=error_msg)
        return [input_prompts]

    # Case 3. A list of prompts.
    output_prompts: List[List[int]] = []
    for input_prompt in input_prompts:
        is_str = isinstance(input_prompt, str)
        is_token_ids = isinstance(input_prompt, list) and all(
            isinstance(token_id, int) for token_id in input_prompt
        )
        if not (is_str or is_token_ids):
            return create_error_response(HTTPStatus.BAD_REQUEST, message=error_msg)
        output_prompts.append(ftokenize(input_prompt) if is_str else input_prompt)  # type: ignore
    return output_prompts


def get_image_from_url(url: str):
    """Get the image from the given URL, process and return the image tensor as TVM NDArray."""
    import tvm
    from transformers import CLIPImageProcessor

    from io import BytesIO

    import requests
    from PIL import Image

    response = requests.get(url)
    image_tensor = Image.open(BytesIO(response.content)).convert("RGB")

    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}
    )
    image_features = tvm.nd.array(
        image_processor.preprocess(image_tensor, return_tensors="np")["pixel_values"].astype(
            "float16"
        ),
        device=tvm.runtime.ndarray.cuda(),
    )
    return image_features


def get_image_embed_size(config_file_path: str) -> int:
    """Get the image embedding size from the model config file."""
    with open(config_file_path, "r") as file:
        config = json.load(file)
    image_size = config["model_config"]["vision_config"]["image_size"]
    patch_size = config["model_config"]["vision_config"]["patch_size"]
    embed_size = (image_size // patch_size) ** 2
    return embed_size
