# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches
import json
import os
from typing import Dict, List, Optional, Tuple

import pytest
import regex
import requests

OPENAI_V1_CHAT_COMPLETION_URL = "http://127.0.0.1:8001/v1/chat/completions"

JSON_TOKEN_PATTERN = (
    r"((-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?)|null|true|false|"
    r'("((\\["\\\/bfnrt])|(\\u[0-9a-fA-F]{4})|[^"\\\x00-\x1f])*")'
)
JSON_TOKEN_RE = regex.compile(JSON_TOKEN_PATTERN)


def is_json_or_json_prefix(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError as e:
        # If the JSON decoder reaches the end of s, it is a prefix of a JSON string.
        if e.pos == len(s):
            return True
        # Since json.loads is token-based instead of char-based, there may remain half a token after
        # the matching position.
        # If the left part is a prefix of a valid JSON token, the output is also valid
        regex_match = JSON_TOKEN_RE.fullmatch(s[e.pos :], partial=True)
        return regex_match is not None


def check_openai_nonstream_response(
    response: Dict,
    *,
    is_chat_completion: bool,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reasons: List[str],
    completion_tokens: Optional[int] = None,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    stop: Optional[List[str]] = None,
    require_substr: Optional[List[str]] = None,
    json_mode: bool = False,
):
    assert response["model"] == model
    assert response["object"] == object_str

    choices = response["choices"]
    assert isinstance(choices, list)
    assert len(choices) <= num_choices
    texts: List[str] = ["" for _ in range(num_choices)]
    for choice in choices:
        idx = choice["index"]
        assert choice["finish_reason"] in finish_reasons

        if not is_chat_completion:
            assert isinstance(choice["text"], str)
            texts[idx] = choice["text"]
            if echo_prompt is not None:
                assert texts[idx]
            if suffix is not None:
                assert texts[idx]
        else:
            message = choice["message"]
            assert message["role"] == "assistant"
            assert isinstance(message["content"], str)
            texts[idx] = message["content"]

        if stop is not None:
            for stop_str in stop:
                assert stop_str not in texts[idx]
        if require_substr is not None:
            for substr in require_substr:
                assert substr in texts[idx]
        if json_mode:
            assert is_json_or_json_prefix(texts[idx])

    usage = response["usage"]
    assert isinstance(usage, dict)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0
    if completion_tokens is not None:
        assert usage["completion_tokens"] == completion_tokens


def check_openai_stream_response(
    responses: List[Dict],
    *,
    is_chat_completion: bool,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reasons: List[str],
    completion_tokens: Optional[int] = None,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    stop: Optional[List[str]] = None,
    require_substr: Optional[List[str]] = None,
    json_mode: bool = False,
):
    assert len(responses) > 0

    finished = [False for _ in range(num_choices)]
    outputs = ["" for _ in range(num_choices)]
    for response in responses:
        assert response["model"] == model
        assert response["object"] == object_str

        choices = response["choices"]
        assert isinstance(choices, list)
        assert len(choices) <= num_choices
        for choice in choices:
            idx = choice["index"]

            if not is_chat_completion:
                assert isinstance(choice["text"], str)
                outputs[idx] += choice["text"]
            else:
                delta = choice["delta"]
                assert delta["role"] == "assistant"
                assert isinstance(delta["content"], str)
                outputs[idx] += delta["content"]

            if finished[idx]:
                assert choice["finish_reason"] in finish_reasons
            elif choice["finish_reason"] is not None:
                assert choice["finish_reason"] in finish_reasons
                finished[idx] = True

        if not is_chat_completion:
            usage = response["usage"]
            assert isinstance(usage, dict)
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert usage["prompt_tokens"] > 0
            if completion_tokens is not None:
                assert usage["completion_tokens"] <= completion_tokens

    if not is_chat_completion:
        if completion_tokens is not None:
            assert responses[-1]["usage"]["completion_tokens"] == completion_tokens

    for i, output in enumerate(outputs):
        if echo_prompt is not None:
            assert output.startswith(echo_prompt)
        if suffix is not None:
            assert output.endswith(suffix)
        if stop is not None:
            for stop_str in stop:
                assert stop_str not in output
        if require_substr is not None:
            for substr in require_substr:
                assert substr in output
        if json_mode:
            assert is_json_or_json_prefix(output)


CHAT_COMPLETION_MESSAGES = [
    # messages #0
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://llava-vl.github.io/static/images/view.jpg",
                },
                {"type": "text", "text": "What does this image represent?"},
            ],
        },
    ],
    # messages #1
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://llava-vl.github.io/static/images/view.jpg",
                },
                {"type": "text", "text": "What does this image represent?"},
            ],
        },
        {
            "role": "assistant",
            "content": "The image represents a serene and peaceful scene of a pier extending over a body of water, such as a lake or a river.er. The pier is made of wood and has a bench on it, providing a place for people to sit and enjoy the view. The pier is situated in a natural environment, surrounded by trees and mountains in the background. This setting creates a tranquil atmosphere, inviting visitors to relax and appreciate the beauty of the landscape.",
        },
        {
            "role": "user",
            "content": "What country is the image set in? Give me 10 ranked guesses and reasons why.",
        },
    ],
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completions(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
    messages: List[Dict[str, str]],
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
    }
    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["stop"],
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reasons=["stop"],
        )


if __name__ == "__main__":
    model_lib = os.environ.get("MLC_SERVE_MODEL_LIB")
    if model_lib is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL_LIB" not found. '
            "Please set it to model lib compiled by MLC LLM "
            "(e.g., `dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so`)."
        )

    model = os.environ.get("MLC_SERVE_MODEL")
    if model is None:
        MODEL = (os.path.dirname(model_lib), model_lib)
    else:
        MODEL = (model, model_lib)

    for msg in CHAT_COMPLETION_MESSAGES:
        test_openai_v1_chat_completions(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions(MODEL, None, stream=True, messages=msg)
