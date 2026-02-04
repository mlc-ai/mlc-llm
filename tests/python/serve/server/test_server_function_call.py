# pylint: disable=line-too-long
"""
Test script for function call in chat completion. To run this script, use the following command:
MLC_SERVE_MODEL_LIB=dist/gorilla-openfunctions-v1-q4f16_1_MLC/gorilla-openfunctions-v1-q4f16_1-cuda.so
MLC_SERVE_MODEL_LIB=${MLC_SERVE_MODEL_LIB} python -m pytest -x tests/python/serve/server/test_server_function_call.py
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches
import json
import os
from typing import Dict, List, Optional, Tuple

import pytest
import requests

OPENAI_V1_CHAT_COMPLETION_URL = "http://127.0.0.1:8000/v1/chat/completions"


def check_openai_nonstream_response(
    response: Dict,
    *,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reason: List[str],
    completion_tokens: Optional[int] = None,
):
    print(response)
    assert response["model"] == model
    assert response["object"] == object_str

    choices = response["choices"]
    assert isinstance(choices, list)
    assert len(choices) == num_choices
    for idx, choice in enumerate(choices):
        assert choice["index"] == idx
        assert choice["finish_reason"] in finish_reason

        # text: str
        message = choice["message"]
        assert message["role"] == "assistant"
        if choice["finish_reason"] == "tool_calls":
            assert message["content"] is None
            assert isinstance(message["tool_calls"], list)
        else:
            assert message["tool_calls"] is None
            assert message["content"] is not None

    usage = response["usage"]
    assert isinstance(usage, dict)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0

    if completion_tokens is not None:
        assert usage["completion_tokens"] == completion_tokens


def check_openai_stream_response(
    responses: List[Dict],
    *,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reason: str,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    stop: Optional[List[str]] = None,
    require_substr: Optional[List[str]] = None,
):
    assert len(responses) > 0

    finished = [False for _ in range(num_choices)]
    outputs = ["" for _ in range(num_choices)]
    for response in responses:
        assert response["model"] == model
        assert response["object"] == object_str

        choices = response["choices"]
        assert isinstance(choices, list)
        assert len(choices) == num_choices
        for idx, choice in enumerate(choices):
            assert choice["index"] == idx

            delta = choice["delta"]
            assert delta["role"] == "assistant"
            assert isinstance(delta["content"], str)
            outputs[idx] += delta["content"]

            if finished[idx]:
                assert choice["finish_reason"] == finish_reason
            elif choice["finish_reason"] is not None:
                assert choice["finish_reason"] == finish_reason
                finished[idx] = True

    for output in outputs:
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


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


CHAT_COMPLETION_MESSAGES = [
    # messages #0
    [
        {
            "role": "user",
            "content": "What is the current weather in Pittsburgh, PA?",
        }
    ],
    # messages #1
    [
        {
            "role": "user",
            "content": "What is the current weather in Pittsburgh, PA and Tokyo, JP?",
        }
    ],
    # messages #2
    [
        {
            "role": "user",
            "content": "What is the current weather in Pittsburgh, PA in fahrenheit?",
        }
    ],
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completion_function_call(
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
        "tools": tools,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reason=["tool_calls", "error"],
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            model=served_model[0],
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reason="tool_calls",
        )


if __name__ == "__main__":
    model_lib = os.environ.get("MLC_SERVE_MODEL_LIB")
    if model_lib is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL_LIB" not found. '
            "Please set it to model lib compiled by MLC LLM "
            "(e.g., `./dist/gorilla-openfunctions-v1-q4f16_1_MLC/gorilla-openfunctions-v1-q4f16_1-cuda.so`) "
            "which supports function calls."
        )
    MODEL = (os.path.dirname(model_lib), model_lib)

    for msg in CHAT_COMPLETION_MESSAGES:
        test_openai_v1_chat_completion_function_call(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completion_function_call(MODEL, None, stream=True, messages=msg)
