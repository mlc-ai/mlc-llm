"""Server tests in MLC LLM.
Before running any test, we use pytest fixtures to launch a
test-session-wide server in a subprocess, and then execute the tests.

The recommended way to run the tests is to use the following command:
  MLC_SERVE_MODEL="YOUR_MODEL_ID" pytest -vv tests/python/serve/server/test_server.py

Here "YOUR_MODEL_ID" can be a small model like `Llama-2-7b-chat-hf-q4f16_1`,
as long as the model is built with batching and embedding separation enabled.

To directly run the Python file (a.k.a., not using pytest), you need to
launch the server in ahead before running this file. This can be done in
two steps:
- start a new shell session, run
  python -m mlc_chat.serve.server --model "YOUR_MODEL_ID"
- start another shell session, run this file
  MLC_SERVE_MODEL="YOUR_MODEL_ID" python tests/python/serve/server/test_server.py
"""
# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals
import json
import os
from typing import Dict, List, Optional

import pytest
import requests

OPENAI_V1_COMPLETION_URL = "http://127.0.0.1:8000/v1/completions"


def check_openai_nonstream_response(
    response: Dict,
    *,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reason: str,
    completion_tokens: Optional[int] = None,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
):
    assert response["model"] == model
    assert response["object"] == object_str

    choices = response["choices"]
    assert isinstance(choices, list)
    assert len(choices) == num_choices
    for idx, choice in enumerate(choices):
        assert choice["index"] == idx
        assert choice["finish_reason"] == finish_reason
        assert isinstance(choice["text"], str)
        if echo_prompt is not None:
            assert choice["text"].startswith(echo_prompt)
        if suffix is not None:
            assert choice["text"].endswith(suffix)

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
    completion_tokens: Optional[int] = None,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
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
            assert isinstance(choice["text"], str)
            outputs[idx] += choice["text"]
            if finished[idx]:
                assert choice["finish_reason"] == finish_reason
            elif choice["finish_reason"] is not None:
                assert choice["finish_reason"] == finish_reason
                finished[idx] = True

        usage = response["usage"]
        assert isinstance(usage, dict)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["prompt_tokens"] > 0
        if completion_tokens is not None:
            assert usage["completion_tokens"] <= completion_tokens

    if completion_tokens is not None:
        assert responses[-1]["usage"]["completion_tokens"] == completion_tokens

    for output in outputs:
        if echo_prompt is not None:
            assert output.startswith(echo_prompt)
        if suffix is not None:
            assert output.endswith(suffix)


def expect_error(response_str: str, msg_prefix: Optional[str] = None):
    response = json.loads(response_str)
    assert response["object"] == "error"
    assert isinstance(response["message"], str)
    if msg_prefix is not None:
        assert response["message"].startswith(msg_prefix)


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixture
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    max_tokens = 256
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [Done]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_echo(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixture
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    max_tokens = 256
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "echo": True,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
            echo_prompt=prompt,
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [Done]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
            echo_prompt=prompt,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_suffix(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixture
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    suffix = "Hello, world!"
    max_tokens = 256
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "suffix": suffix,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
            suffix=suffix,
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [Done]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
            suffix=suffix,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_prompt_overlong(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixture
    # defined in conftest.py.

    num_tokens = 17000
    prompt = [128] * num_tokens
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": 256,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    error_msg_prefix = (
        f"Request prompt has {num_tokens} tokens in total, larger than the model capacity"
    )
    if not stream:
        expect_error(response.json(), msg_prefix=error_msg_prefix)
    else:
        num_chunks = 0
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk:
                continue
            num_chunks += 1
            expect_error(json.loads(chunk.decode("utf-8")), msg_prefix=error_msg_prefix)
        assert num_chunks == 1


if __name__ == "__main__":
    MODEL = os.environ.get("MLC_SERVE_MODEL")
    if MODEL is None:
        MODEL = "Llama-2-7b-chat-hf-q0f16"
        print(
            'WARNING: Variable "MLC_SERVE_MODEL" not found in environment, '
            f"fallback to use model {MODEL}, which requires 16GB of VRAM. "
            "Changing to use a quantized model can reduce the VRAM requirements."
        )

    test_openai_v1_completions(MODEL, None, stream=False)
    test_openai_v1_completions(MODEL, None, stream=True)
    test_openai_v1_completions_echo(MODEL, None, stream=False)
    test_openai_v1_completions_echo(MODEL, None, stream=True)
    test_openai_v1_completions_suffix(MODEL, None, stream=False)
    test_openai_v1_completions_suffix(MODEL, None, stream=True)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=False)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=True)
