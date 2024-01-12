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
# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches
import json
import os
from typing import Dict, List, Optional

import pytest
import requests
from openai import OpenAI

OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
OPENAI_V1_MODELS_URL = "http://127.0.0.1:8000/v1/models"
OPENAI_V1_COMPLETION_URL = "http://127.0.0.1:8000/v1/completions"
OPENAI_V1_CHAT_COMPLETION_URL = "http://127.0.0.1:8000/v1/chat/completions"


def check_openai_nonstream_response(
    response: Dict,
    *,
    is_chat_completion: bool,
    model: str,
    object_str: str,
    num_choices: int,
    finish_reason: str,
    completion_tokens: Optional[int] = None,
    echo_prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    stop: Optional[List[str]] = None,
    require_substr: Optional[List[str]] = None,
):
    assert response["model"] == model
    assert response["object"] == object_str

    choices = response["choices"]
    assert isinstance(choices, list)
    assert len(choices) == num_choices
    for idx, choice in enumerate(choices):
        assert choice["index"] == idx
        assert choice["finish_reason"] == finish_reason

        text: str
        if not is_chat_completion:
            assert isinstance(choice["text"], str)
            text = choice["text"]
            if echo_prompt is not None:
                assert text
            if suffix is not None:
                assert text
        else:
            message = choice["message"]
            assert message["role"] == "assistant"
            assert isinstance(message["content"], str)
            text = message["content"]

        if stop is not None:
            for stop_str in stop:
                assert stop_str not in text
        if require_substr is not None:
            for substr in require_substr:
                assert substr in text

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
    finish_reason: str,
    completion_tokens: Optional[int] = None,
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

            if not is_chat_completion:
                assert isinstance(choice["text"], str)
                outputs[idx] += choice["text"]
            else:
                delta = choice["delta"]
                assert delta["role"] == "assistant"
                assert isinstance(delta["content"], str)
                outputs[idx] += delta["content"]

            if finished[idx]:
                assert choice["finish_reason"] == finish_reason
            elif choice["finish_reason"] is not None:
                assert choice["finish_reason"] == finish_reason
                finished[idx] = True

        if not is_chat_completion:
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
        if stop is not None:
            for stop_str in stop:
                assert stop_str not in output
        if require_substr is not None:
            for substr in require_substr:
                assert substr in output


def expect_error(response_str: str, msg_prefix: Optional[str] = None):
    response = json.loads(response_str)
    assert response["object"] == "error"
    assert isinstance(response["message"], str)
    if msg_prefix is not None:
        assert response["message"].startswith(msg_prefix)


def test_openai_v1_models(
    served_model,
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    response = requests.get(OPENAI_V1_MODELS_URL, timeout=60).json()
    assert response["object"] == "list"
    models = response["data"]
    assert isinstance(models, list)
    assert len(models) == 1

    model_card = models[0]
    assert isinstance(model_card, dict)
    assert model_card["id"] == served_model
    assert model_card["object"] == "model"
    assert model_card["owned_by"] == "MLC-LLM"


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
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
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_openai_package(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key="None")
    prompt = "What is the meaning of life?"
    max_tokens = 256
    response = client.completions.create(
        model=served_model,
        prompt=prompt,
        max_tokens=max_tokens,
        stream=stream,
    )
    if not stream:
        check_openai_nonstream_response(
            response.model_dump(),
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )
    else:
        responses = []
        for chunk in response:  # pylint: disable=not-an-iterable
            responses.append(chunk.model_dump())
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
        )


def test_openai_v1_completions_invalid_requested_model(
    launch_server,  # pylint: disable=unused-argument
):
    # `launch_server` is a pytest fixture defined in conftest.py.

    model = "unserved_model"
    payload = {
        "model": model,
        "prompt": "What is the meaning of life?",
        "max_tokens": 10,
    }
    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    expect_error(
        response_str=response.json(), msg_prefix=f'The requested model "{model}" is not served.'
    )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_echo(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
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
            is_chat_completion=False,
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
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
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
    # `served_model` and `launch_server` are pytest fixtures
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
            is_chat_completion=False,
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
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
            suffix=suffix,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_stop_str(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Choose "in" as the stop string since it is very unlikely that
    # "in" does not appear in the generated output.
    prompt = "What is the meaning of life?"
    stop = ["in"]
    max_tokens = 256
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="stop",
            stop=stop,
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="stop",
            stop=stop,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_temperature(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What's the meaning of life?"
    max_tokens = 128
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model,
            object_str="text_completion",
            num_choices=1,
            finish_reason="length",
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_prompt_overlong(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
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


def test_openai_v1_completions_unsupported_args(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Right now "best_of" is unsupported.
    best_of = 2
    payload = {
        "model": served_model,
        "prompt": "What is the meaning of life?",
        "max_tokens": 256,
        "best_of": best_of,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    error_msg_prefix = 'Request fields "best_of" are not supported right now.'
    expect_error(response.json(), msg_prefix=error_msg_prefix)


def test_openai_v1_completions_request_cancellation(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Use a large max_tokens and small timeout to force timeouts.
    payload = {
        "model": served_model,
        "prompt": "What is the meaning of life?",
        "max_tokens": 2048,
        "stream": False,
    }
    with pytest.raises(requests.exceptions.Timeout):
        requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=1)

    # The server should still be alive after a request cancelled.
    # We query `v1/models` to validate the server liveness.
    response = requests.get(OPENAI_V1_MODELS_URL, timeout=60).json()

    assert response["object"] == "list"
    models = response["data"]
    assert isinstance(models, list)
    assert len(models) == 1

    model_card = models[0]
    assert isinstance(model_card, dict)
    assert model_card["id"] == served_model
    assert model_card["object"] == "model"
    assert model_card["owned_by"] == "MLC-LLM"


CHAT_COMPLETION_MESSAGES = [
    # messages #0
    [{"role": "user", "content": "Hello! Our project is MLC LLM."}],
    # messages #1
    [
        {"role": "user", "content": "Hello! Our project is MLC LLM."},
        {
            "role": "assistant",
            "content": "Hello! It's great to hear about your project, MLC LLM.",
        },
        {"role": "user", "content": "What is the name of our project?"},
    ],
    # messages #2
    [
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. "
            "You always ends your response with an emoji.",
        },
        {"role": "user", "content": "Hello! Our project is MLC LLM."},
    ],
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completions(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
    messages: List[Dict[str, str]],
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    payload = {
        "model": served_model,
        "messages": messages,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model,
            object_str="chat.completion",
            num_choices=1,
            finish_reason="stop",
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
            model=served_model,
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reason="stop",
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completions_openai_package(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
    messages: List[Dict[str, str]],
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key="None")
    response = client.chat.completions.create(
        model=served_model,
        messages=messages,
        stream=stream,
    )
    if not stream:
        check_openai_nonstream_response(
            response.model_dump(),
            is_chat_completion=True,
            model=served_model,
            object_str="chat.completion",
            num_choices=1,
            finish_reason="stop",
        )
    else:
        responses = []
        for chunk in response:  # pylint: disable=not-an-iterable
            responses.append(chunk.model_dump())
        check_openai_stream_response(
            responses,
            is_chat_completion=True,
            model=served_model,
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reason="stop",
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_max_tokens(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    messages = [{"role": "user", "content": "Write a novel with at least 500 words."}]
    max_tokens = 16
    payload = {
        "model": served_model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model,
            object_str="chat.completion",
            num_choices=1,
            finish_reason="length",
            completion_tokens=max_tokens,
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
            model=served_model,
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reason="length",
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_system_prompt_wrong_pos(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    messages = [
        {"role": "user", "content": "Hello! Our project is MLC LLM."},
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. "
            "You always ends your response with an emoji.",
        },
    ]
    payload = {
        "model": served_model,
        "messages": messages,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    error_msg = "System prompt at position 1 in the message list is invalid."
    if not stream:
        expect_error(response.json(), msg_prefix=error_msg)
    else:
        num_chunks = 0
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk:
                continue
            num_chunks += 1
            expect_error(json.loads(chunk.decode("utf-8")), msg_prefix=error_msg)
        assert num_chunks == 1


def test_openai_v1_chat_completions_unsupported_args(
    served_model: str,
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Right now "tool_choice" is unsupported.
    tool_choice = "auto"
    payload = {
        "model": served_model,
        "messages": CHAT_COMPLETION_MESSAGES[0],
        "tool_choice": tool_choice,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    error_msg_prefix = 'Request fields "tool_choice" are not supported right now.'
    expect_error(response.json(), msg_prefix=error_msg_prefix)


if __name__ == "__main__":
    MODEL = os.environ.get("MLC_SERVE_MODEL")
    if MODEL is None:
        MODEL = "Llama-2-7b-chat-hf-q0f16"
        print(
            'WARNING: Variable "MLC_SERVE_MODEL" not found in environment, '
            f"fallback to use model {MODEL}, which requires 16GB of VRAM. "
            "Changing to use a quantized model can reduce the VRAM requirements."
        )

    test_openai_v1_models(MODEL, None)

    test_openai_v1_completions(MODEL, None, stream=False)
    test_openai_v1_completions(MODEL, None, stream=True)
    test_openai_v1_completions_openai_package(MODEL, None, stream=False)
    test_openai_v1_completions_openai_package(MODEL, None, stream=True)
    test_openai_v1_completions_invalid_requested_model(None)
    test_openai_v1_completions_echo(MODEL, None, stream=False)
    test_openai_v1_completions_echo(MODEL, None, stream=True)
    test_openai_v1_completions_suffix(MODEL, None, stream=False)
    test_openai_v1_completions_suffix(MODEL, None, stream=True)
    test_openai_v1_completions_stop_str(MODEL, None, stream=False)
    test_openai_v1_completions_stop_str(MODEL, None, stream=True)
    test_openai_v1_completions_temperature(MODEL, None, stream=False)
    test_openai_v1_completions_temperature(MODEL, None, stream=True)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=False)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=True)
    test_openai_v1_completions_unsupported_args(MODEL, None)
    test_openai_v1_completions_request_cancellation(MODEL, None)

    for msg in CHAT_COMPLETION_MESSAGES:
        test_openai_v1_chat_completions(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions(MODEL, None, stream=True, messages=msg)
        test_openai_v1_chat_completions_openai_package(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions_openai_package(MODEL, None, stream=True, messages=msg)
    test_openai_v1_chat_completions_max_tokens(MODEL, None, stream=False)
    test_openai_v1_chat_completions_max_tokens(MODEL, None, stream=True)
    test_openai_v1_chat_completions_system_prompt_wrong_pos(MODEL, None, stream=False)
    test_openai_v1_chat_completions_system_prompt_wrong_pos(MODEL, None, stream=True)
    test_openai_v1_chat_completions_unsupported_args(MODEL, None)
