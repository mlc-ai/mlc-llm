"""Server tests in MLC LLM.
Before running any test, we use pytest fixtures to launch a
test-session-wide server in a subprocess, and then execute the tests.

The recommended way to run the tests is to use the following command:
  MLC_SERVE_MODEL_LIB="YOUR_MODEL_LIB" pytest -vv tests/python/serve/server/test_server.py

Here "YOUR_MODEL_LIB" is a compiled model library like
`dist/Llama-2-7b-chat-hf-q4f16_1/Llama-2-7b-chat-hf-q4f16_1-cuda.so`,
as long as the model is built with batching and embedding separation enabled.

To directly run the Python file (a.k.a., not using pytest), you need to
launch the server in ahead before running this file. This can be done in
two steps:
- start a new shell session, run
  python -m mlc_llm.serve.server --model "YOUR_MODEL_LIB"
- start another shell session, run this file
  MLC_SERVE_MODEL_LIB="YOUR_MODEL_LIB" python tests/python/serve/server/test_server.py
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches
import json
import os
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

import pytest
import regex
import requests
from openai import OpenAI
from pydantic import BaseModel

from mlc_llm.protocol.openai_api_protocol import (
    CHAT_COMPLETION_MAX_TOP_LOGPROBS,
    COMPLETION_MAX_TOP_LOGPROBS,
)

OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
OPENAI_V1_MODELS_URL = "http://127.0.0.1:8000/v1/models"
OPENAI_V1_COMPLETION_URL = "http://127.0.0.1:8000/v1/completions"
OPENAI_V1_CHAT_COMPLETION_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEBUG_DUMP_EVENT_TRACE_URL = "http://127.0.0.1:8000/debug/dump_event_trace"
METRICS_URL = "http://127.0.0.1:8000/metrics"


JSON_TOKEN_PATTERN = (
    r"((-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?)|null|true|false|"
    r'("((\\["\\\/bfnrt])|(\\u[0-9a-fA-F]{4})|[^"\\\x00-\x1f])*")'
)
JSON_TOKEN_RE = regex.compile(JSON_TOKEN_PATTERN)


def is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def is_json_prefix(s: str) -> bool:
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
    check_json_output: bool = False,
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
        if check_json_output:
            # the output should be json or a prefix of a json string
            # if the output is a prefix of a json string, the output must exceed the max output
            # length
            output_is_json = is_json(texts[idx])
            output_is_json_prefix = is_json_prefix(texts[idx])
            assert output_is_json or output_is_json_prefix
            if not output_is_json and output_is_json_prefix:
                assert choice["finish_reason"] == "length"

    usage = response["usage"]
    if usage is not None:
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
    check_json_output: bool = False,
):
    assert len(responses) > 0

    finished = [False for _ in range(num_choices)]
    outputs = ["" for _ in range(num_choices)]
    finish_reason_list = ["" for _ in range(num_choices)]
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
                finish_reason_list[idx] = choice["finish_reason"]
            elif choice["finish_reason"] is not None:
                assert choice["finish_reason"] in finish_reasons
                finish_reason_list[idx] = choice["finish_reason"]
                finished[idx] = True

        if not is_chat_completion:
            usage = response["usage"]
            if usage is not None:
                assert isinstance(usage, dict)
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                assert usage["prompt_tokens"] >= 0
                if completion_tokens is not None:
                    assert usage["completion_tokens"] <= completion_tokens

    if not is_chat_completion:
        if completion_tokens is not None and responses[-1]["usage"] is not None:
            assert responses[-1]["usage"]["completion_tokens"] == completion_tokens

    for i, (output, finish_reason) in enumerate(zip(outputs, finish_reason_list)):
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
        if check_json_output:
            # the output should be json or a prefix of a json string
            # if the output is a prefix of a json string, the output must exceed the max output
            # length
            output_is_json = is_json(output)
            output_is_json_prefix = is_json_prefix(output)
            assert output_is_json or output_is_json_prefix
            if not output_is_json and output_is_json_prefix:
                assert finish_reason == "length"


def expect_error(response_str: str, msg_prefix: Optional[str] = None):
    response = json.loads(response_str)
    assert response["object"] == "error"
    assert isinstance(response["message"], str)
    if msg_prefix is not None:
        assert response["message"].startswith(msg_prefix)


def test_openai_v1_models(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    response = requests.get(OPENAI_V1_MODELS_URL, timeout=180).json()
    assert response["object"] == "list"
    models = response["data"]
    assert isinstance(models, list)
    assert len(models) == 1

    model_card = models[0]
    assert isinstance(model_card, dict)
    assert model_card["id"] == served_model[0], f"{model_card['id']} {served_model[0]}"
    assert model_card["object"] == "model"
    assert model_card["owned_by"] == "MLC-LLM"


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    max_tokens = 256
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_openai_package(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key="None")
    prompt = "What is the meaning of life?"
    max_tokens = 256
    response = client.completions.create(
        model=served_model[0],
        prompt=prompt,
        max_tokens=max_tokens,
        stream=stream,
    )
    if not stream:
        check_openai_nonstream_response(
            response.model_dump(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            completion_tokens=max_tokens,
        )
    else:
        responses = []
        for chunk in response:  # pylint: disable=not-an-iterable
            responses.append(chunk.model_dump())
        check_openai_stream_response(
            responses,
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_echo(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    max_tokens = 256
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "echo": True,
        "stream": stream,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
            completion_tokens=max_tokens,
            echo_prompt=prompt,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_suffix(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What is the meaning of life?"
    suffix = "Hello, world!"
    max_tokens = 256
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "suffix": suffix,
        "stream": stream,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
            completion_tokens=max_tokens,
            suffix=suffix,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_stop_str(
    served_model: Tuple[str, str],
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
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["stop", "length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["stop", "length"],
            stop=stop,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_temperature(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What's the meaning of life?"
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_json(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "Response with a json object:"
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_json_schema(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = (
        "Generate a json containing three fields: an integer field named size, a "
        "boolean field named is_accepted, and a float field named num:"
    )
    max_tokens = 128

    class Schema(BaseModel):
        size: int
        is_accepted: bool
        num: float

    schema_str = json.dumps(Schema.model_json_schema())

    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "response_format": {"type": "json_object", "schema": schema_str},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_logit_bias(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # NOTE: This test only tests that the system does not break on logit bias.
    #       The test does not promise the correctness of logit bias handling.

    prompt = "What's the meaning of life?"
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "logit_bias": {338: -100},  # 338 is " is" in Llama tokenizer.
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_presence_frequency_penalty(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What's the meaning of life?"
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "frequency_penalty": 2.0,
        "presence_penalty": 2.0,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
        )


def test_openai_v1_completions_seed(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = "What's the meaning of life?"
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "seed": 233,
        "debug_config": {"ignore_eos": True},
    }

    response1 = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    response2 = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    for response in [response1, response2]:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=False,
            model=served_model[0],
            object_str="text_completion",
            num_choices=1,
            finish_reasons=["length"],
        )

    text1 = response1.json()["choices"][0]["text"]
    text2 = response2.json()["choices"][0]["text"]
    assert text1 == text2


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_prompt_overlong(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    num_tokens = 1000000
    prompt = [128] * num_tokens
    payload = {
        "model": served_model[0],
        "prompt": prompt,
        "max_tokens": 256,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    error_msg_prefix = (
        f"Request prompt has {num_tokens} tokens in total, larger than the model input length limit"
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


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_completions_invalid_logprobs(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    payload = {
        "model": served_model[0],
        "prompt": "What is the meaning of life?",
        "max_tokens": 256,
        "stream": stream,
        "logprobs": COMPLETION_MAX_TOP_LOGPROBS + 1,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["detail"][0]["msg"].endswith(
        f'"top_logprobs" must be in range [0, {COMPLETION_MAX_TOP_LOGPROBS}]'
    )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_invalid_logprobs(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    payload = {
        "model": served_model[0],
        "messages": [{"role": "user", "content": "Hello! Our project is MLC LLM."}],
        "max_tokens": 256,
        "stream": stream,
        "logprobs": False,
        "top_logprobs": CHAT_COMPLETION_MAX_TOP_LOGPROBS - 1,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["detail"][0]["msg"].endswith(
        '"logprobs" must be True to support "top_logprobs"'
    )

    payload["logprobs"] = True
    payload["top_logprobs"] = CHAT_COMPLETION_MAX_TOP_LOGPROBS + 1

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["detail"][0]["msg"].endswith(
        f'"top_logprobs" must be in range [0, {CHAT_COMPLETION_MAX_TOP_LOGPROBS}]'
    )


def test_openai_v1_completions_unsupported_args(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Right now "best_of" is unsupported.
    best_of = 2
    payload = {
        "model": served_model[0],
        "prompt": "What is the meaning of life?",
        "max_tokens": 256,
        "best_of": best_of,
    }

    response = requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=180)
    error_msg_prefix = 'Request fields "best_of" are not supported right now.'
    expect_error(response.json(), msg_prefix=error_msg_prefix)


def test_openai_v1_completions_request_cancellation(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    # Use a large max_tokens and small timeout to force timeouts.
    payload = {
        "model": served_model[0],
        "prompt": "What is the meaning of life?",
        "max_tokens": 2048,
        "stream": False,
    }
    with pytest.raises(requests.exceptions.Timeout):
        requests.post(OPENAI_V1_COMPLETION_URL, json=payload, timeout=1)

    # The server should still be alive after a request cancelled.
    # We query `v1/models` to validate the server liveness.
    response = requests.get(OPENAI_V1_MODELS_URL, timeout=180).json()

    assert response["object"] == "list"
    models = response["data"]
    assert isinstance(models, list)
    assert len(models) == 1

    model_card = models[0]
    assert isinstance(model_card, dict)
    assert model_card["id"] == served_model[0]
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completions_n(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
    messages: List[Dict[str, str]],
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    n = 3
    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
        "n": n,
        "max_tokens": 300,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=n,
            finish_reasons=["stop", "length"],
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
            num_choices=n,
            finish_reasons=["stop", "length"],
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completions_openai_package(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
    messages: List[Dict[str, str]],
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key="None")
    response = client.chat.completions.create(
        model=served_model[0],
        messages=messages,
        stream=stream,
        logprobs=True,
        top_logprobs=2,
    )
    if not stream:
        check_openai_nonstream_response(
            response.model_dump(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["stop"],
        )
    else:
        responses = []
        for chunk in response:  # pylint: disable=not-an-iterable
            responses.append(chunk.model_dump())
        check_openai_stream_response(
            responses,
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reasons=["stop"],
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_max_tokens(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    messages = [{"role": "user", "content": "Write a novel with at least 500 words."}]
    max_tokens = 16
    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reasons=["length"],
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_json(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    messages = [{"role": "user", "content": "Response with a json object:"}]
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
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
            finish_reasons=["length", "stop"],
            check_json_output=True,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_json_schema(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    prompt = (
        "Generate a json containing three fields: an integer field named size, a "
        "boolean field named is_accepted, and a float field named num:"
    )
    messages = [{"role": "user", "content": prompt}]
    max_tokens = 128

    class Schema(BaseModel):
        size: int
        is_accepted: bool
        num: float

    schema_str = json.dumps(Schema.model_json_schema())

    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object", "schema": schema_str},
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["length", "stop"],
            check_json_output=True,
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
            finish_reasons=["length", "stop"],
            check_json_output=True,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_ignore_eos(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
    stream: bool,
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.

    messages = [{"role": "user", "content": "Write a sentence with less than 20 words."}]
    max_tokens = 128
    payload = {
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "debug_config": {"ignore_eos": True},
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=180)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            is_chat_completion=True,
            model=served_model[0],
            object_str="chat.completion",
            num_choices=1,
            finish_reasons=["length"],
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
            model=served_model[0],
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reasons=["length"],
            completion_tokens=max_tokens,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_openai_v1_chat_completions_system_prompt_wrong_pos(
    served_model: Tuple[str, str],
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
        "model": served_model[0],
        "messages": messages,
        "stream": stream,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=180)
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


def test_debug_dump_event_trace(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.
    # We only check that the request does not fail.
    payload = {"model": served_model[0]}
    response = requests.post(DEBUG_DUMP_EVENT_TRACE_URL, json=payload, timeout=180)
    assert response.status_code == HTTPStatus.OK


def test_metrics(
    served_model: Tuple[str, str],
    launch_server,  # pylint: disable=unused-argument
):
    # `served_model` and `launch_server` are pytest fixtures
    # defined in conftest.py.
    # We only check that the request does not fail.
    metrics_text = requests.get(METRICS_URL, timeout=180).text
    assert "engine_prefill_time_sum" in metrics_text


if __name__ == "__main__":
    model_lib = os.environ.get("MLC_SERVE_MODEL_LIB")
    if model_lib is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL_LIB" not found. '
            "Please set it to model lib compiled by MLC LLM "
            "(e.g., `dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so`)."
        )
    MODEL = (os.path.dirname(model_lib), model_lib)

    test_openai_v1_models(MODEL, None)

    test_openai_v1_completions(MODEL, None, stream=False)
    test_openai_v1_completions(MODEL, None, stream=True)
    test_openai_v1_completions_openai_package(MODEL, None, stream=False)
    test_openai_v1_completions_openai_package(MODEL, None, stream=True)
    test_openai_v1_completions_echo(MODEL, None, stream=False)
    test_openai_v1_completions_echo(MODEL, None, stream=True)
    test_openai_v1_completions_suffix(MODEL, None, stream=False)
    test_openai_v1_completions_suffix(MODEL, None, stream=True)
    test_openai_v1_completions_stop_str(MODEL, None, stream=False)
    test_openai_v1_completions_stop_str(MODEL, None, stream=True)
    test_openai_v1_completions_temperature(MODEL, None, stream=False)
    test_openai_v1_completions_temperature(MODEL, None, stream=True)
    test_openai_v1_completions_logit_bias(MODEL, None, stream=False)
    test_openai_v1_completions_logit_bias(MODEL, None, stream=True)
    test_openai_v1_completions_presence_frequency_penalty(MODEL, None, stream=False)
    test_openai_v1_completions_presence_frequency_penalty(MODEL, None, stream=True)
    test_openai_v1_completions_seed(MODEL, None)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=False)
    test_openai_v1_completions_prompt_overlong(MODEL, None, stream=True)
    test_openai_v1_completions_invalid_logprobs(MODEL, None, stream=False)
    test_openai_v1_completions_invalid_logprobs(MODEL, None, stream=True)
    test_openai_v1_completions_unsupported_args(MODEL, None)
    test_openai_v1_completions_request_cancellation(MODEL, None)

    for msg in CHAT_COMPLETION_MESSAGES:
        test_openai_v1_chat_completions(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions(MODEL, None, stream=True, messages=msg)
        test_openai_v1_chat_completions_n(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions_n(MODEL, None, stream=True, messages=msg)
        test_openai_v1_chat_completions_openai_package(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completions_openai_package(MODEL, None, stream=True, messages=msg)
    test_openai_v1_chat_completions_max_tokens(MODEL, None, stream=False)
    test_openai_v1_chat_completions_max_tokens(MODEL, None, stream=True)
    test_openai_v1_chat_completions_json(MODEL, None, stream=False)
    test_openai_v1_chat_completions_json(MODEL, None, stream=True)
    test_openai_v1_chat_completions_ignore_eos(MODEL, None, stream=False)
    test_openai_v1_chat_completions_ignore_eos(MODEL, None, stream=True)
    test_openai_v1_chat_completions_system_prompt_wrong_pos(MODEL, None, stream=False)
    test_openai_v1_chat_completions_system_prompt_wrong_pos(MODEL, None, stream=True)

    test_debug_dump_event_trace(MODEL, None)
