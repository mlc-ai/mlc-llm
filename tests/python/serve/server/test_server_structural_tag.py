# pylint: disable=line-too-long
"""
Test script for structural tag in chat completion. To run this script, use the following command:
- start a new shell session, run
  mlc_llm serve --model "YOUR_MODEL" (e.g. ./dist/Llama-2-7b-chat-hf-q0f16-MLC)
- start another shell session, run this file
  MLC_SERVE_MODEL="YOUR_MODEL" python tests/python/serve/server/test_server_structural_tag.py
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches
import json
import os
import re
from typing import Any, Dict, List, Optional

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
    assert response["model"] == model
    assert response["object"] == object_str

    choices = response["choices"]
    assert isinstance(choices, list)
    assert len(choices) == num_choices
    for idx, choice in enumerate(choices):
        assert choice["index"] == idx
        assert choice["finish_reason"] in finish_reason

        find_format_start = set()
        beg_tag_start = set()
        message = choice["message"]["content"]
        print("Outputs:\n-----------")
        print(message, flush=True)
        pattern1 = r"<CALL--->(.*?)\|(.*?)\|End<---(.*?)>"
        pattern2 = r"<call--->(.*?)\|(.*?)\|End<---(.*?)>"
        # check format
        for match in re.finditer(pattern1, message):
            find_format_start.add(match.start())
            check_format(match.group(1), match.group(3), "CALL", match.group(2))
        for match in re.finditer(pattern2, message):
            find_format_start.add(match.start())
            check_format(match.group(1), match.group(3), "call", match.group(2))
        for match in re.finditer(r"<CALL--->", message):
            beg_tag_start.add(match.start())
        for match in re.finditer(r"<call--->", message):
            beg_tag_start.add(match.start())
        assert find_format_start == beg_tag_start

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
        find_format_start = set()
        beg_tag_start = set()
        print("Outputs:\n-----------")
        print(output, flush=True)
        pattern1 = r"<CALL--->(.*?)\|(.*?)\|End<---(.*?)>"
        pattern2 = r"<call--->(.*?)\|(.*?)\|End<---(.*?)>"
        # check format
        for match in re.finditer(pattern1, output):
            find_format_start.add(match.start())
            check_format(match.group(1), match.group(3), "CALL", match.group(2))
        for match in re.finditer(pattern2, output):
            find_format_start.add(match.start())
            check_format(match.group(1), match.group(3), "call", match.group(2))
        for match in re.finditer(r"<CALL--->", output):
            beg_tag_start.add(match.start())
        for match in re.finditer(r"<call--->", output):
            beg_tag_start.add(match.start())
        assert find_format_start == beg_tag_start


def check_format(name_beg: str, name_end: str, beg_tag: str, schema: str):
    try:
        paras: Dict[str, Any] = json.loads(schema)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        assert False
    assert "hash_code" in paras
    assert "hash_code" in schema
    hash_code = paras["hash_code"]
    assert hash_code in CHECK_INFO
    info = CHECK_INFO[hash_code]
    assert name_beg == info["name"]
    assert name_end == info["name"]
    assert beg_tag == info["beg_tag"]
    for key in info["required"]:
        assert key in paras


# NOTE: the end-tag format and the hash_code number is been hidden in the SYSTEM_PROMPT.
# By checking whether the end tag and hash code can be generated correctly without any prompts, the correctness of the structural tag can be verified.

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to find the weather for, e.g. 'San Francisco'",
            },
            "state": {
                "type": "string",
                "description": "the two-letter abbreviation for the state that the city is"
                " in, e.g. 'CA' which would mean 'California'",
            },
            "unit": {
                "type": "string",
                "description": "The unit to fetch the temperature in",
                "enum": ["celsius", "fahrenheit"],
            },
            "hash_code": {
                "type": "string",
            },
        },
        "required": ["city", "state", "unit", "hash_code"],
    },
}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{
    "name": "get_current_date",
    "description": "Get the current date and time for a given timezone",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
            },
            "hash_code": {
                "type": "string",
            },
        },
        "required": ["timezone", "hash_code"],
    },
}
If a you choose to call a function ONLY reply in the following format:
<{start_tag}--->{function_name}|{parameters}|{end_tag}<---{function_name}>
where
start_tag => `<CALL` or `<call`
parameters => a JSON dict with the function argument name as key and function argument value as value.
Here is an example,
<CALL--->example_function_name|{"example_name": "example_value"}...
or
<call--->example_function_name|{"example_name": "example_value"}...
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
You are a helpful assistant.""",
}

STRUCTURAL_TAGS = {
    "triggers": ["<CALL--->", "<call--->"],
    "tags": [
        {
            "begin": "<CALL--->get_current_weather|",
            "schema": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'San Francisco'",
                        },
                        "state": {
                            "type": "string",
                            "description": "the two-letter abbreviation for the state that the city is"
                            " in, e.g. 'CA' which would mean 'California'",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                        "hash_code": {"const": 1234},
                    },
                    "required": ["city", "state", "unit", "hash_code"],
                }
            ),
            "end": "|End<---get_current_weather>",
        },
        {
            "begin": "<CALL--->get_current_date|",
            "schema": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                        },
                        "hash_code": {"const": 2345},
                    },
                    "required": ["timezone", "hash_code"],
                }
            ),
            "end": "|End<---get_current_date>",
        },
        {
            "begin": "<call--->get_current_weather|",
            "schema": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'San Francisco'",
                        },
                        "state": {
                            "type": "string",
                            "description": "the two-letter abbreviation for the state that the city is"
                            " in, e.g. 'CA' which would mean 'California'",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                        "hash_code": {"const": 3456},
                    },
                    "required": ["city", "state", "unit", "hash_code"],
                }
            ),
            "end": "|End<---get_current_weather>",
        },
        {
            "begin": "<call--->get_current_date|",
            "schema": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                        },
                        "hash_code": {"const": 4567},
                    },
                    "required": ["timezone", "hash_code"],
                }
            ),
            "end": "|End<---get_current_date>",
        },
    ],
}

CHECK_INFO = {
    1234: {
        "name": "get_current_weather",
        "beg_tag": "CALL",
        "required": ["city", "state", "unit", "hash_code"],
    },
    2345: {
        "name": "get_current_date",
        "beg_tag": "CALL",
        "required": ["timezone", "hash_code"],
    },
    3456: {
        "name": "get_current_weather",
        "beg_tag": "call",
        "required": ["city", "state", "unit", "hash_code"],
    },
    4567: {
        "name": "get_current_date",
        "beg_tag": "call",
        "required": ["timezone", "hash_code"],
    },
}

CHAT_COMPLETION_MESSAGES = [
    # messages #0
    [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time.",
        },
    ],
    # messages #1
    [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": "You are in New York. Please get the current weather.",
        },
    ],
    # messages #2
    [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ],
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("messages", CHAT_COMPLETION_MESSAGES)
def test_openai_v1_chat_completion_structural_tag(
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
        "response_format": {
            "type": "structural_tag",
            "tags": STRUCTURAL_TAGS["tags"],
            "triggers": STRUCTURAL_TAGS["triggers"],
        },
        "max_tokens": 1024,
    }

    response = requests.post(OPENAI_V1_CHAT_COMPLETION_URL, json=payload, timeout=60)
    if not stream:
        check_openai_nonstream_response(
            response.json(),
            model=served_model,
            object_str="chat.completion",
            num_choices=1,
            finish_reason=["stop"],
        )
    else:
        responses = []
        for chunk in response.iter_lines(chunk_size=512):
            if not chunk or chunk == b"data: [DONE]":
                continue
            responses.append(json.loads(chunk.decode("utf-8")[6:]))
        check_openai_stream_response(
            responses,
            model=served_model,
            object_str="chat.completion.chunk",
            num_choices=1,
            finish_reason="stop",
        )

    print(f"-----------\nCheck for stream={stream} is passed!\n")


if __name__ == "__main__":
    MODEL = os.environ.get("MLC_SERVE_MODEL")
    if MODEL is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL" not found. '
            "Please set it to model compiled by MLC LLM "
            "(e.g., `./dist/Llama-2-7b-chat-hf-q0f16-MLC`) "
        )

    for msg in CHAT_COMPLETION_MESSAGES:
        test_openai_v1_chat_completion_structural_tag(MODEL, None, stream=False, messages=msg)
        test_openai_v1_chat_completion_structural_tag(MODEL, None, stream=True, messages=msg)
