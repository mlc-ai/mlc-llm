import json

import pytest
import tvm

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.testing import require_test_model

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def check_error_handling(engine, expect_str, **params):
    """Check error handling in raw completion API"""
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream_options": {"include_usage": True},
    }
    body.update(params)

    for response in engine._raw_chat_completion(
        json.dumps(body), include_usage=False, request_id="123"
    ):
        if response.choices[0].finish_reason is not None:
            break
    if response.choices[0].finish_reason != "error":
        raise RuntimeError(f"expect the request {params} to hit an error")

    if expect_str not in response.choices[0].delta.content:
        raise RuntimeError(
            f"expect '{expect_str}' in error msg, " f"but get '{response.choices[0].delta.content}'"
        )


# NOTE: we only need tokenizers in folder
# launch time of mock test is fast so we can put it in unittest
@require_test_model("Llama-3-8B-Instruct-q4f16_1-MLC")
def test_chat_completion_misuse(model: str):
    engine = JSONFFIEngine(model, tvm.cpu(), model_lib="mock://echo")
    # Test malformed requests.
    for response in engine._raw_chat_completion(
        "malformed_string", include_usage=False, request_id="123"
    ):
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "error"
    # check parameters
    check_error_handling(engine, "should be non-negative", temperature=-1)
    check_error_handling(engine, "in range [0, 1]", top_p=100)
    check_error_handling(engine, "frequency_penalty", frequency_penalty=100)


def check_normal_param_passing(engine):
    json_schema = """
    {"properties": {"result": {"items": {"type": "Integer"}, "title": "Result", "type": "array"}},
      "required": ["result"], "title": "Output", "type": "object"}
    """
    param_dict = {
        "top_p": 0.6,
        "temperature": 0.8,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }
    usage = None
    for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
        response_format={"type": "json_object", "schema": json_schema},
        **param_dict,  # type: ignore
    ):
        if response.usage is not None:
            usage = response.usage

    # echo mock will echo back the generation config
    for k, v in param_dict.items():
        assert usage.extra[k] == v, f"{k} mismatch"
    assert "response_format" in usage.extra
    assert usage.extra["response_format"]["type"] == "json_object"
    assert "schema" in usage.extra["response_format"]


def check_n_generation(engine):
    hit_set = set()
    for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
        n=3,
    ):
        for choice in response.choices:
            hit_set.add(choice.index)
    for i in range(3):
        assert i in hit_set, f"{i} not in n generation"


@require_test_model("Llama-3-8B-Instruct-q4f16_1-MLC")
def test_chat_completion_api(model: str):
    engine = JSONFFIEngine(model, tvm.cpu(), model_lib="mock://echo")
    check_normal_param_passing(engine)
    check_n_generation(engine)


if __name__ == "__main__":
    test_chat_completion_api()
    test_chat_completion_misuse()
