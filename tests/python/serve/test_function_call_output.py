"""Parsing the model output of a chat completion with tools into tool calls."""

import pytest

from mlc_llm.serve import engine_base

pytestmark = [pytest.mark.unittest]


@pytest.mark.parametrize(
    "output_text",
    [
        "",
        '[{"name": "get_current_weather", "arguments": {"location": "Pittsburgh, PA"}}]',
        "[1, 2]",
    ],
)
def test_unparsable_output_is_an_error(output_text):
    """Output that is not a function call must come back as an error, not raise."""
    finish_reasons = ["tool_calls"]
    use_function_calling, tool_calls_list = engine_base.process_function_call_output(
        [output_text], finish_reasons
    )
    assert use_function_calling
    assert finish_reasons == ["error"]
    assert tool_calls_list == [[]]


def test_function_call_output_is_parsed():
    finish_reasons = ["tool_calls"]
    _, tool_calls_list = engine_base.process_function_call_output(
        ['get_current_weather(location="Pittsburgh, PA")'], finish_reasons
    )
    assert finish_reasons == ["tool_calls"]
    (tool_call,) = tool_calls_list[0]
    assert tool_call.function.name == "get_current_weather"
    assert tool_call.function.arguments == {"location": "Pittsburgh, PA"}
