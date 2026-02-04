import json
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.testing import require_test_model

# test category "engine_feature"
pytestmark = [pytest.mark.engine_feature]


chat_completion_prompts = [
    "What is the meaning of life?",
    "Introduce the history of Pittsburgh to me. Please elaborate in detail.",
    "Write a three-day Seattle travel plan. Please elaborate in detail.",
    "What is Alaska famous of? Please elaborate in detail.",
    "What is the difference between Lambda calculus and Turing machine? Please elaborate in detail.",
    "What are the necessary components to assemble a desktop computer? Please elaborate in detail.",
    "Why is Vitamin D important to human beings? Please elaborate in detail.",
    "Where is milk tea originated from? Please elaborate in detail.",
    "Where is the southernmost place in United States? Please elaborate in detail.",
    "Do you know AlphaGo? What capabilities does it have, and what achievements has it got? Please elaborate in detail.",
]

function_calling_prompts = [
    "What is the temperature in Pittsburgh, PA?",
    "What is the temperature in Tokyo, JP?",
    "What is the temperature in Pittsburgh, PA and Tokyo, JP?",
]

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


def run_chat_completion(
    engine: JSONFFIEngine,
    model: str,
    prompts: List[str] = chat_completion_prompts,
    tools: Optional[List[Dict]] = None,
):
    num_requests = 2
    max_tokens = 64
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"chat completion for request {rid}")
        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompts[rid]}]}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
            tools=tools,
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                assert isinstance(choice.delta.content, str)
                output_texts[rid][choice.index] += choice.delta.content

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


def run_json_schema_function_calling(
    engine: JSONFFIEngine,
    model: str,
    prompts: List[str] = function_calling_prompts,
    tools: Optional[List[Dict]] = None,
):
    num_requests = 2
    max_tokens = 64
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    class ToolCall(BaseModel):
        name: str
        arguments: Dict[str, str]

    class Schema(BaseModel):
        tool_calls: List[ToolCall]

    schema_str = json.dumps(Schema.model_json_schema())
    print("Schema str", schema_str)

    for rid in range(num_requests):
        print(f"chat completion for request {rid}")
        for response in engine.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a function calling AI model. You are provided with function signatures within "
                    "<tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make "
                    f"assumptions about what values to plug into functions. Here are the available tools: <tools> {json.dumps(tools)} </tools> "
                    "Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10. "
                    "Calling multiple functions at once can overload the system and increase cost so call one function at a time please. "
                    "If you plan to continue with analysis, always call another function. Return a valid json object (using double "
                    f"quotes) in the following schema: {schema_str}",
                },
                {"role": "user", "content": [{"type": "text", "text": prompts[rid]}]},
            ],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
            response_format={"type": "json_object", "schema": schema_str},
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                assert isinstance(choice.delta.content, str)
                output_texts[rid][choice.index] += choice.delta.content

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_chat_completion(model):
    # Create engine.
    engine = JSONFFIEngine(model)

    run_chat_completion(engine, model)

    # Test malformed requests.
    for response in engine._raw_chat_completion(
        "malformed_string", include_usage=False, request_id="123"
    ):
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "error"

    engine.terminate()


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_reload_reset_unload(model):
    # Create engine.
    engine = JSONFFIEngine(model)

    # Run chat completion before and after reload/reset.
    run_chat_completion(engine, model)
    engine._test_reload()
    run_chat_completion(engine, model)
    engine._test_reset()
    run_chat_completion(engine, model)
    engine._test_unload()

    engine.terminate()


@require_test_model("Hermes-2-Pro-Mistral-7B-q4f16_1-MLC")
def test_json_schema_with_system_prompt(model):
    engine = JSONFFIEngine(model)

    # run function calling
    run_json_schema_function_calling(engine, model, function_calling_prompts, tools)

    engine.terminate()


if __name__ == "__main__":
    test_chat_completion()
    test_reload_reset_unload()
    test_json_schema_with_system_prompt()
