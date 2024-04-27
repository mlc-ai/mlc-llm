from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

from mlc_llm.json_ffi import JSONFFIEngine

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
        for response in engine.chat_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompts[rid]}]}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
            tools=tools,
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                assert isinstance(choice.delta.content[0], Dict)
                assert choice.delta.content[0]["type"] == "text"
                output_texts[rid][choice.index] += choice.delta.content[0]["text"]

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


def test_chat_completion():
    # Create engine.
    model = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    engine = JSONFFIEngine(
        model,
        max_total_sequence_length=1024,
    )

    run_chat_completion(engine, model)

    # Test malformed requests.
    for response in engine._handle_chat_completion("malformed_string", n=1, request_id="123"):
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "error"

    engine.terminate()


def test_reload_reset_unload():
    # Create engine.
    model = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    engine = JSONFFIEngine(
        model,
        max_total_sequence_length=1024,
    )

    # Run chat completion before and after reload/reset.
    run_chat_completion(engine, model)
    engine._test_reload()
    run_chat_completion(engine, model)
    engine._test_reset()
    run_chat_completion(engine, model)
    engine._test_unload()

    engine.terminate()


def test_function_calling():
    model = "dist/gorilla-openfunctions-v1-q4f16_1-MLC"
    model_lib_path = (
        "dist/gorilla-openfunctions-v1-q4f16_1-MLC/gorilla-openfunctions-v1-q4f16_1-cuda.so"
    )
    engine = JSONFFIEngine(
        model,
        model_lib_path=model_lib_path,
        max_total_sequence_length=1024,
    )

    # run function calling
    run_chat_completion(engine, model, function_calling_prompts, tools)

    engine.terminate()


if __name__ == "__main__":
    test_chat_completion()
    test_reload_reset_unload()
    test_function_calling()
