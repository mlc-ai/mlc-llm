# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import List

import pytest

from mlc_llm.serve import GenerationConfig, MLCEngine

prompts = [
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

test_models = [
    (
        "dist/Llama-2-7b-chat-hf-q0f16-MLC",
        "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so",
    ),
    (
        "dist/rwkv-6-world-1b6-q0f16-MLC",
        "dist/rwkv-6-world-1b6-q0f16-MLC/rwkv-6-world-1b6-q0f16-MLC-cuda.so",
    ),
]


def create_engine(model: str, model_lib: str):
    if "rwkv" in model:
        return MLCEngine(
            model=model,
            model_lib=model_lib,
            mode="server",
            max_batch_size=8,
            max_history_size=1,
        )
    else:
        return MLCEngine(
            model=model,
            model_lib=model_lib,
            mode="server",
            max_total_sequence_length=4096,
        )


@pytest.mark.parametrize("model,model_lib", test_models)
def test_engine_generate(model: str, model_lib: str):
    engine = create_engine(model, model_lib)

    num_requests = 10
    max_tokens = 256
    generation_cfg = GenerationConfig(max_tokens=max_tokens, n=7)

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_cfg.n)] for _ in range(num_requests)
    ]
    for rid in range(num_requests):
        print(f"generating for request {rid}")
        for delta_outputs in engine._generate(prompts[rid], generation_cfg, request_id=str(rid)):
            assert len(delta_outputs) == generation_cfg.n
            for i, delta_output in enumerate(delta_outputs):
                output_texts[rid][i] += delta_output.delta_text

    # Print output.
    print("All finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    engine.terminate()
    del engine


@pytest.mark.parametrize("model,model_lib", test_models)
def test_chat_completion(model: str, model_lib: str):
    # Create engine
    engine = create_engine(model, model_lib)

    num_requests = 2
    max_tokens = 64
    n = 2
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"chat completion for request {rid}")
        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": prompts[rid]}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
            stream=True,
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
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

    engine.terminate()
    del engine


@pytest.mark.parametrize("model,model_lib", test_models)
def test_chat_completion_non_stream(model: str, model_lib: str):
    engine = create_engine(model, model_lib)

    num_requests = 2
    max_tokens = 64
    n = 2
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"chat completion for request {rid}")
        response = engine.chat.completions.create(
            messages=[{"role": "user", "content": prompts[rid]}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
        )
        for choice in response.choices:
            assert choice.message.role == "assistant"
            output_texts[rid][choice.index] += choice.message.content

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    engine.terminate()
    del engine


@pytest.mark.parametrize("model,model_lib", test_models)
def test_completion(model: str, model_lib: str):
    engine = create_engine(model, model_lib)

    num_requests = 2
    max_tokens = 128
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"completion for request {rid}")
        for response in engine.completions.create(
            prompt=prompts[rid],
            model=model,
            max_tokens=max_tokens,
            n=n,
            ignore_eos=True,
            request_id=str(rid),
            stream=True,
        ):
            for choice in response.choices:
                output_texts[rid][choice.index] += choice.text

    # Print output.
    print("Completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    engine.terminate()
    del engine


@pytest.mark.parametrize("model,model_lib", test_models)
def test_completion_non_stream(model: str, model_lib: str):
    engine = create_engine(model, model_lib)

    num_requests = 2
    max_tokens = 128
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"completion for request {rid}")
        response = engine.completions.create(
            prompt=prompts[rid],
            model=model,
            max_tokens=max_tokens,
            n=n,
            ignore_eos=True,
            request_id=str(rid),
        )
        for choice in response.choices:
            output_texts[rid][choice.index] += choice.text

    # Print output.
    print("Completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    engine.terminate()
    del engine


if __name__ == "__main__":
    for model, model_lib in test_models:
        test_engine_generate(model, model_lib)
        test_chat_completion(model, model_lib)
        test_chat_completion_non_stream(model, model_lib)
        test_completion(model, model_lib)
        test_completion_non_stream(model, model_lib)
