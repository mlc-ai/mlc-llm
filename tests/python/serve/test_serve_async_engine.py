# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
from typing import List

from mlc_llm.serve import AsyncEngine, GenerationConfig

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


async def test_engine_generate():
    # Create engine
    model = "dist/Llama-2-7b-chat-hf-q0f16-MLC"
    model_lib_path = "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so"
    async_engine = AsyncEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
    )

    num_requests = 10
    max_tokens = 256
    generation_cfg = GenerationConfig(max_tokens=max_tokens, n=7)

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_cfg.n)] for _ in range(num_requests)
    ]

    async def generate_task(
        async_engine: AsyncEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"generate task for request {request_id}")
        rid = int(request_id)
        async for delta_outputs in async_engine._generate(
            prompt, generation_cfg, request_id=request_id
        ):
            assert len(delta_outputs) == generation_cfg.n
            for i, delta_output in enumerate(delta_outputs):
                output_texts[rid][i] += delta_output.delta_text

    tasks = [
        asyncio.create_task(
            generate_task(async_engine, prompts[i], generation_cfg, request_id=str(i))
        )
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("All finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()
    del async_engine


async def test_chat_completion():
    # Create engine
    model = "dist/Llama-2-7b-chat-hf-q0f16-MLC"
    model_lib_path = "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so"
    async_engine = AsyncEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
    )

    num_requests = 2
    max_tokens = 32
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    async def generate_task(prompt: str, request_id: str):
        print(f"generate chat completion task for request {request_id}")
        rid = int(request_id)
        async for response in await async_engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=request_id,
            stream=True,
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                output_texts[rid][choice.index] += choice.delta.content

    tasks = [
        asyncio.create_task(generate_task(prompts[i], request_id=str(i)))
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()
    del async_engine


async def test_chat_completion_non_stream():
    # Create engine
    model = "dist/Llama-2-7b-chat-hf-q0f16-MLC"
    model_lib_path = "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so"
    async_engine = AsyncEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
    )

    num_requests = 2
    max_tokens = 32
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    async def generate_task(prompt: str, request_id: str):
        print(f"generate chat completion task for request {request_id}")
        rid = int(request_id)
        response = await async_engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=request_id,
        )
        for choice in response.choices:
            assert choice.message.role == "assistant"
            output_texts[rid][choice.index] += choice.message.content

    tasks = [
        asyncio.create_task(generate_task(prompts[i], request_id=str(i)))
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()
    del async_engine


async def test_completion():
    # Create engine
    model = "dist/Llama-2-7b-chat-hf-q0f16-MLC"
    model_lib_path = "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so"
    async_engine = AsyncEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
    )

    num_requests = 2
    max_tokens = 128
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    async def generate_task(prompt: str, request_id: str):
        print(f"generate completion task for request {request_id}")
        rid = int(request_id)
        async for response in await async_engine.completions.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            n=n,
            ignore_eos=True,
            request_id=request_id,
            stream=True,
        ):
            for choice in response.choices:
                output_texts[rid][choice.index] += choice.text

    tasks = [
        asyncio.create_task(generate_task(prompts[i], request_id=str(i)))
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("Completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()
    del async_engine


async def test_completion_non_stream():
    # Create engine
    model = "dist/Llama-2-7b-chat-hf-q0f16-MLC"
    model_lib_path = "dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so"
    async_engine = AsyncEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
    )

    num_requests = 2
    max_tokens = 128
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    async def generate_task(prompt: str, request_id: str):
        print(f"generate completion task for request {request_id}")
        rid = int(request_id)
        response = await async_engine.completions.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            n=n,
            ignore_eos=True,
            request_id=request_id,
        )
        for choice in response.choices:
            output_texts[rid][choice.index] += choice.text

    tasks = [
        asyncio.create_task(generate_task(prompts[i], request_id=str(i)))
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("Completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()
    del async_engine


if __name__ == "__main__":
    asyncio.run(test_engine_generate())
    asyncio.run(test_chat_completion())
    asyncio.run(test_chat_completion_non_stream())
    asyncio.run(test_completion())
    asyncio.run(test_completion_non_stream())
