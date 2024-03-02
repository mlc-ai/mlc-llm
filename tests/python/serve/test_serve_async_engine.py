# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
from typing import List

from mlc_chat.serve import AsyncThreadedEngine, GenerationConfig, KVCacheConfig
from mlc_chat.serve.engine import ModelInfo

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
    # Initialize model loading info and KV cache config
    model = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q0f16-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16, max_total_sequence_length=4096)
    # Create engine
    async_engine = AsyncThreadedEngine(model, kv_cache_config)

    num_requests = 10
    max_tokens = 256
    generation_cfg = GenerationConfig(max_tokens=max_tokens, n=3)

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_cfg.n)] for _ in range(num_requests)
    ]

    async def generate_task(
        async_engine: AsyncThreadedEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"generate task for request {request_id}")
        rid = int(request_id)
        async for delta_outputs in async_engine.generate(
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


if __name__ == "__main__":
    asyncio.run(test_engine_generate())
