# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals
import asyncio
from typing import List

<<<<<<< HEAD
from mlc_llm.serve import (
    AsyncThreadedEngine,
    EngineMode,
    GenerationConfig,
    KVCacheConfig,
)
from mlc_llm.serve.engine import ModelInfo
=======
from mlc_llm.serve import AsyncLLMEngine, GenerationConfig, SpeculativeMode
>>>>>>> upstream/main

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
    small_model = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    small_model_lib_path = (
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC/Llama-2-7b-chat-hf-q4f16_1-MLC-cuda.so"
    )
    async_engine = AsyncLLMEngine(
        model=model,
        model_lib_path=model_lib_path,
        mode="server",
        additional_models=[small_model + ":" + small_model_lib_path],
        speculative_mode=SpeculativeMode.SMALL_DRAFT,
    )

    num_requests = 10
    max_tokens = 256
    generation_cfg = GenerationConfig(max_tokens=max_tokens)

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_cfg.n)] for _ in range(num_requests)
    ]

    async def generate_task(
        async_engine: AsyncLLMEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"generate task for request {request_id}")
        rid = int(request_id)
<<<<<<< HEAD
        async for delta_outputs in async_engine.generate(
=======
        async for delta_outputs in async_engine._generate(
>>>>>>> upstream/main
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
