# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
from typing import List

import pytest

from mlc_chat.serve import Engine, GenerationConfig, KVCacheConfig
from mlc_chat.serve.async_engine import AsyncThreadedEngine
from mlc_chat.serve.config import ResponseFormat
from mlc_chat.serve.engine import ModelInfo

prompts_list = [
    "Generate a JSON string containing 20 objects:",
    "Generate a JSON containing a list:",
    "Generate a JSON with 5 elements:",
]
model_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
model_lib_path = "dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so"


def test_batch_generation_with_grammar():
    # Initialize model loading info and KV cache config
    model = ModelInfo(model_path, model_lib_path=model_lib_path)
    kv_cache_config = KVCacheConfig(page_size=16)
    # Create engine
    engine = Engine(model, kv_cache_config)

    prompts = prompts_list * 2

    temperature = 1
    repetition_penalty = 1
    max_tokens = 512
    generation_config_no_json = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=[2],
        response_format=ResponseFormat(type="text"),
    )
    generation_config_json = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=[2],
        response_format=ResponseFormat(type="json_object"),
    )
    all_generation_configs = [generation_config_no_json] * 3 + [generation_config_json] * 3

    # Generate output.
    output_texts, _ = engine.generate(prompts, all_generation_configs)
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


async def run_async_engine():
    # Initialize model loading info and KV cache config
    model = ModelInfo(model_path, model_lib_path=model_lib_path)
    kv_cache_config = KVCacheConfig(page_size=16)
    # Create engine
    async_engine = AsyncThreadedEngine(model, kv_cache_config, enable_tracing=True)

    prompts = prompts_list * 20

    max_tokens = 256
    temperature = 1
    repetition_penalty = 1
    max_tokens = 512
    generation_config = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=[2],
        response_format=ResponseFormat(type="json_object"),
    )

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_config.n)] for _ in range(len(prompts))
    ]

    async def generate_task(
        async_engine: AsyncThreadedEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"Start generation task for request {request_id}")
        rid = int(request_id)
        async for delta_outputs in async_engine.generate(
            prompt, generation_cfg, request_id=request_id
        ):
            assert len(delta_outputs) == generation_cfg.n
            for i, delta_output in enumerate(delta_outputs):
                output_texts[rid][i] += delta_output.delta_text

    tasks = [
        asyncio.create_task(
            generate_task(async_engine, prompts[i], generation_config, request_id=str(i))
        )
        for i in range(len(prompts))
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

    print(async_engine.trace_recorder.dump_json(), file=open("tmpfiles/tmp.json", "w"))

    async_engine.terminate()


def test_async_engine():
    asyncio.run(run_async_engine())


def test_generation_config_error():
    with pytest.raises(ValueError):
        GenerationConfig(
            temperature=1.0,
            repetition_penalty=1.0,
            max_tokens=128,
            stop_token_ids=[2],
            response_format=ResponseFormat(type="text", json_schema="{}"),
        )


if __name__ == "__main__":
    test_batch_generation_with_grammar()
    test_async_engine()
    test_generation_config_error()
