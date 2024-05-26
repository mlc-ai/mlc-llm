# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
import json
from typing import List

import pytest
from pydantic import BaseModel

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.protocol.openai_api_protocol import RequestResponseFormat as ResponseFormat
from mlc_llm.serve import AsyncMLCEngine
from mlc_llm.serve.sync_engine import SyncMLCEngine
from mlc_llm.testing import require_test_model

prompts_list = [
    "Generate a JSON string containing 20 objects:",
    "Generate a JSON containing a non-empty list:",
    "Generate a JSON with 5 elements:",
]


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_batch_generation_with_grammar(model: str):
    # Create engine
    engine = SyncMLCEngine(
        model=model,
        mode="server",
    )

    prompt_len = len(prompts_list)
    prompts = prompts_list * 3

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
    generation_config_json_no_stop_token = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        response_format=ResponseFormat(type="json_object"),
    )
    all_generation_configs = (
        [generation_config_no_json] * prompt_len
        + [generation_config_json] * prompt_len
        + [generation_config_json_no_stop_token] * prompt_len
    )

    # Generate output.
    output_texts, _ = engine.generate(prompts, all_generation_configs)
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_batch_generation_with_schema(model: str):
    # Create engine
    engine = SyncMLCEngine(model=model, mode="server")

    prompt = (
        "Generate a json containing three fields: an integer field named size, a "
        "boolean field named is_accepted, and a float field named num:"
    )
    repeat_cnt = 3
    prompts = [prompt] * repeat_cnt * 2

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

    class Schema(BaseModel):
        size: int
        is_accepted: bool
        num: float

    schema_str = json.dumps(Schema.model_json_schema())

    generation_config_json = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=[2],
        response_format=ResponseFormat(type="json_object", schema=schema_str),
    )

    all_generation_configs = [generation_config_no_json] * repeat_cnt + [
        generation_config_json
    ] * repeat_cnt

    # Generate output.
    output_texts, _ = engine.generate(prompts, all_generation_configs)
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}: {outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}): {output}\n")


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
async def run_async_engine(model: str):
    # Create engine
    async_engine = AsyncMLCEngine(model=model, mode="server")

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
        async_engine: AsyncMLCEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"Start generation task for request {request_id}")
        rid = int(request_id)
        async for delta_outputs in async_engine._generate(
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
            response_format=ResponseFormat(type="text", schema="{}"),
        )


if __name__ == "__main__":
    test_batch_generation_with_grammar()
    test_async_engine()
    test_generation_config_error()
