# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
import json
import random
from typing import Dict, List, Literal

from pydantic import BaseModel

from mlc_llm.protocol.debug_protocol import DebugConfig
from mlc_llm.protocol.openai_api_protocol import ChatCompletionResponse
from mlc_llm.serve import AsyncMLCEngine, MLCEngine
from mlc_llm.testing import require_test_model

LLAMA_2_MODEL = "Llama-2-7b-chat-hf-q4f16_1-MLC"
LLAMA_3_MODEL = "Meta-Llama-3-8B-Instruct-q4f16_1-MLC"


@require_test_model(LLAMA_3_MODEL)
def test_batch_generation_with_grammar(model: str):
    # Engine
    engine = MLCEngine(model=model, mode="server")

    # Inputs
    system_prompt = "You are a helpful assistant. Always respond only with json."
    prompts_list = [
        "Generate a JSON string containing 20 objects:",
        "Generate a JSON containing a non-empty list:",
        "Generate a JSON with 5 elements:",
        "Generate a JSON with a number list, counting from 1 to 20:",
    ]

    repeat = 3
    top_p = 0.9
    temperature = 0.6
    max_tokens = 4096

    # non-json output
    responses_text: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        for p in prompts_list:
            print(f"Start generation task for request {len(responses_text)}")
            responses_text.append(
                engine.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    response_format={"type": "text"},
                    top_p=top_p,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=random.randint(0, 1 << 30),
                    extra_body={"debug_config": DebugConfig(grammar_execution_mode="constraint")},
                )
            )

    print("Text output")
    for req_id, response in enumerate(responses_text):
        prompt = prompts_list[req_id % len(prompts_list)]
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output
    responses_json: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        for p in prompts_list:
            print(f"Start generation task for request {len(responses_json)}")
            responses_json.append(
                engine.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    response_format={"type": "json_object"},
                    top_p=top_p,
                    temperature=temperature,
                    seed=random.randint(0, 1 << 30),
                )
            )

    print("JSON output")
    for req_id, response in enumerate(responses_json):
        prompt = prompts_list[req_id % len(prompts_list)]
        output = str(response.choices[0].message.content)
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")
        json.loads(output)

    print("Engine metrics:", engine.metrics())

    engine.terminate()


@require_test_model(LLAMA_3_MODEL)
def test_batch_generation_with_schema(model: str):
    # Create engine
    engine = MLCEngine(model=model, mode="server")

    class Product(BaseModel):
        product_id: int
        is_available: bool
        price: float
        is_featured: Literal[True]
        category: Literal["Electronics", "Clothing", "Food"]
        tags: List[str]
        stock: Dict[str, int]

    schema_str = json.dumps(Product.model_json_schema())

    system_prompt = (
        "You are a helpful assistant. Always respond only with JSON based on the "
        f"following JSON schema: {schema_str}."
    )
    prompt = "Generate a JSON that describes the product according to the given JSON schema."

    repeat = 8
    top_p = 0.9
    temperature = 0.6
    max_tokens = 4096

    # non-json output
    responses_text: List[ChatCompletionResponse] = []
    for i in range(repeat):
        print(f"Start generation task for request {i}")
        responses_text.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "text"},
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=random.randint(0, 1 << 30),
                extra_body={"debug_config": DebugConfig(grammar_execution_mode="constraint")},
            )
        )

    print("Text output")
    for req_id, response in enumerate(responses_text):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output without schema
    responses_json: List[ChatCompletionResponse] = []
    for i in range(repeat):
        print(f"Start generation task for request {i}")
        responses_json.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=random.randint(0, 1 << 30),
                extra_body={"debug_config": DebugConfig(grammar_execution_mode="constraint")},
            )
        )

    print("JSON output")
    for req_id, response in enumerate(responses_json):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output with schema
    responses_schema: List[ChatCompletionResponse] = []
    for i in range(repeat):
        print(f"Start generation task for request {i}")
        responses_schema.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": schema_str},
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=random.randint(0, 1 << 30),
                extra_body={"debug_config": DebugConfig(grammar_execution_mode="constraint")},
            )
        )

    print("JSON Schema output")
    for req_id, response in enumerate(responses_schema):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    print("Engine metrics:", engine.metrics())

    engine.terminate()


@require_test_model(LLAMA_3_MODEL)
def test_batch_generation_jump_forward(model: str, jump_forward: bool = True, repeat: int = 1):
    # Create engine
    engine = MLCEngine(model=model, mode="server")

    class Product(BaseModel):
        product_id: int
        is_available: bool
        price: float
        is_featured: Literal[True]
        category: Literal["Electronics", "Clothing", "Food"]
        tags: List[str]
        stock: Dict[str, int]

    schema_str = json.dumps(Product.model_json_schema())

    system_prompt = (
        "You are a helpful assistant. Always respond only with JSON based on the "
        f"following JSON schema: {schema_str}."
    )
    prompt = "Generate a JSON that describes the product according to the given JSON schema."

    top_p = 0.9
    temperature = 0.6
    max_tokens = 4096
    grammar_execution_mode = "jump_forward" if jump_forward else "constraint"

    # json output with schema
    responses: List[ChatCompletionResponse] = []
    for i in range(repeat):
        print(f"Start generation task for request {i}")
        responses.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": schema_str},
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=random.randint(0, 1 << 30),
                extra_body={
                    "debug_config": DebugConfig(grammar_execution_mode=grammar_execution_mode)
                },
            )
        )

    print(f"Jump forward: {jump_forward}, Repeat: {repeat}")
    for req_id, response in enumerate(responses):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    print("Engine metrics:", engine.metrics())

    engine.terminate()


@require_test_model(LLAMA_3_MODEL)
async def run_async_engine(
    model: str,
    mode: Literal["text", "json", "schema"] = "schema",
    jump_forward: bool = True,
    num_requests: int = 8,
):
    # Create engine
    async_engine = AsyncMLCEngine(model=model, mode="server")

    class Product(BaseModel):
        product_id: int
        is_available: bool
        price: float
        is_featured: Literal[True]
        category: Literal["Electronics", "Clothing", "Food"]
        tags: List[str]
        stock: Dict[str, int]

    schema_str = json.dumps(Product.model_json_schema())

    if mode == "text":
        response_format = {"type": "text"}
    elif mode == "json":
        response_format = {"type": "json_object"}
    elif mode == "schema":
        response_format = {"type": "json_object", "schema": schema_str}

    system_prompt = (
        "You are a helpful assistant. Always respond only with JSON based on the "
        f"following JSON schema: {schema_str}."
    )
    prompt = "Generate a JSON that describes the product according to the given JSON schema."

    top_p = 0.9
    temperature = 0.6
    max_tokens = 4096
    grammar_execution_mode = "jump_forward" if jump_forward else "constraint"

    responses = ["" for _ in range(num_requests)]

    async def generate_task(prompt: str, request_id: str):
        print(f"Start generation task for request {request_id}")
        rid = int(request_id)
        async for response in await async_engine.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=random.randint(0, 1 << 30),
            stream=True,
            extra_body={"debug_config": DebugConfig(grammar_execution_mode=grammar_execution_mode)},
        ):
            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.delta.role == "assistant"
            assert isinstance(choice.delta.content, str)
            responses[rid] += choice.delta.content

    tasks = [
        asyncio.create_task(generate_task(prompt, request_id=str(i))) for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    print(f"Mode: {mode}, Jump forward: {jump_forward}, Num requests: {num_requests}")
    for req_id, output in enumerate(responses):
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    print("Engine metrics:", await async_engine.metrics())

    async_engine.terminate()
    del async_engine


def test_async_engine(
    mode: Literal["text", "json", "schema"] = "schema",
    jump_forward: bool = True,
    num_requests: int = 8,
):
    asyncio.run(run_async_engine(mode, jump_forward, num_requests))


if __name__ == "__main__":
    test_batch_generation_with_grammar()
    test_batch_generation_with_schema()
    test_batch_generation_jump_forward(False)
    test_batch_generation_jump_forward(True)
    test_async_engine("schema", False, 1)
    test_async_engine("schema", True, 1)
    test_async_engine("schema", False, 8)
    test_async_engine("schema", True, 8)
