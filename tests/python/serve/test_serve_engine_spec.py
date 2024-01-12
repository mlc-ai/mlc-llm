# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import Callable, List, Optional

import numpy as np

from mlc_chat.serve import (
    Engine,
    EngineMode,
    GenerationConfig,
    KVCacheConfig,
    Request,
    RequestStreamOutput,
    data,
)
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


def create_requests(
    num_requests: int,
    stop_token_id: Optional[int] = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    max_tokens_low: int = 256,
    max_tokens_high: int = 257,
) -> List[Request]:
    assert num_requests >= 0 and num_requests <= len(prompts)

    stop_token_ids = [stop_token_id] if stop_token_id is not None else []
    requests = []
    for req_id, prompt in zip(range(num_requests), prompts):
        max_tokens = np.random.randint(max_tokens_low, max_tokens_high)
        requests.append(
            Request(
                request_id=str(req_id),
                inputs=data.TextData(prompt),
                generation_config=GenerationConfig(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                ),
            )
        )
    return requests


def test_engine_basic():
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_tokens - 1). Then check the output of each request.
    """

    # Initialize model loading info and KV cache config
    ssm = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q4f16_1-MLC/Llama-2-7b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    model = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q0f16-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)
    engine_mode = EngineMode(enable_speculative=True)

    # Hyperparameters for tests (you can try different combinations).
    num_requests = len(prompts)  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 256  # [32, 128, 256]
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, delta_tokens, _ = delta_output.unpack()
            outputs[int(request_id)] += delta_tokens.token_ids

    # Create engine
    engine = Engine([model, ssm], kv_cache_config, engine_mode, fcallback)

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    # Run steps
    for step in range(num_steps):
        engine.step()

    for req_id, output in enumerate(outputs):
        print(f"Prompt {req_id}: {requests[req_id].inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


def test_engine_continuous_batching_1():
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_tokens - 1). Then check the output
    of each request.
    """

    # Initialize model loading info and KV cache config
    ssm = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q4f16_1-MLC/Llama-2-7b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    model = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q0f16-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)
    engine_mode = EngineMode(enable_speculative=True)

    # Hyperparameters for tests (you can try different combinations)
    num_requests = len(prompts)  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    max_tokens_low = 128
    max_tokens_high = 384
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]
    finish_time = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, delta_tokens, finish_reason = delta_output.unpack()
                    if finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                    outputs[int(request_id)] += delta_tokens.token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = Engine([model, ssm], kv_cache_config, engine_mode, timer.callback_getter())

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens_low,
        max_tokens_high=max_tokens_high,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max(request.generation_config.max_tokens for request in requests) - 1
    # Run steps
    for step in range(num_steps):
        timer.step()
        assert timer.timer == step
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")
        # assert fin_time == request.generation_config.max_tokens - 1


def test_engine_generate():
    # Initialize model loading info and KV cache config
    ssm = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q4f16_1-MLC/Llama-2-7b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    model = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q0f16-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)
    engine_mode = EngineMode(enable_speculative=True)
    # Create engine
    engine = Engine([model, ssm], kv_cache_config, engine_mode)

    num_requests = 10
    max_tokens = 256

    # Generate output.
    outputs = engine.generate(prompts[:num_requests], GenerationConfig(max_tokens=max_tokens))
    for req_id, output in enumerate(outputs):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        print(f"Output {req_id}:{output}\n")


def test_engine_efficiency():
    """Test engine speculative decoding efficiency."""

    # Initialize model loading info and KV cache config
    model = ModelInfo(
        "dist/Llama-2-13b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-13b-chat-hf-q4f16_1-MLC/Llama-2-13b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 512
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, delta_tokens, _ = delta_output.unpack()
            outputs[int(request_id)] += delta_tokens.token_ids

    # Create engine
    engine = Engine(model, kv_cache_config, request_stream_callback=fcallback)

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    # Run steps
    for step in range(num_steps):
        engine.step()

    for eg, name in zip([engine], ["Normal Deconding"]):
        stats = eg.stats()
        print("engine name:", name)
        if name == "Speculative Decoding":
            print("total draft tokens:", stats["total_draft_tokens"])
            print("total accepted tokens:", stats["total_accepted_tokens"])
            print(
                "Accept rate:",
                stats["total_accepted_tokens"] / (1e-10 + stats["total_draft_tokens"]),
            )
        print("engine total decode time:", stats["engine_total_decode_time"])
        print()


def test_engine_spec_efficiency():
    """Test engine speculative decoding efficiency."""

    # Initialize model loading info and KV cache config
    ssm = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-7b-chat-hf-q4f16_1-MLC/Llama-2-7b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    # If Flashinfer allows head_dim < 128, we can test this model
    # ssm = ModelInfo(
    #     "dist/TinyLlama-1.1B-Chat-v1.0-q0f16-MLC",
    #     model_lib_path="dist/TinyLlama-1.1B-Chat-v1.0-q0f16-MLC/TinyLlama-1.1B-Chat-v1.0-q0f16-MLC-cuda.so",
    # )
    model = ModelInfo(
        "dist/Llama-2-13b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/Llama-2-13b-chat-hf-q4f16_1-MLC/Llama-2-13b-chat-hf-q4f16_1-MLC-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)
    engine_mode = EngineMode(enable_speculative=True, spec_draft_length=6)

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 512
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, delta_tokens, _ = delta_output.unpack()
            outputs[int(request_id)] += delta_tokens.token_ids

    # Create engine
    spec_engine = Engine([model, ssm], kv_cache_config, engine_mode, fcallback)

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    # Add all requests to engine
    for request in requests:
        spec_engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    # Run steps
    for step in range(num_steps):
        spec_engine.step()

    for eg, name in zip([spec_engine], ["Speculative Decoding"]):
        stats = eg.stats()
        print("engine name:", name)
        if name == "Speculative Decoding":
            print("total draft tokens:", stats["total_draft_tokens"])
            print("total accepted tokens:", stats["total_accepted_tokens"])
            print(
                "Accept rate:",
                stats["total_accepted_tokens"] / (1e-10 + stats["total_draft_tokens"]),
            )
        print("engine total decode time:", stats["engine_total_decode_time"])
        print()


if __name__ == "__main__":
    test_engine_basic()
    test_engine_continuous_batching_1()
    test_engine_generate()
    test_engine_efficiency()
    test_engine_spec_efficiency()
