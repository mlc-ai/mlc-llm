# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import Callable, List, Optional

import numpy as np

from mlc_chat.serve import GenerationConfig, KVCacheConfig, Request, data
from mlc_chat.serve.engine import Engine, ModelInfo

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
    stop_token: Optional[int] = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    max_new_tokens_low: int = 256,
    max_new_tokens_high: int = 257,
) -> List[Request]:
    assert num_requests >= 0 and num_requests <= len(prompts)

    stop_tokens = [stop_token] if stop_token is not None else []
    requests = []
    for req_id, prompt in zip(range(num_requests), prompts):
        max_new_tokens = np.random.randint(max_new_tokens_low, max_new_tokens_high)
        requests.append(
            Request(
                request_id=str(req_id),
                inputs=data.TextData(prompt),
                generation_config=GenerationConfig(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                ),
            )
        )
    return requests


def test_engine_basic():
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_new_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_new_tokens - 1). Then check the output of each request.
    """

    # Initialize model loading info and KV cache config
    model = ModelInfo("Llama-2-7b-chat-hf-q4f16_1")
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_new_tokens: int = 256  # [32, 128, 256]
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(request_id: str, token_data: data.TokenData, finish_reason: Optional[str]):
        outputs[int(request_id)] += token_data.token_ids

    # Create engine
    engine = Engine(model, kv_cache_config, fcallback)

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens_low=max_new_tokens,
        max_new_tokens_high=max_new_tokens + 1,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_new_tokens - 1
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
    requests + the maximum max_new_tokens - 1). Then check the output
    of each request.
    """

    # Initialize model loading info and KV cache config
    model = ModelInfo("Llama-2-7b-chat-hf-q4f16_1")
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    max_new_tokens_low = 128
    max_new_tokens_high = 384
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]
    finish_time = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[str, data.TokenData, Optional[str]], None]:
            def fcallback(
                request_id: str, token_data: data.TokenData, finish_reason: Optional[str]
            ):
                if finish_reason is not None:
                    print(f"Request {request_id} finished at step {self.timer}.")
                outputs[int(request_id)] += token_data.token_ids
                finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = Engine(model, kv_cache_config, timer.callback_getter())

    # Create requests
    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens_low=max_new_tokens_low,
        max_new_tokens_high=max_new_tokens_high,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = (
        num_requests + max(request.generation_config.max_new_tokens for request in requests) - 1
    )
    # Run steps
    for step in range(num_steps):
        timer.step()
        assert timer.timer == step
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")
        assert fin_time == request.generation_config.max_new_tokens - 1


def test_engine_continuous_batching_2():
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the stop token. So each request keeps generating
    until having the stop token or reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_new_tokens - 1). Then check the output
    of each request.
    """

    # Initialize model loading info and KV cache config
    model = ModelInfo("Llama-2-7b-chat-hf-q4f16_1")
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    stop_token = 2
    max_new_tokens = 512
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]
    finish_time = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[str, data.TokenData, Optional[str]], None]:
            def fcallback(
                request_id: str, token_data: data.TokenData, finish_reason: Optional[str]
            ):
                if finish_reason is not None:
                    print(f"Request {request_id} finished at step {self.timer}.")
                outputs[int(request_id)] += token_data.token_ids
                finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = Engine(model, kv_cache_config, timer.callback_getter())

    # Create requests
    requests = create_requests(
        num_requests,
        stop_token=stop_token,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens_low=max_new_tokens,
        max_new_tokens_high=max_new_tokens + 1,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_new_tokens - 1
    # Run steps
    for step in range(num_steps):
        timer.step()
        assert timer.timer == step
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        if fin_time < num_requests + max_new_tokens - 2:
            print(f"Request {req_id} ends early on the stop token")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


def test_engine_continuous_batching_3():
    """Test engine **with continuous batching**.

    - Add requests randomly between time [0, 200).
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` until all requests finish.
    Then check the output of each request.
    """

    # Initialize model loading info and KV cache config
    model = ModelInfo("Llama-2-7b-chat-hf-q4f16_1")
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    stop_token = 2
    max_new_tokens_low = 64
    max_new_tokens_high = 192
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]
    finish_time = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1
        finished_requests: int = 0

        def callback_getter(self) -> Callable[[str, data.TokenData, Optional[str]], None]:
            def fcallback(
                request_id: str, token_data: data.TokenData, finish_reason: Optional[str]
            ):
                if finish_reason is not None:
                    print(f"Request {request_id} finished at step {self.timer}.")
                    self.finished_requests += 1
                outputs[int(request_id)] += token_data.token_ids
                finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

        def all_finished(self) -> bool:
            return self.finished_requests == num_requests

    # Create engine
    timer = CallbackTimer()
    engine = Engine(model, kv_cache_config, timer.callback_getter())

    # Create requests
    requests = create_requests(
        num_requests,
        stop_token=stop_token,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens_low=max_new_tokens_low,
        max_new_tokens_high=max_new_tokens_high,
    )

    # Assign the time to add requests to engine
    request_add_time = [np.random.randint(0, 200) for _ in range(num_requests)]

    # Run steps
    while not timer.all_finished():
        timer.step()

        # Add requests to engine
        for req_id, add_time in enumerate(request_add_time):
            if add_time == timer.timer:
                print(f"add request {req_id} at step {timer.timer}")
                engine.add_request(requests[req_id])

        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        print(f"Finish time: {fin_time}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


def test_engine_generate():
    # Initialize model loading info and KV cache config
    model = ModelInfo("Llama-2-7b-chat-hf-q4f16_1")
    kv_cache_config = KVCacheConfig(page_size=16)
    # Create engine
    engine = Engine(model, kv_cache_config)

    num_requests = 10
    max_new_tokens = 256

    # Generate output.
    outputs = engine.generate(
        prompts[:num_requests], GenerationConfig(max_new_tokens=max_new_tokens)
    )
    for req_id, output in enumerate(outputs):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        print(f"Output {req_id}:{output}\n")


if __name__ == "__main__":
    test_engine_basic()
    test_engine_continuous_batching_1()
    test_engine_continuous_batching_2()
    test_engine_continuous_batching_3()
    test_engine_generate()
