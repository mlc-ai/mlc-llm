# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import Callable, List, Optional

import numpy as np

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import Request, RequestStreamOutput, data
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine
from mlc_llm.testing import require_test_model

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
    engine: SyncMLCEngine,
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
            engine.create_request(
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


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_basic(model: str):
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_tokens - 1). Then check the output of each request.
    """

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 256  # [32, 128, 256]
    np.random.seed(0)

    # Output list
    outputs: List[List[int]] = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            assert len(stream_outputs) == 1
            outputs[int(request_id)] += stream_outputs[0].delta_token_ids

    # Create engine
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        request_stream_callback=fcallback,
    )

    # Create requests
    requests = create_requests(
        engine,
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


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_continuous_batching_1(model: str):
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_tokens - 1). Then check the output
    of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    max_tokens_low = 128
    max_tokens_high = 384
    np.random.seed(0)

    # Output list
    outputs: List[List[int]] = [[] for _ in range(num_requests)]
    finish_time: List[Optional[int]] = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, stream_outputs = delta_output.unpack()
                    assert len(stream_outputs) == 1
                    if stream_outputs[0].finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                    outputs[int(request_id)] += stream_outputs[0].delta_token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        request_stream_callback=timer.callback_getter(),
    )

    # Create requests
    requests = create_requests(
        engine,
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
        assert (
            fin_time == request.generation_config.max_tokens - 1
        ), f"finish time = {fin_time}, max tokens = {request.generation_config.max_tokens - 1}"


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_continuous_batching_2(model: str):
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the stop token. So each request keeps generating
    until having the stop token or reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_tokens - 1). Then check the output
    of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    stop_token_id = 2
    max_tokens = 512
    np.random.seed(0)

    # Output list
    outputs: List[List[int]] = [[] for _ in range(num_requests)]
    finish_time: List[Optional[int]] = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, stream_outputs = delta_output.unpack()
                    assert len(stream_outputs) == 1
                    if stream_outputs[0].finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                    outputs[int(request_id)] += stream_outputs[0].delta_token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        request_stream_callback=timer.callback_getter(),
    )

    # Create requests
    requests = create_requests(
        engine,
        num_requests,
        stop_token_id=stop_token_id,
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
        timer.step()
        assert timer.timer == step
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        if fin_time < num_requests + max_tokens - 2:
            print(f"Request {req_id} ends early on the stop token")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_continuous_batching_3(model: str):
    """Test engine **with continuous batching**.

    - Add requests randomly between time [0, 200).
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` until all requests finish.
    Then check the output of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    stop_token_id = 2
    max_tokens_low = 64
    max_tokens_high = 192
    np.random.seed(0)

    # Output list
    outputs: List[List[int]] = [[] for _ in range(num_requests)]
    finish_time: List[Optional[int]] = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1
        finished_requests: int = 0

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, stream_outputs = delta_output.unpack()
                    assert len(stream_outputs) == 1
                    if stream_outputs[0].finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                        self.finished_requests += 1
                    outputs[int(request_id)] += stream_outputs[0].delta_token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

        def all_finished(self) -> bool:
            return self.finished_requests == num_requests

    # Create engine
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        request_stream_callback=timer.callback_getter(),
    )

    # Create requests
    requests = create_requests(
        engine,
        num_requests,
        stop_token_id=stop_token_id,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens_low,
        max_tokens_high=max_tokens_high,
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


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_generate(model: str):
    # Create engine
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(max_total_sequence_length=4096),
    )

    num_requests = 10
    max_tokens = 256

    # Generate output.
    output_texts, _ = engine.generate(
        prompts[:num_requests], GenerationConfig(max_tokens=max_tokens, n=7)
    )
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_hybrid_prefill(model: str):
    """Test engine **with hybrid prefill**.

    - Add each single request step by step.
    - All requests have the same generation length. But due to hybrid prefill,
    the earlier request will decode with later request prefill, in single step.
    So each request lasts the same steps, and stops generation step by step as well.
    - Engine keeps running `step` for the generation length, to finish the last request.
    Then check the output of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = 10  # [4, 8, 10]
    temperature = 0.9  # [0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.00  # [1.0, 1.01]
    max_tokens = 15
    np.random.seed(0)

    # Output list
    outputs: List[List[int]] = [[] for _ in range(num_requests)]
    finish_time: List[Optional[int]] = [None] * num_requests

    # Define the callback class for request generation results
    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, stream_outputs = delta_output.unpack()
                    assert len(stream_outputs) == 1
                    if stream_outputs[0].finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                    outputs[int(request_id)] += stream_outputs[0].delta_token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    # Create engine
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        request_stream_callback=timer.callback_getter(),
        engine_config=EngineConfig(prefill_mode="hybrid"),
    )

    # Create requests
    requests = create_requests(
        engine,
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    # Add all requests to engine step by step
    for step, request in enumerate(requests):
        engine.add_request(request)
        timer.step()
        assert timer.timer == step
        engine.step()

    # Run steps
    for step in range(max_tokens):
        timer.step()
        assert timer.timer == step + num_requests
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")
        assert (
            fin_time == req_id + request.generation_config.max_tokens - 1
        ), f"finish time = {fin_time}, max tokens = {req_id + request.generation_config.max_tokens - 1}"


if __name__ == "__main__":
    test_engine_basic()
    test_engine_continuous_batching_1()
    test_engine_continuous_batching_2()
    test_engine_continuous_batching_3()
    test_engine_generate()
    test_engine_hybrid_prefill()
