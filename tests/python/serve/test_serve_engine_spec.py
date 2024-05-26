# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals
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


@require_test_model(
    "Llama-2-7b-chat-hf-q0f16-MLC",
    "Llama-2-7b-chat-hf-q4f16_1-MLC",
)
def test_engine_basic(model: str, small_model: str):
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_tokens - 1). Then check the output of each request.
    """

    # Hyperparameters for tests (you can try different combinations).
    num_requests = len(prompts)  # [4, 8, 10]
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
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[small_model],
            speculative_mode="small_draft",
        ),
        request_stream_callback=fcallback,
    )

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


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_eagle_basic(model: str):
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_tokens - 1). Then check the output of each request.
    - Use Eagle model as speculative model
    """

    # Hyperparameters for tests (you can try different combinations).
    num_requests = len(prompts)  # [4, 8, 10]
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
    small_model = "dist/Eagle-llama2-7b-chat-q0f16-MLC"
    small_model_lib = "dist/Eagle-llama2-7b-chat-q0f16-MLC/Eagle-llama2-7b-chat-q0f16-MLC-cuda.so"
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="eagle",
            spec_draft_length=2,
        ),
        request_stream_callback=fcallback,
    )

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


@require_test_model(
    "Llama-2-7b-chat-hf-q0f16-MLC",
    "Llama-2-7b-chat-hf-q4f16_1-MLC",
)
def test_engine_continuous_batching_1(model: str, small_model: str):
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_tokens - 1). Then check the output
    of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = len(prompts)  # [4, 8, 10]
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
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[small_model],
            speculative_mode="small_draft",
        ),
        request_stream_callback=timer.callback_getter(),
    )

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


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_engine_eagle_continuous_batching_1(model: str):
    """Test engine **with continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have a random maximum generation length. So each
    request keeps generating until reaching the maximum length.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + the maximum max_tokens - 1). Then check the output
    of each request.
    """

    # Hyperparameters for tests (you can try different combinations)
    num_requests = len(prompts)  # [4, 8, 10]
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
    small_model = "dist/Eagle-llama2-7b-chat-q4f16_1-MLC"
    small_model_lib = (
        "dist/Eagle-llama2-7b-chat-q4f16_1-MLC/Eagle-llama2-7b-chat-q4f16_1-MLC-cuda.so"
    )
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="eagle",
        ),
        request_stream_callback=timer.callback_getter(),
    )

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


def compare_output_text(output_text1, output_text2):
    if isinstance(output_text1, list) and isinstance(output_text2, list):
        for item1, item2 in zip(output_text1, output_text2):
            if not compare_output_text(item1, item2):
                return False
    elif output_text1 != output_text2:
        print(output_text1)
        print(output_text2)
        return False
    return True


@require_test_model(
    "Llama-2-7b-chat-hf-q0f16-MLC",
    "Llama-2-7b-chat-hf-q4f16_1-MLC",
)
def test_engine_generate(model: str, small_model: str, compare_precision=False):
    # Create engine
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[small_model],
            speculative_mode="small_draft",
        ),
    )

    num_requests = 10
    max_tokens = 256

    # Generate output.
    if compare_precision:
        print("compare precision")
        generation_config = GenerationConfig(
            temperature=0.0, top_p=0, max_tokens=1024, stop_token_ids=[2], n=1
        )
        engine_single_model = SyncMLCEngine(
            model=model,
            mode="server",
            engine_config=EngineConfig(
                max_total_sequence_length=4096,
            ),
        )
        output_texts_single_model, _ = engine_single_model.generate(
            prompts[:num_requests], generation_config
        )
        for req_id, outputs in enumerate(output_texts_single_model):
            print(f"Prompt {req_id}: {prompts[req_id]}")
            if len(outputs) == 1:
                print(f"Output {req_id}:{outputs[0]}\n")
            else:
                for i, output in enumerate(outputs):
                    print(f"Output {req_id}({i}):{output}\n")
        # TODO: Add pytorch precision
    else:
        generation_config = GenerationConfig(max_tokens=max_tokens, n=3)
    output_texts, _ = engine.generate(prompts[:num_requests], generation_config)
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")
    if compare_precision:
        precision_flag = compare_output_text(output_texts, output_texts_single_model)
        if precision_flag:
            print(f"Accuracy verification succeed\n")
        else:
            print(f"Accuracy verification failed\n")


@require_test_model("Llama-2-7b-chat-hf-q0f16-MLC")
def test_engine_eagle_generate(model: str):
    # Create engine
    small_model = "dist/Eagle-llama2-7b-chat-q4f16_1-MLC"
    small_model_lib = (
        "dist/Eagle-llama2-7b-chat-q4f16_1-MLC/Eagle-llama2-7b-chat-q4f16_1-MLC-cuda.so"
    )
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="eagle",
        ),
    )

    num_requests = 10
    max_tokens = 256

    # Generate output.
    output_texts, _ = engine.generate(
        prompts[:num_requests], GenerationConfig(max_tokens=max_tokens, n=3)
    )
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("Llama-2-13b-chat-hf-q4f16_1-MLC")
def test_engine_efficiency(model: str):
    """Test engine speculative decoding efficiency."""

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 512
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
        engine_config=EngineConfig(max_total_sequence_length=4096),
        request_stream_callback=fcallback,
    )

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
        metrics = eg.metrics()
        print("engine name:", name)
        if name == "Speculative Decoding":
            print("spec decode metrics:", metrics["spec_decode"])
        print("engine total decode time:", metrics["engine_decode_time_sum"])
        print()


@require_test_model(
    "Llama-2-13b-chat-hf-q4f16_1-MLC",
    "Llama-2-7b-chat-hf-q4f16_1-MLC",
)
def test_engine_spec_efficiency(model: str, small_model: str):
    """Test engine speculative decoding efficiency."""

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 512
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
    spec_engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[small_model],
            spec_draft_length=6,
            speculative_mode="small_draft",
        ),
        request_stream_callback=fcallback,
    )

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
        metrics = eg.metrics()
        print("engine name:", name)
        if name == "Speculative Decoding":
            print("total draft tokens:", metrics["sum_num_draft_tokens"])
            print("total accepted tokens:", metrics["sum_num_accepted_tokens"])
            print(
                "Accept rate:",
                metrics["sum_num_accepted_tokens"] / (1e-10 + metrics["sum_num_draft_tokens"]),
            )
        print("engine total decode time:", metrics["engine_decode_time_sum"])
        print()


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_engine_eagle_spec_efficiency(model: str):
    """Test engine speculative decoding efficiency."""

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 512
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
    small_model = "dist/Eagle-llama2-7b-chat-q0f16-MLC"
    small_model_lib = "dist/Eagle-llama2-7b-chat-q0f16-MLC/Eagle-llama2-7b-chat-q0f16-MLC-cuda.so"
    spec_engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            spec_draft_length=6,
            speculative_mode="eagle",
        ),
        request_stream_callback=fcallback,
    )

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
        metrics = eg.metrics()
        print("engine name:", name)
        if name == "Speculative Decoding":
            print("spec decode:", metrics["spec_decode"])
        print("engine total decode time:", metrics["engine_decode_time_sum"])
        print()


if __name__ == "__main__":
    test_engine_basic()
    test_engine_eagle_basic()
    test_engine_continuous_batching_1()
    test_engine_eagle_continuous_batching_1()
    test_engine_generate(compare_precision=True)
    test_engine_eagle_generate()
    test_engine_efficiency()
    test_engine_spec_efficiency()
    test_engine_eagle_spec_efficiency()
