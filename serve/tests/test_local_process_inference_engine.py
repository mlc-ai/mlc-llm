from mlc_serve.engine import (
    Request,
    RequestId,
    SamplingParams,
    SequenceGenerationRequest,
    SequenceGenerationResponse,
    StoppingCriteria,
)
from mlc_serve.engine.local import LocalProcessInferenceEngine


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] * len(text.split())

    def decode(self, tokens: list[int]) -> str:
        return "test " * len(tokens)


class DummyModelExecutor:
    def __init__(self, max_cached_tokens: int):
        self.max_cached_tokens = max_cached_tokens
        self.cached_requests = dict[RequestId, int]()

    def generate(
        self, requests: list[SequenceGenerationRequest]
    ) -> list[SequenceGenerationResponse]:
        result = []
        for req in requests:
            if (
                req.start_position + len(req.token_ids)
                > self.cached_requests[req.request_id]
            ):
                raise RuntimeError(f"Cache out of space for request {req.request_id}")
            result.append(
                SequenceGenerationResponse(
                    request_id=req.request_id,
                    token_ids=[1],
                    error=None,
                )
            )
        return result

    def allocate(self, request_id: RequestId, num_tokens: int) -> bool:
        self.cached_requests[request_id] = num_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")

    def extend(self, request_id: RequestId, new_tokens: int) -> bool:
        self.cached_requests[request_id] += new_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")

    def free(self, request_id: RequestId):
        del self.cached_requests[request_id]

    def get_kv_cache_size(self) -> int:
        return self.max_cached_tokens

    def get_free_space(self) -> int:
        return self.max_cached_tokens - sum(self.cached_requests.values())

    def get_max_new_tokens(self) -> int:
        if not self.cached_requests:
            return self.get_kv_cache_size()
        return self.get_free_space() / len(self.cached_requests)


def test_single_request():
    engine = LocalProcessInferenceEngine(DummyModelExecutor(30), DummyTokenizer())

    request_id = engine.add(
        [
            Request(
                prompt="test prompt",
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=5),
            ),
        ]
    )[0]

    step = engine.step()

    assert step.outputs[0].request_id == request_id
    assert len(step.outputs) == 1
    assert not step.errors


def test_single_request_step_to_finish():
    engine = LocalProcessInferenceEngine(DummyModelExecutor(30), DummyTokenizer())

    request_id = engine.add(
        [
            Request(
                prompt="test prompt",
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=10),
            ),
        ]
    )[0]

    steps = [engine.step() for _ in range(10)]

    assert steps[-1].outputs[0].request_id == request_id
    assert steps[-1].outputs[0].finish_reason == "length"
    assert len(steps[-1].outputs) == 1


def test_multiple_requests_wait_queue():
    engine = LocalProcessInferenceEngine(DummyModelExecutor(20), DummyTokenizer())

    request_id_1 = engine.add(
        [
            Request(
                prompt="test " * 11,  # 11 tokens
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=2),
            ),
        ]
    )[0]

    request_id_2 = engine.add(
        [
            Request(
                prompt="test " * 11,  # 11 tokens
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=2),
            ),
        ]
    )[0]

    steps = [engine.step() for _ in range(3)]

    assert len(steps[0].outputs) == 1
    assert steps[0].outputs[0].request_id == request_id_1

    assert len(steps[1].outputs) == 1
    assert steps[1].outputs[0].request_id == request_id_1
    assert steps[1].outputs[0].finish_reason == "length"

    assert len(steps[2].outputs) == 1
    assert steps[2].outputs[0].request_id == request_id_2


def test_multiple_requests_preempt():
    engine = LocalProcessInferenceEngine(DummyModelExecutor(30), DummyTokenizer())

    request_id_1 = engine.add(
        [
            Request(
                prompt="test " * 10,
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=7),
            ),
        ]
    )[0]

    request_id_2 = engine.add(
        [
            Request(
                prompt="test " * 10,
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=7),
            ),
        ]
    )[0]

    steps = [engine.step() for _ in range(8)]

    assert len(steps[0].outputs) == 2
    assert len(steps[5].outputs) == 2

    assert len(steps[6].outputs) == 1
    assert steps[6].outputs[0].finish_reason == "length"
    assert len(steps[7].outputs) == 1
    assert steps[7].outputs[0].finish_reason == "length"
    assert steps[6].outputs[0].request_id != steps[7].outputs[0].request_id
