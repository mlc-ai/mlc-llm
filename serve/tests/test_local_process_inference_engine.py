from typing import Optional, Union

from mlc_serve.engine import (
    ChatMessage,
    DebugOptions,
    FinishReason,
    Request,
    RequestId,
    RequestOutput,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.engine.local import LocalProcessInferenceEngine
from mlc_serve.engine.model_module import (
    ConversationTemplate,
    DecodeRequest,
    KVCache,
    KVCacheManager,
    ModelModule,
    PrefillRequest,
    SequenceId,
    TextGenerationResult,
    TextGenerator,
    Tokenizer,
)


class DummyTokenizer:
    @property
    def eos_token_id(self):
        return 2

    def encode(self, text: str, **kwargs) -> list[int]:
        return [1] * len(text.split())

    def decode(self, tokens: list[int], **kwargs) -> str:
        return "test " * len(tokens)


class DummyConversationTemplate:
    def apply(self, messages: list[ChatMessage]) -> str:
        return " ".join(m.content for m in messages if m.content is not None)


class DummyCache:
    def __init__(self, max_cached_tokens: int):
        self.max_cached_tokens = max_cached_tokens
        self.cached_requests = dict[RequestId, int]()


class DummyCacheManager:
    def __init__(self, max_cached_tokens: int):
        self.cache = DummyCache(max_cached_tokens)

    def get_cache(self) -> KVCache:
        return self.cache

    def allocate(self, request_id: RequestId, num_tokens: int) -> bool:
        self.cache.cached_requests[request_id] = num_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")

    def extend(self, sequence_id: SequenceId, new_tokens: int) -> bool:
        if sequence_id.sequence_index > 0:
            raise RuntimeError("Multiple generated sequences not supported")
        self.cache.cached_requests[sequence_id.request_id] += new_tokens
        if self.get_free_space() < 0:
            raise RuntimeError("Cache out of space")

    def free(self, sequence_id: SequenceId):
        if sequence_id.sequence_index > 0:
            raise RuntimeError("Multiple generated sequences not supported")
        del self.cache.cached_requests[sequence_id.request_id]

    def get_kv_cache_size(self) -> int:
        return self.cache.max_cached_tokens

    def get_free_space(self) -> int:
        return self.cache.max_cached_tokens - sum(self.cache.cached_requests.values())

    def get_max_new_tokens(self) -> int:
        if not self.cache.cached_requests:
            return self.get_kv_cache_size()
        return self.get_free_space() // len(self.cache.cached_requests)


class DummyTextGenerator:
    def generate(
        self,
        requests: list[Union[PrefillRequest, DecodeRequest]],
        kv_cache: KVCache,
    ) -> list[TextGenerationResult]:
        result = []
        for req in requests:
            if isinstance(req, DecodeRequest):
                request_id = req.sequence_id.request_id
                if req.sequence_id.sequence_index > 0:
                    raise RuntimeError("Multiple generated sequences not supported")
            else:
                request_id = req.request_id

            if len(req.token_ids) > kv_cache.cached_requests[request_id]:
                raise RuntimeError(f"Cache out of space for request {req.request_id}")
            result.append(
                TextGenerationResult(
                    sequence_id=SequenceId(
                        request_id=request_id,
                        sequence_index=0,
                    ),
                    generated_tokens=[1],
                    error=None,
                )
            )
        return result


class DummaryModelModule:
    def __init__(self, max_cached_tokens: int):
        self.tokenizer = DummyTokenizer()
        self.conversation_template = DummyConversationTemplate()
        self.text_generator = DummyTextGenerator()
        self.cache_manager = DummyCacheManager(max_cached_tokens)


def create_messages(prompt) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=prompt)]


def get_output_for_request(
    outputs: list[RequestOutput], request_id: RequestId
) -> Optional[RequestOutput]:
    for o in outputs:
        if o.request_id == request_id:
            return o
    return None


def test_single_request():
    engine = LocalProcessInferenceEngine(DummaryModelModule(30))
    request_id = "1"
    engine.add(
        [
            Request(
                request_id=request_id,
                messages=create_messages("test prompt"),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=5),
            ),
        ]
    )

    step = engine.step()

    assert step.outputs[0].request_id == request_id
    assert step.outputs[0].error is None
    assert len(step.outputs) == 1


def test_single_request_step_to_finish():
    engine = LocalProcessInferenceEngine(DummaryModelModule(30))

    request_id = "1"
    engine.add(
        [
            Request(
                request_id=request_id,
                messages=create_messages("test prompt"),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=10),
            ),
        ]
    )

    steps = [engine.step() for _ in range(11)]

    assert steps[-1].outputs[0].request_id == request_id
    assert steps[-1].outputs[0].sequences[0].finish_reason == FinishReason.Length
    assert len(steps[-1].outputs) == 1


def test_multiple_requests_wait_queue():
    engine = LocalProcessInferenceEngine(DummaryModelModule(20))

    request_id_1 = "1"
    request_id_2 = "2"

    engine.add(
        [
            Request(
                request_id=request_id_1,
                messages=create_messages("test " * 11),  # 11 tokens
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=2),
            ),
        ]
    )

    engine.add(
        [
            Request(
                request_id=request_id_2,
                messages=create_messages("test " * 11),  # 11 tokens
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=2),
            ),
        ]
    )

    steps = [engine.step() for _ in range(3)]

    assert len(steps[0].outputs) == 1
    assert steps[0].outputs[0].request_id == request_id_1

    assert len(steps[1].outputs) == 1
    assert steps[1].outputs[0].request_id == request_id_1

    assert len(steps[2].outputs) == 2
    assert (
        get_output_for_request(steps[2].outputs, request_id_1)
        .sequences[0]
        .finish_reason
        == FinishReason.Length
    )
    assert get_output_for_request(steps[2].outputs, request_id_2) is not None


def test_multiple_requests_preempt():
    engine = LocalProcessInferenceEngine(DummaryModelModule(30), min_decode_steps=1)

    request_id_1 = "1"
    request_id_2 = "2"

    engine.add(
        [
            Request(
                request_id=request_id_1,
                messages=create_messages("test " * 10),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=7),
            ),
        ]
    )

    engine.add(
        [
            Request(
                request_id=request_id_2,
                messages=create_messages("test " * 10),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=7),
            ),
        ]
    )

    steps = [engine.step() for _ in range(8)]

    assert len(steps[0].outputs) == 2
    assert len(steps[5].outputs) == 2

    assert len(steps[6].outputs) == 1
    finished_request_id = steps[6].outputs[0].request_id

    assert len(steps[7].outputs) == 2
    assert get_output_for_request(steps[7].outputs, finished_request_id).is_finished
