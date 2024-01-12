from typing import Optional, Union

from mlc_serve.engine import (
    ChatMessage,
    FinishReason,
    Request,
    RequestId,
    RequestOutput,
    SamplingParams,
    StoppingCriteria,
    get_engine_config
)
from mlc_serve.model.base import ModelArtifactConfig
from mlc_serve.engine.model_module import (
    DecodeRequest,
    KVCache,
    PrefillRequest,
    SequenceId,
    TextGenerationResult,
)

from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.engine.staging_engine import StagingInferenceEngine

from mlc_serve.model.dummy_model import (
    DummyModelModule,
    DummyTokenizerModule,
)


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
    engine = SynchronousInferenceEngine(DummyModelModule(30))
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
    engine = SynchronousInferenceEngine(DummyModelModule(30))

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

    steps = [engine.step() for _ in range(10)]

    assert steps[-1].outputs[0].request_id == request_id
    assert steps[-1].outputs[0].sequences[0].finish_reason == FinishReason.Length
    assert len(steps[-1].outputs) == 1


def test_multiple_requests_wait_queue():
    engine = SynchronousInferenceEngine(DummyModelModule(20))

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

    assert steps[1].outputs[0].sequences[0].finish_reason == FinishReason.Length

    assert len(steps[2].outputs) == 1
    assert steps[2].outputs[0].request_id == request_id_2

def test_multiple_requests_preempt():
    engine = SynchronousInferenceEngine(DummyModelModule(30))

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
    assert steps[6].outputs[0].is_finished


# Test to verify if evicted request from active batch which in intermediate
# state exceeding the max_num_batched_tokens can be processed successfully and will
# not hang the server in infinite attempt to return it back to the active loop
def test_cache_evict_hang():
    engine = SynchronousInferenceEngine(DummyModelModule(40, 10, 2))

    request_id_1 = "1"
    request_id_2 = "2"

    engine.add(
        [
            Request(
                request_id=request_id_1,
                messages=create_messages("A " * 10),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=20),
            ),
        ]
    )

    engine.add(
        [
            Request(
                request_id=request_id_2,
                messages=create_messages("A " * 10),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=20),
            ),
        ]
    )

    steps = [engine.step() for _ in range(40)]

    finished = {}
    empty_step = 0
    for s in steps:
        for o in s.outputs:
            if o.is_finished:
                finished[o.request_id] = o
        if not len(s.outputs):
            empty_step += 1

    assert len(finished) == 2
    assert finished['1'].sequences[0].num_generated_tokens == 20
    assert finished['2'].sequences[0].num_generated_tokens == 20
    assert empty_step <= 10


# Test to verify if new comming request with big prompt can be put into inference
# and does not have issues with cache size limits verification
def test_big_prompt_fit_to_cache():
    engine = SynchronousInferenceEngine(DummyModelModule(40, 30, 1))

    request_id_1 = "1"

    engine.add(
        [
            Request(
                request_id=request_id_1,
                messages=create_messages("A " * 30),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=5),
            ),
        ]
    )

    steps = [engine.step() for _ in range(40)]

    finished = {}
    return_token_step = 0
    for s in steps:
        for o in s.outputs:
            if o.is_finished:
                finished[o.request_id] = True
        if len(s.outputs):
            return_token_step += 1

    assert len(finished) == 1
    assert return_token_step >= 5


# Test to verify if new comming request with big prompt is handled properly
def test_big_prompt_not_fit_to_cache():
    engine = SynchronousInferenceEngine(DummyModelModule(29, 30, 1))

    request_id_1 = "1"

    engine.add(
        [
            Request(
                request_id=request_id_1,
                messages=create_messages("A " * 30),
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=2),
            ),
        ]
    )

    steps = [engine.step() for _ in range(5)]

    assert len(steps[0].outputs) == 1
    assert steps[0].outputs[0].is_finished
    # TODO(amalyshe): the behaviour of sync and staged engines are not consistent
    # Staging engine handles this situation better, it returns error and no sequences
    assert steps[0].outputs[0].sequences[0].finish_reason == FinishReason.Cancelled
    # TODO(amalyshe:)
    # There must be error, but currently error is lost in the engine, need to fix
    # assert steps[0].outputs[0].error
    assert len(steps[1].outputs) == 0