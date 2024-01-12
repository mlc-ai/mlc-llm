from typing import Optional
import pytest

from mlc_serve.engine import (
    ChatMessage,
    FinishReason,
    Request,
    RequestId,
    RequestOutput,
    SamplingParams,
    StoppingCriteria,
)
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
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 30
        }
        )
    engine.start()

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
    # execute everything to free server and allow it to destroy
    for i in range(0,100):
        engine.step()
    engine.stop()

    assert step.outputs[0].request_id == request_id
    assert step.outputs[0].error is None
    assert len(step.outputs) == 1


def test_single_request_step_to_finish():
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 30
        }
        )
    engine.start()

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
    # execute everything to free server and allow it to destroy
    for i in range(0,100):
        engine.step()
    engine.stop()

    assert steps[-1].outputs[0].request_id == request_id
    assert steps[-1].outputs[0].sequences[0].finish_reason == FinishReason.Length
    assert len(steps[-1].outputs) == 1


def test_multiple_requests_wait_queue():
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 20
        }
        )
    engine.start()

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
    # execute everything to free server and allow it to destroy
    for i in range(0,100):
        engine.step()
    engine.stop()

    assert len(steps[0].outputs) == 1
    assert steps[0].outputs[0].request_id == request_id_1

    assert len(steps[1].outputs) == 1
    assert steps[1].outputs[0].request_id == request_id_1

    assert steps[1].outputs[0].sequences[0].finish_reason == FinishReason.Length

    assert len(steps[2].outputs) == 1
    assert steps[2].outputs[0].request_id == request_id_2


def test_multiple_requests_preempt():
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 30
        }
        )
    engine.start()

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

    steps = [engine.step() for _ in range(20)]
    # execute everything to free server and allow it to destroy
    for i in range(0,100):
        engine.step()
    engine.stop()

    # Due to asynchronious nature of request submission and processing, we cannot
    # track exactly on each step certain request is processed
    # but we can catch a pattern that it is two requests processed simultaneously,
    # then 1 and it is finished and then the latest and it should be  finished as well
    stage = 0

    for s in steps:
        if stage == 0 and len(s.outputs) == 2:
            stage = 1
        elif stage == 1 and len(s.outputs) == 1:
            stage = 2
            assert s.outputs[0].is_finished
        elif stage == 2 and len(s.outputs) == 1:
            stage = 3
            assert s.outputs[0].is_finished
        elif stage == 3 and len(s.outputs) > 0:
            stage = 4

    assert stage == 3


# Test to verify if evicted request from active batch which in intermediate
# state exceeding the max_num_batched_tokens can be processed successfully and will
# not hang the server in infinite attempt to return it back to the active loop
@pytest.mark.xfail
def test_cache_evict_hang_staging():
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 40,
            "max_input_len": 10,
            "max_num_sequences": 2
        }
        )
    engine.start()

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
    engine.stop()

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
def test_big_prompt_fit_to_cache_staging():
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 40,
            "max_input_len": 30,
            "max_num_sequences": 1
        }
        )
    engine.start()

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
    engine.stop()

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
    engine = StagingInferenceEngine(
        tokenizer_module=DummyTokenizerModule(),
        model_module_loader=DummyModelModule,
        model_module_loader_kwargs = {
            "max_cached_tokens": 29,
            "max_input_len": 30,
            "max_num_sequences": 1
        }
        )
    engine.start()

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
    engine.stop()

    assert len(steps[0].outputs) == 1
    assert steps[0].outputs[0].is_finished
    assert steps[0].outputs[0].error
    assert len(steps[1].outputs) == 0
