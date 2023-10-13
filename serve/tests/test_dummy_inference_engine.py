from mlc_serve.engine import Request, SamplingParams, StoppingCriteria
from mlc_serve.engine.dummy import DummyInferenceEngine


def test_single_request():
    engine = DummyInferenceEngine()

    request_id = engine.add(
        [
            Request(
                request_id="1",
                prompt="test prompt",
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=20),
            ),
        ]
    )[0]

    result = engine.step()

    assert result.outputs[0].request_id == request_id
    assert len(result.outputs) == 1
    assert not result.errors
