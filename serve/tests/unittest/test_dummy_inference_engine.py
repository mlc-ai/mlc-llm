from mlc_serve.engine import ChatMessage, Request, SamplingParams, StoppingCriteria
from mlc_serve.engine.dummy import DummyInferenceEngine


def test_single_request():
    engine = DummyInferenceEngine()

    request_id = "1"
    engine.add(
        [
            Request(
                request_id=request_id,
                messages=[ChatMessage(role="user", content="test prompt")],
                sampling_params=SamplingParams(temperature=1),
                stopping_criteria=StoppingCriteria(max_tokens=20),
            ),
        ]
    )

    result = engine.step()

    assert result.outputs[0].request_id == request_id
    assert result.outputs[0].error is None
    assert len(result.outputs) == 1
