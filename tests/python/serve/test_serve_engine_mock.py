"""Mock testing engine I/O conventions

Mock test only can help checking the overall input
output processing options are passed correctly
"""

from types import SimpleNamespace

import pytest
import tvm

from mlc_llm.serve import MLCEngine
from mlc_llm.serve.engine_base import EngineState
from mlc_llm.testing import require_test_model

# test category "unittest"
pytestmark = [pytest.mark.unittest]


# NOTE: we only need tokenizers in folder
# launch time of mock test is fast so we can put it in unittest
@require_test_model("Llama-3-8B-Instruct-q4f16_1-MLC")
def test_completion_api(model: str):
    engine = MLCEngine(model, tvm.cpu(), model_lib="mock://echo")
    param_dict = {
        "top_p": 0.6,
        "temperature": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "n": 2,
    }
    response = engine.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
        **param_dict,
    )
    # echo mock will echo back the generation config
    for k, v in param_dict.items():
        assert response.usage.extra[k] == v


def test_generate_ignores_output_for_another_request(monkeypatch):
    monkeypatch.setattr("mlc_llm.serve.engine.TextStreamer", lambda _: object())
    monkeypatch.setattr(
        "mlc_llm.serve.engine.engine_utils.convert_prompts_to_data", lambda prompt: prompt
    )

    engine = MLCEngine.__new__(MLCEngine)
    engine.state = EngineState(enable_tracing=False)
    engine.tokenizer = object()
    engine._terminated = False
    stale = SimpleNamespace(
        unpack=lambda: (
            "cancelled",
            [SimpleNamespace(request_final_usage_json_str='{"request":"cancelled"}')],
        )
    )
    fresh = SimpleNamespace(
        unpack=lambda: (
            "fresh",
            [SimpleNamespace(request_final_usage_json_str='{"request":"fresh"}')],
        )
    )
    engine._ffi = {
        "create_request": lambda *_: object(),
        "add_request": lambda _: (
            engine.state._sync_request_stream_callback([stale]),
            engine.state._sync_request_stream_callback([fresh]),
        ),
    }
    engine.abort = lambda _: None
    config = SimpleNamespace(n=1, model_dump_json=lambda **_: "{}")

    outputs = list(engine._generate([0], config, request_id="fresh"))
    engine._terminated = True

    assert len(outputs) == 1
    assert outputs[0][0].request_final_usage_json_str == '{"request":"fresh"}'


if __name__ == "__main__":
    test_completion_api()
