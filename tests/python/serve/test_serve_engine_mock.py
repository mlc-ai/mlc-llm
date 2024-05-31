"""Mock testing engine I/O conventions

Mock test only can help checking the overall input
output processing options are passed correctly
"""

import pytest
import tvm

from mlc_llm.serve import MLCEngine
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
    response = engine.chat.completions.create(  # type: ignore
        messages=[{"role": "user", "content": "hello"}],
        **param_dict,
    )
    # echo mock will echo back the generation config
    for k, v in param_dict.items():
        assert response.usage.extra[k] == v


if __name__ == "__main__":
    test_completion_api()
