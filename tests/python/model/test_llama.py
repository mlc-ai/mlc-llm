# pylint: disable=invalid-name,missing-docstring
import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS


@pytest.mark.parametrize(
    "model_name", ["llama2_7b", "llama2_13b", "llama2_70b", "tinyllama_1b_chat_v1.0"]
)
def test_llama2_creation(model_name: str):
    model_info = MODELS["llama"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_llama2_creation("llama2_7b")
    test_llama2_creation("llama2_13b")
    test_llama2_creation("llama2_70b")
    test_llama2_creation("tinyllama_1b_chat_v1")
