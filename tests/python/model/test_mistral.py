# pylint: disable=invalid-name,missing-docstring
import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS


@pytest.mark.parametrize("model_name", ["mistral_7b"])
def test_mistral_creation(model_name: str):
    model_info = MODELS["mistral"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_mistral_creation("mistral_7b")
