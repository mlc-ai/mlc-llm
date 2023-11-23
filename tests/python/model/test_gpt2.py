# pylint: disable=invalid-name,missing-docstring
import pytest

from mlc_chat.compiler import MODEL_PRESETS, MODELS


@pytest.mark.parametrize("model_name", ["gpt2"])
def test_gpt2_creation(model_name: str):
    model_info = MODELS["gpt2"]
    config = model_info.config.from_dict(MODEL_PRESETS[model_name])
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_gpt2_creation("gpt2")
