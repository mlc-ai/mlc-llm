# pylint: disable=invalid-name,missing-docstring
import pytest
from mlc_chat.compiler.model.llama import LlamaConfig, LlamaForCasualLM


@pytest.mark.parametrize("model_name", ["llama2_7b", "llama2_13b", "llama2_70b"])
def test_llama2_creation(model_name: str):
    config = LlamaConfig.from_predefined(model_name)
    model = LlamaForCasualLM(config)
    mod, named_params = model.export_tvm(spec=model.get_default_spec())
    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


if __name__ == "__main__":
    test_llama2_creation("llama2_7b")
    test_llama2_creation("llama2_13b")
    test_llama2_creation("llama2_70b")
