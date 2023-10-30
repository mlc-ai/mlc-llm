from mlc_chat.compiler import MODELS
from mlc_chat.compiler import QUANT
from mlc_chat.compiler.model.llama_quantization import llama_group_quantization


def test_llama_group_quantization(model_name: str, quant_name: str):
    model_info = MODELS["llama"]
    model_config = model_info.config.from_predefined(model_name)
    quant_config = QUANT[quant_name]
    model = model_info.model(model_config)
    model, quant_map = llama_group_quantization(model, quant_config)
    print(quant_map.param_map)


if __name__ == "__main__":
    test_llama_group_quantization("llama2_7b", "q4f16_1")
