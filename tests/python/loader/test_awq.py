# pylint: disable=missing-docstring
from pathlib import Path
from typing import Union

import pytest
import tvm

from mlc_llm.loader import HuggingFaceLoader
from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support import logging, tqdm

logging.enable_logging()


@pytest.mark.parametrize(
    "param_path",
    [
        "./dist/models/llama-2-7b-w4-g128-awq.pt",
        "./dist/models/Llama-2-7B-AWQ/model.safetensors",
    ],
)
def test_load_llama(param_path: Union[str, Path]):
    path_params = Path(param_path)

    model = MODELS["llama"]
    quantization = QUANTIZATION["q4f16_awq"]
    config = model.config.from_dict(MODEL_PRESETS["llama2_7b"])
    loader = HuggingFaceLoader(
        path=path_params,
        extern_param_map=model.source["awq"](config, quantization),
    )
    with tqdm.redirect():
        for _name, _param in loader.load(tvm.device("cpu")):
            ...


if __name__ == "__main__":
    test_load_llama(param_path="./dist/models/llama-2-7b-w4-g128-awq.pt")
    test_load_llama(param_path="./dist/models/Llama-2-7B-AWQ/model.safetensors")
