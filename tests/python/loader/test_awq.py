# pylint: disable=missing-docstring
import logging
from pathlib import Path
from typing import Union

import pytest
from mlc_chat.compiler import MODELS, QUANTIZATION
from mlc_chat.compiler.parameter import HuggingFaceLoader
from mlc_chat.support import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


@pytest.mark.parametrize(
    "param_path",
    [
        "./dist/models/llama-2-7b-w4-g128-awq.pt",
    ],
)
def test_load_llama(param_path: Union[str, Path]):
    path_params = Path(param_path)

    model = MODELS["llama"]
    quantization = QUANTIZATION["q4f16_awq"]
    config = model.config.from_predefined("llama2_7b")
    loader = HuggingFaceLoader(
        path=path_params,
        extern_param_map=model.source["awq"](config, quantization),
    )
    with tqdm.redirect():
        for _name, _param in loader.load():
            ...


if __name__ == "__main__":
    test_load_llama(param_path="./dist/models/llama-2-7b-w4-g128-awq.pt")
