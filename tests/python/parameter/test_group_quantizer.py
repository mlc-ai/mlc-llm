# pylint: disable=missing-docstring
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler import MODELS
from mlc_chat.compiler.model.llama_config import LlamaConfig
from mlc_chat.compiler.model.llama_parameter import hf_torch_group_quantize
from mlc_chat.compiler.parameter import GroupQuantizer, HuggingFaceLoader
from mlc_chat.support import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def test_load_torch_llama_group_quantize(base_path: Union[str, Path]):
    base_path = Path(base_path)
    path_config = base_path / "config.json"
    path_params = base_path / "pytorch_model.bin.index.json"

    model = MODELS["llama"]
    config = LlamaConfig.from_file(path_config)
    loader = HuggingFaceLoader(
        path=path_params,
        extern_param_map=model.source["huggingface-torch"](config, None),
        quantizer=GroupQuantizer(hf_torch_group_quantize(config, "q4f16_1")),
    )
    with tqdm.redirect():
        for _name, _param in loader.load():
            ...


if __name__ == "__main__":
    test_load_torch_llama_group_quantize(base_path="./dist/models/Llama-2-7b-hf")
