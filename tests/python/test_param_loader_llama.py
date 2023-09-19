# pylint: disable=missing-docstring
import logging
from pathlib import Path

from mlc_llm.models.llama import LlamaConfig
from mlc_llm.models.llama_param_map import hf_torch
from mlc_llm.param_loader import HFTorchLoader

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)


def test_load_7b():
    prefix = Path("./dist/models/llama-2-7b-chat-hf/")
    path_config = prefix / "config.json"
    path_params = prefix / "pytorch_model.bin.index.json"

    model_config = LlamaConfig.from_file(path_config)
    with HFTorchLoader(
        config_path=path_params,
        param_map=hf_torch(model_config),
    ) as loader:
        for name in loader.suggest_loading_order():
            loader.load_param(name=name)


if __name__ == "__main__":
    test_load_7b()
