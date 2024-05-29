# pylint: disable=missing-docstring
import json
import os
import tempfile
from pathlib import Path

import pytest

from mlc_llm.support import logging
from mlc_llm.support.auto_weight import detect_weight

logging.enable_logging()

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def _create_json_file(json_path, data):
    with open(json_path, "w", encoding="utf-8") as i_f:
        json.dump(data, i_f)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"),
        ("auto", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("auto", "model.safetensors.index.json", "huggingface-safetensor"),
    ],
)
def test_detect_weight(weight_format, index_filename, result):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        if index_filename is not None:
            weight_index_file = base_path / index_filename
            _create_json_file(weight_index_file, {})
        assert detect_weight(base_path, None, weight_format) == (weight_index_file, result)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"),
        ("auto", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("auto", "model.safetensors.index.json", "huggingface-safetensor"),
    ],
)
def test_detect_weight_in_config_json(weight_format, index_filename, result):
    with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as weight_dir:
        config_path = Path(config_dir)
        weight_path = Path(weight_dir)
        config_json_path = config_path / "config.json"
        _create_json_file(config_json_path, {"weight_path": weight_dir})
        if index_filename is not None:
            weight_index_file = weight_path / index_filename
            _create_json_file(weight_index_file, {})

        assert detect_weight(None, config_json_path, weight_format) == (weight_index_file, result)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"),
        ("auto", "pytorch_model.bin.index.json", "huggingface-torch"),
        ("auto", "model.safetensors.index.json", "huggingface-safetensor"),
    ],
)
def test_detect_weight_same_dir_config_json(weight_format, index_filename, result):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        config_json_path = base_path / "config.json"
        _create_json_file(config_json_path, {})
        if index_filename is not None:
            weight_index_file = Path(os.path.join(tmpdir, index_filename))
            _create_json_file(weight_index_file, {})
        assert detect_weight(None, config_json_path, weight_format) == (weight_index_file, result)


def test_find_weight_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        with pytest.raises(ValueError):
            detect_weight(Path("do/not/exist"), base_path, "awq")
    with pytest.raises(AssertionError):
        detect_weight(None, Path("do/not/exist"), "awq")


if __name__ == "__main__":
    test_detect_weight("huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch")
    test_detect_weight(
        "huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"
    )
    test_detect_weight("auto", "pytorch_model.bin.index.json", "huggingface-torch")
    test_detect_weight("auto", "model.safetensors.index.json", "huggingface-safetensor")
    test_detect_weight_in_config_json(
        "huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch"
    )
    test_detect_weight_in_config_json(
        "huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"
    )
    test_detect_weight_in_config_json("auto", "pytorch_model.bin.index.json", "huggingface-torch")
    test_detect_weight_in_config_json(
        "auto", "model.safetensors.index.json", "huggingface-safetensor"
    )
    test_detect_weight_same_dir_config_json(
        "huggingface-torch", "pytorch_model.bin.index.json", "huggingface-torch"
    )
    test_detect_weight_same_dir_config_json(
        "huggingface-safetensor", "model.safetensors.index.json", "huggingface-safetensor"
    )
    test_detect_weight_same_dir_config_json(
        "auto", "pytorch_model.bin.index.json", "huggingface-torch"
    )
    test_detect_weight_same_dir_config_json(
        "auto", "model.safetensors.index.json", "huggingface-safetensor"
    )
    test_find_weight_fail()
