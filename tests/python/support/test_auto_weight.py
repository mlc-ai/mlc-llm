# pylint: disable=missing-docstring
import json
import logging
import os
import tempfile
from pathlib import Path

import pytest
from mlc_chat.support.auto_weight import detect_weight

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)


def _create_json_file(json_path, data):
    with open(json_path, "w", encoding="utf-8") as i_f:
        json.dump(data, i_f)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("PyTorch", "pytorch_model.bin.index.json", "PyTorch"),
        ("SafeTensor", "model.safetensors.index.json", "SafeTensor"),
        ("GGML", None, "GGML"),
        ("GGUF", None, "GGUF"),
        ("AWQ", None, "AWQ"),
        ("auto", "pytorch_model.bin.index.json", "PyTorch"),
        ("auto", "model.safetensors.index.json", "SafeTensor"),
    ],
)
def test_detect_weight(weight_format, index_filename, result):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        if index_filename is not None:
            weight_index_file = base_path / index_filename
            _create_json_file(weight_index_file, {})
        assert detect_weight(base_path, None, weight_format) == (base_path, result)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("PyTorch", "pytorch_model.bin.index.json", "PyTorch"),
        ("SafeTensor", "model.safetensors.index.json", "SafeTensor"),
        ("GGML", None, "GGML"),
        ("GGUF", None, "GGUF"),
        ("AWQ", None, "AWQ"),
        ("auto", "pytorch_model.bin.index.json", "PyTorch"),
        ("auto", "model.safetensors.index.json", "SafeTensor"),
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

        assert detect_weight(None, config_json_path, weight_format) == (weight_path, result)


@pytest.mark.parametrize(
    "weight_format, index_filename, result",
    [
        ("PyTorch", "pytorch_model.bin.index.json", "PyTorch"),
        ("SafeTensor", "model.safetensors.index.json", "SafeTensor"),
        ("GGML", None, "GGML"),
        ("GGUF", None, "GGUF"),
        ("AWQ", None, "AWQ"),
        ("auto", "pytorch_model.bin.index.json", "PyTorch"),
        ("auto", "model.safetensors.index.json", "SafeTensor"),
    ],
)
def test_detect_weight_same_dir_config_json(weight_format, index_filename, result):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        config_json_path = base_path / "config.json"
        _create_json_file(config_json_path, {})
        if index_filename is not None:
            weight_index_file = os.path.join(tmpdir, index_filename)
            _create_json_file(weight_index_file, {})
        assert detect_weight(None, config_json_path, weight_format) == (base_path, result)


def test_find_weight_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        with pytest.raises(ValueError):
            detect_weight(Path("do/not/exist"), base_path, "AWQ")
    with pytest.raises(AssertionError):
        detect_weight(None, Path("do/not/exist"), "AWQ")


if __name__ == "__main__":
    test_detect_weight("PyTorch", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight("SafeTensor", "model.safetensors.index.json", "SafeTensor")
    test_detect_weight("GGML", None, "GGML")
    test_detect_weight("GGUF", None, "GGUF")
    test_detect_weight("AWQ", None, "AWQ")
    test_detect_weight("auto", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight("auto", "model.safetensors.index.json", "SafeTensor")
    test_detect_weight_in_config_json("PyTorch", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight_in_config_json("SafeTensor", "model.safetensors.index.json", "SafeTensor")
    test_detect_weight_in_config_json("GGML", None, "GGML")
    test_detect_weight_in_config_json("GGUF", None, "GGUF")
    test_detect_weight_in_config_json("AWQ", None, "AWQ")
    test_detect_weight_in_config_json("auto", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight_in_config_json("auto", "model.safetensors.index.json", "SafeTensor")
    test_detect_weight_same_dir_config_json("PyTorch", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight_same_dir_config_json(
        "SafeTensor", "model.safetensors.index.json", "SafeTensor"
    )
    test_detect_weight_same_dir_config_json("GGML", None, "GGML")
    test_detect_weight_same_dir_config_json("GGUF", None, "GGUF")
    test_detect_weight_same_dir_config_json("AWQ", None, "AWQ")
    test_detect_weight_same_dir_config_json("auto", "pytorch_model.bin.index.json", "PyTorch")
    test_detect_weight_same_dir_config_json("auto", "model.safetensors.index.json", "SafeTensor")
    test_find_weight_fail()
