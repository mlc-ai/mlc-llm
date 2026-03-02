# pylint: disable=missing-docstring
import json
import tempfile
from pathlib import Path

import pytest

from mlc_llm.cli import convert_weight as convert_weight_cli

pytestmark = [pytest.mark.unittest]


def test_convert_weight_cli_passes_lora_adapter(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        config_path = temp_path / "config.json"
        source_dir = temp_path / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_index = source_dir / "pytorch_model.bin.index.json"
        adapter_dir = temp_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_path / "output"

        config_path.write_text(json.dumps({}), encoding="utf-8")
        source_index.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")

        def _fake_detect_device(device):
            return device

        def _fake_detect_weight(_weight_path, _config_json_path, _weight_format):
            return source_index, "huggingface-torch"

        def _fake_detect_model_type(_model_type, _config):
            return "dummy"

        monkeypatch.setattr(convert_weight_cli, "detect_config", Path)
        monkeypatch.setattr(convert_weight_cli, "detect_device", _fake_detect_device)
        monkeypatch.setattr(convert_weight_cli, "detect_weight", _fake_detect_weight)
        monkeypatch.setattr(convert_weight_cli, "detect_model_type", _fake_detect_model_type)
        monkeypatch.setattr(convert_weight_cli, "MODELS", {"dummy": object()})
        monkeypatch.setattr(convert_weight_cli, "QUANTIZATION", {"q0f16": object()})

        call_args = {}

        def _fake_convert_weight(**kwargs):
            call_args.update(kwargs)

        monkeypatch.setattr(convert_weight_cli, "convert_weight", _fake_convert_weight)

        convert_weight_cli.main(
            [
                str(config_path),
                "--quantization",
                "q0f16",
                "--model-type",
                "dummy",
                "--source",
                str(source_dir),
                "--source-format",
                "auto",
                "--output",
                str(output_dir),
                "--lora-adapter",
                str(adapter_dir),
            ]
        )

        assert call_args["lora_adapter"] == adapter_dir
        assert call_args["source"] == source_index
        assert call_args["source_format"] == "huggingface-torch"
