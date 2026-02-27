# pylint: disable=missing-docstring,protected-access
import contextlib
import json
import tempfile
from pathlib import Path

import pytest

from mlc_llm.interface import convert_weight as convert_weight_interface

pytestmark = [pytest.mark.unittest]


def test_resolve_base_model_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        model_dir = temp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        source_file = model_dir / "pytorch_model.bin.index.json"
        source_file.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")

        assert convert_weight_interface._resolve_base_model_dir(model_dir) == model_dir
        assert convert_weight_interface._resolve_base_model_dir(source_file) == model_dir


def test_convert_weight_with_lora_uses_merged_source(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        config_path = temp_path / "config.json"
        config_path.write_text(json.dumps({}), encoding="utf-8")

        source_dir = temp_path / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / "pytorch_model.bin.index.json"
        source_file.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")

        adapter_dir = temp_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        merged_dir = temp_path / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_file = merged_dir / "pytorch_model.bin"
        merged_file.write_bytes(b"")

        captured = {}

        @contextlib.contextmanager
        def _fake_merge(base_source: Path, lora_adapter: Path):
            captured["merge_base_source"] = base_source
            captured["merge_lora_adapter"] = lora_adapter
            yield merged_dir

        def _fake_detect_weight(weight_path: Path, config_json_path: Path, weight_format: str):
            captured["detect_weight_path"] = weight_path
            captured["detect_weight_config"] = config_json_path
            captured["detect_weight_format"] = weight_format
            return merged_file, "huggingface-torch"

        def _fake_convert_args(args):
            captured["converted_args"] = args

        monkeypatch.setattr(
            convert_weight_interface, "_merge_lora_adapter_with_base_model", _fake_merge
        )
        monkeypatch.setattr(convert_weight_interface, "detect_weight", _fake_detect_weight)
        monkeypatch.setattr(convert_weight_interface, "_convert_args", _fake_convert_args)
        monkeypatch.setattr(convert_weight_interface.ConversionArgs, "display", lambda self: None)

        convert_weight_interface.convert_weight(
            config=config_path,
            quantization=object(),
            model=type("DummyModel", (), {"name": "dummy"})(),
            device=object(),
            source=source_file,
            source_format="huggingface-safetensor",
            output=temp_path / "output",
            lora_adapter=adapter_dir,
        )

        converted_args = captured["converted_args"]
        assert captured["merge_base_source"] == source_file
        assert captured["merge_lora_adapter"] == adapter_dir
        assert captured["detect_weight_path"] == merged_dir
        assert captured["detect_weight_config"] == config_path
        assert captured["detect_weight_format"] == "auto"
        assert converted_args.source == merged_file
        assert converted_args.source_format == "huggingface-torch"
        assert converted_args.lora_adapter == adapter_dir


def test_convert_weight_with_lora_rejects_awq():
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        config_path = temp_path / "config.json"
        config_path.write_text(json.dumps({}), encoding="utf-8")
        adapter_dir = temp_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="only supports source formats"):
            convert_weight_interface.convert_weight(
                config=config_path,
                quantization=object(),
                model=type("DummyModel", (), {"name": "dummy"})(),
                device=object(),
                source=temp_path / "source",
                source_format="awq",
                output=temp_path / "output",
                lora_adapter=adapter_dir,
            )
