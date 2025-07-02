import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlc_llm.lora import lora as lora_module
from mlc_llm.serve.engine import MLCEngine


@pytest.fixture(name="dummy_pkg")
def _dummy_pkg(tmp_path: Path):
    """Create a minimal compiled package structure with LoRA metadata."""

    # create ndarray-cache stub
    (tmp_path / "params").mkdir()
    (tmp_path / "ndarray-cache.json").write_text("{}")

    # LoRA adapter file
    adapter_rel = Path("adapters/adapter0.npz")
    (tmp_path / adapter_rel.parent).mkdir()
    (tmp_path / adapter_rel).write_bytes(b"FAKE")

    # metadata
    meta = {
        "LoRASeparate": True,
        "LoRAPaths": [str(adapter_rel)],
        "LoRAAlpha": 1.0,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))

    return tmp_path


def test_engine_uploads_separate_lora(monkeypatch, dummy_pkg):
    called = []

    def _fake_upload(path):
        called.append(Path(path))

    monkeypatch.setattr(lora_module, "upload_lora", _fake_upload)

    # minimal engine_config stub with required attribute
    engine_cfg = SimpleNamespace(lora_dirs=[])

    # Instantiate engine (CPU target implied by default)
    engine = MLCEngine(model=str(dummy_pkg), mode="local", engine_config=engine_cfg)

    expected_path = dummy_pkg / "adapters/adapter0.npz"
    assert called == [expected_path] 