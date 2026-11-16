"""Speaker export preserves the pinned voice IDs and rejects invalid conditioning."""

import runpy
from pathlib import Path

import pytest


def validator():
    path = Path(__file__).resolve().parents[3] / "examples/python/omni_audio/export_speakers.py"
    return runpy.run_path(str(path))["validate_speakers"]


@pytest.fixture
def voices():
    torch = pytest.importorskip("torch")
    return {
        name: dict(bos_token=bos, cond=torch.zeros(1, 192), ref_mel=torch.zeros(1, 400, 80))
        for name, bos in (("Ethan", 151870), ("Chelsie", 151872))
    }


def test_voice_ids_and_exact_values(voices):
    voices["Ethan"]["cond"][0, 0] = 0.125
    result = validator()(voices)
    assert result["Ethan"]["cond"][0][0] == 0.125
    assert result["Ethan"]["bos_token"] == 151870
    assert result["Chelsie"]["bos_token"] == 151872


@pytest.mark.parametrize("invalid", ["missing", "bos", "shape", "dtype", "nan"])
def test_reject_invalid_voice_assets(voices, invalid):
    if invalid == "missing":
        del voices["Chelsie"]
    elif invalid == "bos":
        voices["Ethan"]["bos_token"] = 151872
    elif invalid == "shape":
        voices["Ethan"]["cond"] = voices["Ethan"]["cond"][0]
    elif invalid == "dtype":
        voices["Ethan"]["cond"] = voices["Ethan"]["cond"].half()
    else:
        voices["Ethan"]["cond"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        validator()(voices)
