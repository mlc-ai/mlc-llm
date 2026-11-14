import argparse as stdlib_argparse
from pathlib import Path

import pytest

from mlc_llm.cli import gen_config as gen_config_cli
from mlc_llm.support.argparse import boolean

pytestmark = [pytest.mark.unittest]


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on", " true "])
def test_boolean_accepts_true_spellings(value):
    assert boolean(value) is True


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "off", " false "])
def test_boolean_accepts_false_spellings(value):
    assert boolean(value) is False


@pytest.mark.parametrize("value", ["", "maybe", "2"])
def test_boolean_rejects_other_values(value):
    with pytest.raises(stdlib_argparse.ArgumentTypeError):
        boolean(value)


@pytest.mark.parametrize(
    "flag_value, expected",
    [("false", False), ("true", True), ("0", False), ("1", True)],
)
def test_gen_config_cli_forwards_disaggregation(monkeypatch, tmp_path, flag_value, expected):
    """`--disaggregation false` must reach gen_config() as False.

    `ModelConfigOverride.apply` skips a `None` override, so `False` is a distinct value
    that force-disables disaggregation. With `type=bool` it was unreachable, because
    argparse converts the raw string and `bool("false")` is `True`.
    """
    captured = {}

    monkeypatch.setattr(gen_config_cli, "detect_config", Path)
    monkeypatch.setattr(gen_config_cli, "detect_model_type", lambda _type, _config: "dummy")
    monkeypatch.setattr(gen_config_cli, "MODELS", {"dummy": object()})
    monkeypatch.setattr(gen_config_cli, "QUANTIZATION", {"q0f16": object()})
    monkeypatch.setattr(gen_config_cli, "CONV_TEMPLATES", ["llama-3"])
    monkeypatch.setattr(gen_config_cli, "gen_config", captured.update)

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    gen_config_cli.main(
        [
            str(config_path),
            "--quantization",
            "q0f16",
            "--conv-template",
            "llama-3",
            "--disaggregation",
            flag_value,
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert captured["disaggregation"] is expected
