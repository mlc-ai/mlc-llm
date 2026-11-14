"""Shared cross-language contracts and malformed graph rejection."""

import copy
import json
from pathlib import Path

import pytest

from mlc_llm.protocol.artifact_manifest_v2 import (
    compute_interface_id_v2,
    parse_compiled_program_v2,
    parse_model_package_v2,
    plan_task_outputs,
    validate_compiled_v2,
    verify_model_package_v2,
)

FIXTURES = Path(__file__).parent / "fixtures/artifact_manifest_v2"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def rehash(package):
    package["interface_id"] = compute_interface_id_v2(package)
    return package


def test_shared_hash_and_output_closure():
    package, compiled, expected = fixture("model"), fixture("compiled"), fixture("expected")
    parsed = parse_model_package_v2(json.dumps(package))
    assert compute_interface_id_v2(parsed) == expected["interface_id"]
    assert (
        compute_interface_id_v2(dict(reversed(list(package.items())))) == expected["interface_id"]
    )
    validate_compiled_v2(parsed, parse_compiled_program_v2(compiled))
    assert plan_task_outputs(parsed, "chat.completions") == {
        "components": expected["text_components"],
        "inputs": expected["inputs"],
    }
    assert plan_task_outputs(package, "chat.completions", ["text", "audio"]) == {
        "components": expected["all_components"],
        "inputs": expected["inputs"],
    }
    for outputs in ([], ["audio"], ["text", "text"], ["text", "missing"]):
        with pytest.raises(ValueError):
            plan_task_outputs(package, "chat.completions", outputs)


@pytest.mark.parametrize("bad", [True, 1.5, 0, -1, 2**53, "constructor", "bad\n"])
def test_strict_dimensions(bad):
    package = fixture("model")
    package["components"]["audio_encoder"]["inputs"]["audio"]["shape"] = [bad]
    with pytest.raises(ValueError):
        parse_model_package_v2(package)


@pytest.mark.parametrize(
    "path", ["../model.wasm", "/model.wasm", "a//b", "https://host/a", "a%20b", "a\\b"]
)
def test_compiled_path_rejection(path):
    compiled = fixture("compiled")
    compiled["components"]["thinker"]["module"] = path
    with pytest.raises(ValueError):
        parse_compiled_program_v2(compiled)


def test_unknown_prototype_duplicate_fields_and_identity():
    for key in ("unknown", "constructor", "__proto__", "prototype"):
        bad = fixture("model")
        bad[key] = {}
        with pytest.raises(ValueError):
            parse_model_package_v2(bad)
    bad = fixture("model")
    bad["interface_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="interface_id"):
        verify_model_package_v2(bad)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unbound", "cycle", "direction", "dead", "optional", "incompatible"]
)
def test_invalid_graphs(mutation):
    package = fixture("model")
    task = package["tasks"]["chat.completions"]
    if mutation == "duplicate":
        task["edges"].append(copy.deepcopy(task["edges"][0]))
    elif mutation == "unbound":
        task["edges"].pop(0)
    elif mutation == "cycle":
        del task["inputs"]["audio"]
        # Give the back edge exactly matching ports, so rejection must detect the cycle.
        port = package["components"]["audio_encoder"]["inputs"]["audio"]
        package["components"]["thinker"]["outputs"]["back"] = copy.deepcopy(port)
        task["edges"].append(
            dict(
                source=dict(component="thinker", port="back"),
                target=dict(component="audio_encoder", port="audio"),
            )
        )
    elif mutation == "direction":
        task["edges"][0]["source"]["port"] = "audio"
    elif mutation == "dead":
        del task["outputs"]["audio"]
    elif mutation == "optional":
        task["outputs"]["text"]["optional"] = True
    else:
        task["outputs"]["text"]["spec"]["semantic"] = "different.v1"
    with pytest.raises(ValueError):
        parse_model_package_v2(rehash(package))


def test_compiled_disagreement_and_state_exports():
    package, compiled = fixture("model"), fixture("compiled")
    compiled["components"]["thinker"]["spec"]["contract"] = "different.v1"
    with pytest.raises(ValueError, match="specification"):
        validate_compiled_v2(package, compiled)
    compiled = fixture("compiled")
    del compiled["components"]["thinker"]["exports"]["reset_state"]
    with pytest.raises(ValueError, match="exports"):
        parse_compiled_program_v2(compiled)
    compiled = fixture("compiled")
    compiled["components"]["thinker"]["resources"]["required_features"] = ["shader-f16"] * 2
    with pytest.raises(ValueError, match="unique"):
        parse_compiled_program_v2(compiled)


def test_json_integral_number_spellings():
    package = fixture("model")
    encoded = json.dumps(package)
    for spelling in ("2.0", "2e0"):
        parsed = parse_model_package_v2(
            encoded.replace('"schema_version": 2', '"schema_version": ' + spelling)
        )
        assert parsed.interface_id == package["interface_id"]


def test_authoring_without_identity_and_forged_pair_rejection():
    package = fixture("model")
    claimed = package.pop("interface_id")
    assert compute_interface_id_v2(package) == claimed
    package["interface_id"] = "sha256:" + "0" * 64
    # Structural parsing remains useful while authoring; consumers verify identity.
    parse_model_package_v2(package)
    compiled = fixture("compiled")
    compiled["interface_id"] = package["interface_id"]
    with pytest.raises(ValueError, match="interface_id"):
        validate_compiled_v2(package, compiled)
    with pytest.raises(ValueError, match="interface_id"):
        plan_task_outputs(package, "chat.completions")


def test_identity_covers_component_weights_and_state():
    package = fixture("model")
    original = package["interface_id"]
    package["components"]["audio_encoder"]["weights"]["manifest"] = "other/tensor-cache.json"
    assert compute_interface_id_v2(package) != original
    package = fixture("model")
    package["components"]["audio_encoder"]["state"]["kind"] = "recurrent"
    assert compute_interface_id_v2(package) != original


def test_raw_json_matches_json_parse_and_plan_key_order():
    package = fixture("model")
    encoded = json.dumps(package).replace(
        '"schema_version": 2', '"schema_version": 1, "schema_version": 2'
    )
    assert verify_model_package_v2(encoded).interface_id == package["interface_id"]
    expected = plan_task_outputs(package, "chat.completions", ["text", "audio"])
    task = package["tasks"]["chat.completions"]
    task["outputs"] = dict(reversed(list(task["outputs"].items())))
    assert plan_task_outputs(package, "chat.completions", ["audio", "text"]) == expected
