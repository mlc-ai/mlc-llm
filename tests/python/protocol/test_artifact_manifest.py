"""Tests for the model-package and compiled-program artifact contract."""

import json
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from mlc_llm.protocol.artifact_manifest import (
    MODEL_PACKAGE_MANIFEST_FILENAME,
    CompiledProgramArtifact,
    ModelPackageManifest,
    build_compiled_program_artifact,
    build_model_package_manifest,
    compute_interface_id,
    compute_parameter_schema_id,
    dump_model_package_manifest,
)


@dataclass
class _Parameter:
    shape: tuple
    dtype: str


def _tasks():
    return {
        "chat.completions": {
            "executor": "generation",
            "inputs": {
                "text": {"processor": "tokenizer"},
                "audio": {
                    "processor": {
                        "kind": "audio_decode",
                        "format": "pcm_f32",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "max_samples": 480000,
                    },
                    "adapter": "audio",
                    "prompt": {
                        "prefix_token_ids": [256000],
                        "placeholder_token_id": 258881,
                        "suffix_token_ids": [258883],
                    },
                },
            },
            "output": "text",
        }
    }


def _programs():
    return {
        "generation": {
            "kind": "token_generation",
            "exports": {
                "embed_tokens": "embed",
                "prefill_prompt": "prefill_prompt",
                "decode_tokens": "decode_tokens",
                "create_kv_cache": "create_tir_paged_kv_cache",
            },
            "adapters": {"audio": "audio_embed"},
        }
    }


def _params():
    return [
        ("b", _Parameter((5,), "uint32")),
        ("a", _Parameter((2, 3), "float16")),
    ]


def test_interface_hash_is_canonical_and_sensitive():
    tasks = _tasks()
    reordered = json.loads(json.dumps(tasks, sort_keys=True))
    assert compute_interface_id(tasks) == compute_interface_id(reordered)

    changed = _tasks()
    changed["chat.completions"]["inputs"]["audio"]["processor"]["sample_rate_hz"] = 8000
    assert compute_interface_id(tasks) != compute_interface_id(changed)


def test_parameter_schema_hash_is_sorted_and_sensitive():
    assert compute_parameter_schema_id(_params()) == compute_parameter_schema_id(
        reversed(_params())
    )
    changed = [("a", _Parameter((2, 4), "float16")), _params()[0]]
    assert compute_parameter_schema_id(_params()) != compute_parameter_schema_id(changed)


def test_package_and_compiled_contract_match():
    package = build_model_package_manifest(_tasks(), _params())
    compiled = build_compiled_program_artifact(
        _tasks(), _programs(), _params(), required_features=["shader-f16", "shader-f16"]
    )
    assert package.interface_id == compiled.interface_id
    assert package.weights.parameter_schema_id == compiled.parameter_schema_id
    assert compiled.resources.required_features == ("shader-f16",)
    assert compiled.resources.max_storage_buffer_binding_size == 20
    assert compiled.resources.estimated_device_memory_bytes == 32


def test_contract_forbids_unknown_fields_and_versions():
    package = build_model_package_manifest(_tasks(), _params()).model_dump(by_alias=True)
    package["unexpected"] = True
    with pytest.raises(ValidationError):
        ModelPackageManifest.model_validate(package)

    compiled = build_compiled_program_artifact(_tasks(), _programs(), _params()).model_dump(
        by_alias=True
    )
    compiled["schema_version"] = 2
    with pytest.raises(ValidationError):
        CompiledProgramArtifact.model_validate(compiled)


def test_compiled_contract_rejects_missing_executor_or_adapter():
    programs = _programs()
    del programs["generation"]["adapters"]["audio"]
    with pytest.raises(ValueError, match="missing adapter"):
        build_compiled_program_artifact(_tasks(), programs, _params())

    tasks = _tasks()
    tasks["chat.completions"]["executor"] = "missing"
    with pytest.raises(ValueError, match="missing executor"):
        build_compiled_program_artifact(tasks, _programs(), _params())


def test_contract_rejects_invalid_audio_bounds_and_token_ids():
    tasks = _tasks()
    tasks["chat.completions"]["inputs"]["audio"]["processor"]["min_samples"] = 9
    tasks["chat.completions"]["inputs"]["audio"]["processor"]["max_samples"] = 8
    with pytest.raises(ValidationError, match="min_samples"):
        build_model_package_manifest(tasks, _params())

    tasks = _tasks()
    tasks["chat.completions"]["inputs"]["audio"]["prompt"]["placeholder_token_id"] = -1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        build_model_package_manifest(tasks, _params())


def test_dump_model_package_manifest(tmp_path):
    manifest = build_model_package_manifest(_tasks(), _params())
    path = dump_model_package_manifest(manifest, tmp_path)
    assert path.name == MODEL_PACKAGE_MANIFEST_FILENAME
    assert json.loads(path.read_text())["schema"] == "mlc.model-package"
    assert "schema_" not in json.loads(path.read_text())
    assert ModelPackageManifest.model_validate_json(path.read_text()) == manifest
