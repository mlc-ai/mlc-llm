"""Tests for the model-package and compiled-program artifact contract."""

import json
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from mlc_llm.model import MODELS
from mlc_llm.model.gemma4.gemma4_config import Gemma4Config
from mlc_llm.protocol.artifact_manifest import (
    MODEL_PACKAGE_MANIFEST_FILENAME,
    CompiledProgramArtifact,
    ImageDecodeProcessor,
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


def _image_tasks():
    return {
        "chat.completions": {
            "executor": "generation",
            "inputs": {
                "text": {"processor": "tokenizer"},
                "image": {
                    "processor": {
                        "kind": "image_decode",
                        "format": "rgb_u8",
                        "layout": "nhwc",
                        "resize": {"mode": "center_crop", "height": 336, "width": 336},
                        "num_embeddings": 576,
                    },
                    "adapter": "image",
                    "prompt": {"placeholder_token_id": 32000},
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
                "prefill_tokens": "prefill_tokens",
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
    assert package.weights.manifest == "tensor-cache.json"
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


def test_image_processor_round_trips_through_the_contract():
    programs = _programs()
    programs["generation"]["adapters"] = {"image": "image_embed"}
    package = build_model_package_manifest(_image_tasks(), _params())
    compiled = build_compiled_program_artifact(_image_tasks(), programs, _params())
    assert package.schema_version == 1
    assert package.interface_id == compiled.interface_id
    assert package.interface_id != compute_interface_id(_tasks())

    processor = package.tasks["chat.completions"].inputs["image"].processor
    assert isinstance(processor, ImageDecodeProcessor)
    assert (processor.resize.height, processor.resize.width) == (336, 336)
    assert ModelPackageManifest.model_validate_json(package.model_dump_json(by_alias=True)) == (
        package
    )

    changed = _image_tasks()
    changed["chat.completions"]["inputs"]["image"]["processor"]["resize"]["mode"] = "stretch"
    assert compute_interface_id(changed) != package.interface_id


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("kind",), "video_decode", "does not match any of the expected tags"),
        (("format",), "rgba_u8", "rgb_u8"),
        (("layout",), "nchw", "nhwc"),
        (("num_embeddings",), 0, "greater than 0"),
        (("resize", "mode"), "dynamic_grid", "stretch"),
        (("resize", "height"), 0, "greater than 0"),
        (("sample_rate_hz",), 16000, "Extra inputs are not permitted"),
    ],
)
def test_contract_rejects_invalid_image_processor(path, value, match):
    tasks = _image_tasks()
    target = tasks["chat.completions"]["inputs"]["image"]["processor"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        build_model_package_manifest(tasks, _params())


def test_gemma4_interface_id_is_pinned():
    tasks = MODELS["gemma4"].artifact.tasks(Gemma4Config.from_dict({}))
    assert compute_interface_id(tasks) == (
        "sha256:6453d39d6c1a05b41e3d10ac1547892e2fde2ae228c6705b122ffde5e4c9c490"
    )
    manifest = build_model_package_manifest(tasks, _params())
    assert manifest.schema_version == 1
    assert manifest.model_dump(exclude_none=True, by_alias=True)["tasks"] == {
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
                        "min_samples": 161,
                        "max_samples": 480000,
                    },
                    "adapter": "audio",
                    "prompt": {
                        "prefix_token_ids": (256000,),
                        "placeholder_token_id": 258881,
                        "suffix_token_ids": (258883,),
                    },
                },
            },
            "output": "text",
        }
    }


def test_dump_model_package_manifest(tmp_path):
    manifest = build_model_package_manifest(_tasks(), _params())
    path = dump_model_package_manifest(manifest, tmp_path)
    assert path.name == MODEL_PACKAGE_MANIFEST_FILENAME
    assert json.loads(path.read_text())["schema"] == "mlc.model-package"
    assert "schema_" not in json.loads(path.read_text())
    assert ModelPackageManifest.model_validate_json(path.read_text()) == manifest


@dataclass
class _Dimension:
    name: str


def test_resource_sizes_resolve_named_dimensions():
    params = [
        ("embed", _Parameter((_Dimension("vocab_size"), 4), "float16")),
        ("bias", _Parameter((6,), "float32")),
    ]
    compiled = build_compiled_program_artifact(
        _tasks(), _programs(), params, symbolic_sizes={"vocab_size": 10}
    )
    assert compiled.resources.max_storage_buffer_binding_size == 10 * 4 * 2
    assert compiled.resources.estimated_device_memory_bytes == 10 * 4 * 2 + 6 * 4
    # The schema hash records the name, so it does not depend on the value.
    assert compiled.parameter_schema_id == compute_parameter_schema_id(params)

    with pytest.raises(ValueError, match="vocab_size"):
        build_compiled_program_artifact(_tasks(), _programs(), params)


def _exports(**roles):
    return {"embed_tokens": "embed", "create_kv_cache": "create_tir_paged_kv_cache", **roles}


@pytest.mark.parametrize(
    "roles",
    [
        {"prefill_tokens": "prefill_tokens", "decode_tokens": "decode_tokens"},
        {"prefill_embeds": "prefill", "decode_embeds": "decode"},
        {
            "prefill_tokens": "prefill_tokens",
            "decode_tokens": "decode_tokens",
            "prefill_embeds": "prefill",
            "decode_embeds": "decode",
        },
    ],
)
def test_token_generation_accepts_either_role_pair(roles):
    programs = {"generation": {"kind": "token_generation", "exports": _exports(**roles)}}
    tasks = _tasks()
    del tasks["chat.completions"]["inputs"]["audio"]
    compiled = build_compiled_program_artifact(tasks, programs, _params())
    assert compiled.programs["generation"].exports == _exports(**roles)


@pytest.mark.parametrize(
    "exports",
    [
        _exports(),
        _exports(prefill_tokens="prefill_tokens"),
        _exports(prefill_tokens="prefill_tokens", decode_embeds="decode"),
        _exports(
            prefill_tokens="prefill_tokens",
            decode_tokens="decode_tokens",
            prefill_embeds="prefill",
        ),
        {"prefill_embeds": "prefill", "decode_embeds": "decode", "embed_tokens": "embed"},
    ],
)
def test_token_generation_rejects_incomplete_role_pairs(exports):
    programs = {"generation": {"kind": "token_generation", "exports": exports}}
    with pytest.raises(ValidationError, match="token_generation requires"):
        build_compiled_program_artifact(_tasks(), programs, _params())
