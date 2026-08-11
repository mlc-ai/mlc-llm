"""Versioned contract between a model package and its compiled program."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import (  # noqa: UP035
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tvm.runtime import DataType

MODEL_PACKAGE_MANIFEST_FILENAME = "mlc-model-manifest.json"
MODEL_PACKAGE_SCHEMA = "mlc.model-package"
COMPILED_PROGRAM_SCHEMA = "mlc.compiled-program"
ARTIFACT_SCHEMA_VERSION = 1


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class PromptInsertion(_ContractModel):
    """Token sequence which reserves one contiguous adapter output span."""

    prefix_token_ids: tuple[int, ...] = ()
    placeholder_token_id: int = Field(ge=0)
    suffix_token_ids: tuple[int, ...] = ()

    @field_validator("prefix_token_ids", "suffix_token_ids")
    @classmethod
    def _validate_token_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(token_id < 0 for token_id in value):
            raise ValueError("token IDs must be non-negative")
        return value


class AudioDecodeProcessor(_ContractModel):
    """Canonical PCM representation accepted by a compiled audio adapter."""

    kind: Literal["audio_decode"]
    format: Literal["pcm_f32"]
    sample_rate_hz: int = Field(gt=0)
    channels: Literal[1]
    min_samples: int = Field(default=1, gt=0)
    max_samples: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_sample_range(self):
        if self.min_samples > self.max_samples:
            raise ValueError("min_samples must not exceed max_samples")
        return self


class TaskInput(_ContractModel):
    """One named input role in a task."""

    processor: Union[str, AudioDecodeProcessor]  # noqa: UP007
    adapter: str | None = None
    prompt: PromptInsertion | None = None

    @field_validator("processor")
    @classmethod
    def _validate_processor(cls, value: Any) -> Any:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, AudioDecodeProcessor):
            return value
        raise ValueError("processor must be a non-empty name or a supported processor object")

    @model_validator(mode="after")
    def _validate_adapter_prompt(self):
        if (self.adapter is None) != (self.prompt is None):
            raise ValueError("adapter and prompt must be declared together")
        return self


class TaskSpec(_ContractModel):
    """Public task roles and their canonical representations."""

    executor: str
    inputs: Dict[str, TaskInput] = Field(min_length=1)  # noqa: UP006
    output: str

    @field_validator("executor", "output")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class WeightContract(_ContractModel):
    manifest: Literal["ndarray-cache.json"]
    parameter_schema_id: str

    @field_validator("parameter_schema_id")
    @classmethod
    def _validate_parameter_schema_id(cls, value: str) -> str:
        return _validate_sha256(value)


class ModelPackageManifest(_ContractModel):
    schema_: Literal["mlc.model-package"] = Field(
        default=MODEL_PACKAGE_SCHEMA,
        alias="schema",
    )
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    chat_config: Literal["mlc-chat-config.json"] = "mlc-chat-config.json"
    interface_id: str
    weights: WeightContract
    tasks: Dict[str, TaskSpec] = Field(min_length=1)  # noqa: UP006

    @field_validator("schema_version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported model package schema version: {value}")
        return value

    @field_validator("interface_id")
    @classmethod
    def _validate_interface_id(cls, value: str) -> str:
        return _validate_sha256(value)


class ProgramSpec(_ContractModel):
    kind: str
    exports: Dict[str, str] = Field(min_length=1)  # noqa: UP006
    adapters: Dict[str, str] = Field(default_factory=dict)  # noqa: UP006

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        if not value:
            raise ValueError("kind must not be empty")
        return value

    @field_validator("exports", "adapters")
    @classmethod
    def _validate_entrypoints(cls, value: Dict[str, str]) -> Dict[str, str]:  # noqa: UP006
        if any(not name or not entrypoint for name, entrypoint in value.items()):
            raise ValueError("entrypoint names and symbols must not be empty")
        return value


class ResourceRequirements(_ContractModel):
    required_features: tuple[str, ...] = ()
    max_storage_buffer_binding_size: int = Field(ge=0)
    estimated_device_memory_bytes: int = Field(ge=0)


class CompiledProgramArtifact(_ContractModel):
    schema_: Literal["mlc.compiled-program"] = Field(
        default=COMPILED_PROGRAM_SCHEMA,
        alias="schema",
    )
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    interface_id: str
    parameter_schema_id: str
    programs: Dict[str, ProgramSpec]  # noqa: UP006
    resources: ResourceRequirements

    @field_validator("schema_version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported compiled program schema version: {value}")
        return value

    @field_validator("interface_id", "parameter_schema_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _validate_sha256(value)


@dataclasses.dataclass(frozen=True)
class ArtifactDefinition:
    """Architecture-owned factories for public tasks and compiled programs."""

    tasks: Callable[[Any], Mapping[str, Any]]
    programs: Callable[[Any], Mapping[str, Any]]
    required_features: tuple[str, ...] = ()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str) -> str:
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("expected a lowercase sha256:<64 hex digits> identifier")
    return value


def normalize_tasks(tasks: Mapping[str, Any]) -> Dict[str, TaskSpec]:  # noqa: UP006
    """Parse task definitions and return a deterministically ordered mapping."""
    if not tasks:
        raise ValueError("at least one task must be declared")
    return {name: TaskSpec.model_validate(tasks[name]) for name in sorted(tasks)}


def normalize_programs(programs: Mapping[str, Any]) -> Dict[str, ProgramSpec]:  # noqa: UP006
    """Parse program definitions and return a deterministically ordered mapping."""
    if not programs:
        raise ValueError("at least one compiled program must be declared")
    return {name: ProgramSpec.model_validate(programs[name]) for name in sorted(programs)}


def compute_interface_id(tasks: Mapping[str, Any]) -> str:
    """Hash only the public task roles and canonical representations."""
    normalized = normalize_tasks(tasks)
    payload = {name: spec.model_dump(exclude_none=True) for name, spec in normalized.items()}
    return _sha256_json({"tasks": payload})


def parameter_specs(named_parameters: Iterable[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Return sorted post-quantization parameter name/shape/dtype records."""

    def _dimension(value: Any) -> Any:
        if isinstance(value, int):
            return value
        if hasattr(value, "value") and isinstance(value.value, int):
            return value.value
        if hasattr(value, "name"):
            return value.name
        return str(value)

    result = [
        {
            "name": name,
            "shape": [_dimension(dim) for dim in parameter.shape],
            "dtype": str(parameter.dtype),
        }
        for name, parameter in named_parameters
    ]
    names = [item["name"] for item in result]
    if len(names) != len(set(names)):
        raise ValueError("parameter names must be unique")
    return sorted(result, key=lambda item: item["name"])


def compute_parameter_schema_id(named_parameters: Iterable[tuple[str, Any]]) -> str:
    """Hash the post-quantization parameter schema independently of iteration order."""
    return _sha256_json(parameter_specs(named_parameters))


def _parameter_resources(named_parameters: Iterable[tuple[str, Any]]) -> tuple[int, int]:
    sizes = []
    for spec in parameter_specs(named_parameters):
        if not all(isinstance(dim, int) for dim in spec["shape"]):
            raise ValueError(f"resource size requires static parameter shape: {spec['name']}")
        elements = 1
        for dim in spec["shape"]:
            elements *= dim
        sizes.append(elements * DataType(spec["dtype"]).itemsize)
    return (max(sizes, default=0), sum(sizes))


def build_model_package_manifest(
    tasks: Mapping[str, Any],
    named_parameters: Iterable[tuple[str, Any]],
) -> ModelPackageManifest:
    """Build the model-package half of the contract."""
    named_parameters = list(named_parameters)
    return ModelPackageManifest(
        interface_id=compute_interface_id(tasks),
        weights=WeightContract(
            manifest="ndarray-cache.json",
            parameter_schema_id=compute_parameter_schema_id(named_parameters),
        ),
        tasks=normalize_tasks(tasks),
    )


def build_compiled_program_artifact(
    tasks: Mapping[str, Any],
    programs: Mapping[str, Any],
    named_parameters: Iterable[tuple[str, Any]],
    required_features: Iterable[str] = (),
) -> CompiledProgramArtifact:
    """Build metadata embedded in the compiled VM library."""
    named_parameters = list(named_parameters)
    normalized_tasks = normalize_tasks(tasks)
    normalized_programs = normalize_programs(programs)
    for task_name, task in normalized_tasks.items():
        if task.executor not in normalized_programs:
            raise ValueError(f"Task {task_name!r} references missing executor {task.executor!r}")
        program = normalized_programs[task.executor]
        for input_name, task_input in task.inputs.items():
            if task_input.adapter is not None and task_input.adapter not in program.adapters:
                raise ValueError(
                    f"Task {task_name!r} input {input_name!r} references missing adapter "
                    f"{task_input.adapter!r}"
                )
    max_buffer_size, total_size = _parameter_resources(named_parameters)
    return CompiledProgramArtifact(
        interface_id=compute_interface_id(normalized_tasks),
        parameter_schema_id=compute_parameter_schema_id(named_parameters),
        programs=normalized_programs,
        resources=ResourceRequirements(
            required_features=tuple(sorted(set(required_features))),
            max_storage_buffer_binding_size=max_buffer_size,
            estimated_device_memory_bytes=total_size,
        ),
    )


def dump_model_package_manifest(manifest: ModelPackageManifest, output: Path) -> Path:
    """Write the canonical sidecar JSON and return its path."""
    path = output / MODEL_PACKAGE_MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest.model_dump(exclude_none=True, by_alias=True), file, indent=2)
        file.write("\n")
    return path
