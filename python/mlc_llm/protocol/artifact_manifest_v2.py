"""Strict descriptive component graphs; no runtime execution or v1 behavior changes."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

_SAFE = 2**53 - 1
_RESERVED = {"__proto__", "prototype", "constructor"}


def _not_reserved(value: str) -> str:
    if value in _RESERVED:
        raise ValueError("reserved identifier")
    return value


Identifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]*$"), AfterValidator(_not_reserved)
]
TaskName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.]*$"), AfterValidator(_not_reserved)
]
Symbol = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"), AfterValidator(_not_reserved)
]
Feature = Annotated[
    str, StringConstraints(pattern=r"^[a-z][a-z0-9-]*$"), AfterValidator(_not_reserved)
]
Semantic = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.]*\.v[1-9][0-9]*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Positive = Annotated[int, Field(gt=0, le=_SAFE)]
Nonnegative = Annotated[int, Field(ge=0, le=_SAFE)]
RelativePath = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$"),
]


def _plain(value: Any) -> Any:
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key.isascii() or key in _RESERVED:
                raise ValueError("object keys must be ASCII and not prototype keys")
            result[key] = _plain(item)
        return result
    if type(value) is list:
        return [_plain(item) for item in value]
    if type(value) is str:
        if not value.isascii():
            raise ValueError("contract strings must be ASCII")
    elif type(value) is float:
        if not math.isfinite(value) or not value.is_integer() or abs(value) > _SAFE:
            raise ValueError("expected a safe integer number")
        return int(value)
    elif value is not None and type(value) not in (bool, int):
        raise ValueError("expected plain JSON with safe integer numbers")
    elif type(value) is int and abs(value) > _SAFE:
        raise ValueError("integer exceeds JSON safe range")
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=False)

    @model_validator(mode="before")
    @classmethod
    def _plain_json(cls, value):
        return _plain(value)


class PortV2(_Contract):
    semantic: Semantic
    dtype: Literal["int32", "float16", "float32", "uint8"]
    shape: list[Positive | Identifier]
    delivery: Literal["value", "stream"]


class WeightsV2(_Contract):
    manifest: RelativePath
    parameter_schema_id: Sha256


class StateV2(_Contract):
    kind: Literal["none", "kv_cache", "recurrent", "hybrid"]
    scope: Literal["request"]


class ComponentV2(_Contract):
    kind: Literal["encoder", "autoregressive", "code_predictor", "waveform_decoder", "transform"]
    contract: Semantic
    inputs: dict[Identifier, PortV2]
    outputs: dict[Identifier, PortV2] = Field(min_length=1)
    weights: WeightsV2 | None
    state: StateV2


class EndpointV2(_Contract):
    component: Identifier
    port: Identifier


class TaskInputV2(_Contract):
    spec: PortV2
    targets: list[EndpointV2] = Field(min_length=1)


class TaskOutputV2(_Contract):
    spec: PortV2
    source: EndpointV2
    optional: bool


class EdgeV2(_Contract):
    source: EndpointV2
    target: EndpointV2


class TaskV2(_Contract):
    inputs: dict[Identifier, TaskInputV2] = Field(min_length=1)
    outputs: dict[Identifier, TaskOutputV2] = Field(min_length=1)
    edges: list[EdgeV2]


class ModelPackageV2(_Contract):
    schema_: Literal["mlc.model-package"] = Field(alias="schema")
    schema_version: Literal[2]
    interface_id: Sha256
    components: dict[Identifier, ComponentV2] = Field(min_length=1)
    tasks: dict[TaskName, TaskV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_graphs(self):
        for task in self.tasks.values():
            _validate_graph(self.components, task)
        return self


class ResourcesV2(_Contract):
    required_features: list[Feature]
    max_storage_buffer_binding_size: Nonnegative
    estimated_device_memory_bytes: Nonnegative

    @field_validator("required_features")
    @classmethod
    def _unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("required_features must be unique")
        return value


class CompiledComponentV2(_Contract):
    spec: ComponentV2
    module: RelativePath
    exports: dict[Identifier, Symbol] = Field(min_length=1)
    resources: ResourcesV2

    @model_validator(mode="after")
    def _exports(self):
        required = {"run"}
        if self.spec.state.kind != "none":
            required |= {"create_state", "reset_state"}
        if not required <= self.exports.keys():
            raise ValueError("missing required component exports")
        return self


class CompiledProgramV2(_Contract):
    schema_: Literal["mlc.compiled-program"] = Field(alias="schema")
    schema_version: Literal[2]
    interface_id: Sha256
    components: dict[Identifier, CompiledComponentV2] = Field(min_length=1)


def _endpoint(components, endpoint, direction):
    if endpoint.component not in components:
        raise ValueError("endpoint names an unknown component")
    ports = getattr(components[endpoint.component], direction)
    if endpoint.port not in ports:
        raise ValueError("endpoint names an unknown or wrong-direction port")
    return ports[endpoint.port]


def _topological(active, edges, roots=None):
    dependencies = {name: set() for name in active}
    for edge in edges:
        if edge.source.component in active and edge.target.component in active:
            dependencies[edge.target.component].add(edge.source.component)
    ordered, visiting, done = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("component graph contains a cycle")
        if name in done:
            return
        visiting.add(name)
        for dependency in sorted(dependencies[name]):
            visit(dependency)
        visiting.remove(name)
        done.add(name)
        ordered.append(name)

    for name in sorted(active) if roots is None else roots:
        visit(name)
    return ordered


def _closure(outputs, edges):
    needed = set(outputs)
    while True:
        more = {edge.source.component for edge in edges if edge.target.component in needed}
        if more <= needed:
            return needed
        needed |= more


def _validate_graph(components, task):
    active, bound = set(), set()

    def bind(endpoint, spec):
        actual = _endpoint(components, endpoint, "inputs")
        if actual != spec:
            raise ValueError("connected ports have incompatible specifications")
        key = (endpoint.component, endpoint.port)
        if key in bound:
            raise ValueError("component input is bound more than once")
        bound.add(key)
        active.add(endpoint.component)

    for item in task.inputs.values():
        for endpoint in item.targets:
            bind(endpoint, item.spec)
    for edge in task.edges:
        spec = _endpoint(components, edge.source, "outputs")
        active.add(edge.source.component)
        bind(edge.target, spec)
    for output in task.outputs.values():
        if _endpoint(components, output.source, "outputs") != output.spec:
            raise ValueError("public output specification does not match its source")
        active.add(output.source.component)
    if not any(not output.optional for output in task.outputs.values()):
        raise ValueError("task needs a required output")
    for name in active:
        if any((name, port) not in bound for port in components[name].inputs):
            raise ValueError("active component input is unbound")
    _topological(active, task.edges)
    useful = _closure({output.source.component for output in task.outputs.values()}, task.edges)
    if active != useful:
        raise ValueError("task contains a dead component branch")


def _decode(value):
    # Match JSON.parse: duplicate raw keys resolve to their final value, then the
    # resulting plain object undergoes strict contract validation.
    if isinstance(value, str):
        value = json.loads(value)
    return _plain(value)


def compute_interface_id_v2(value: Any) -> str:
    """Hash the complete explicit package contract, excluding only its identity field."""
    if isinstance(value, ModelPackageV2):
        value = value.model_dump(by_alias=True)
    payload = dict(_decode(value))
    payload["interface_id"] = "sha256:" + "0" * 64
    payload = ModelPackageV2.model_validate(payload).model_dump(by_alias=True)
    payload.pop("interface_id")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_model_package_v2(value: Any) -> ModelPackageV2:
    """Validate v2 structure and graph; verify_model_package_v2 also checks identity."""
    return ModelPackageV2.model_validate(_decode(value))


def verify_model_package_v2(value: Any) -> ModelPackageV2:
    """Validate structure and require the claimed canonical interface identity."""
    if isinstance(value, ModelPackageV2):
        value = value.model_dump(by_alias=True)
    package = parse_model_package_v2(value)
    if package.interface_id != compute_interface_id_v2(package):
        raise ValueError("interface_id does not match canonical package")
    return package


def parse_compiled_program_v2(value: Any) -> CompiledProgramV2:
    """Parse a v2 compiled descriptor; this does not load or execute its modules."""
    return CompiledProgramV2.model_validate(_decode(value))


def validate_compiled_v2(package: Any, compiled: Any) -> None:
    """Require exact package/compiled component agreement before any execution."""
    package = verify_model_package_v2(
        package.model_dump(by_alias=True) if isinstance(package, ModelPackageV2) else package
    )
    compiled = parse_compiled_program_v2(
        compiled.model_dump(by_alias=True) if isinstance(compiled, CompiledProgramV2) else compiled
    )
    if (
        package.interface_id != compiled.interface_id
        or package.components.keys() != compiled.components.keys()
    ):
        raise ValueError("compiled interface or component identities differ")
    for name, component in package.components.items():
        if component != compiled.components[name].spec:
            raise ValueError("compiled component specification differs")


def plan_task_outputs(package: Any, task_name: str, requested_outputs=None) -> dict[str, list[str]]:
    """Plan only dependencies needed by selected outputs; never execute components."""
    package = verify_model_package_v2(
        package.model_dump(by_alias=True) if isinstance(package, ModelPackageV2) else package
    )
    if task_name not in package.tasks:
        raise ValueError("unknown task")
    task = package.tasks[task_name]
    required = {name for name, output in task.outputs.items() if not output.optional}
    selected = sorted(required) if requested_outputs is None else requested_outputs
    if (
        type(selected) is not list
        or not selected
        or any(type(name) is not str for name in selected)
    ):
        raise ValueError("requested outputs must be a nonempty list of names")
    if (
        len(selected) != len(set(selected))
        or not required <= set(selected)
        or not set(selected) <= task.outputs.keys()
    ):
        raise ValueError("requested outputs must be unique, known, and include required outputs")
    needed = _closure({task.outputs[name].source.component for name in selected}, task.edges)
    inputs = sorted(
        name
        for name, item in task.inputs.items()
        if any(target.component in needed for target in item.targets)
    )
    roots = [task.outputs[name].source.component for name in sorted(task.outputs)]
    active = _closure(set(roots), task.edges)
    order = _topological(active, task.edges, roots)
    return {"components": [name for name in order if name in needed], "inputs": inputs}
