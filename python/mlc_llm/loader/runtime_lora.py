"""Loader mapping for PEFT weights used by runtime LoRA branches."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Dict, Iterable  # noqa: UP035

from tvm.relax.frontend import nn

from .mapping import ExternMapping


def _cast_runtime_lora_weight(tensor, *, dtype, scale: float):
    if scale != 1.0:
        tensor = tensor * scale
    return tensor.astype(dtype)


def resolve_runtime_lora_weight(adapter_dir: Path) -> Path:
    """Resolve the safetensors file (or index) in a PEFT adapter directory."""
    for filename in (
        "adapter_model.safetensors.index.json",
        "adapter_model.safetensors",
    ):
        candidate = adapter_dir / filename
        if candidate.is_file():
            return candidate
    if (adapter_dir / "adapter_model.bin").is_file():
        raise ValueError(
            "Runtime LoRA currently requires safetensors adapter weights; "
            "`adapter_model.bin` is not supported."
        )
    raise ValueError(f"Cannot find PEFT safetensors weights in: {adapter_dir}")


def _read_safetensor_keys(path: Path) -> Iterable[str]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as index_file:
            return tuple(json.load(index_file)["weight_map"].keys())

    with path.open("rb") as weight_file:
        header_length_bytes = weight_file.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_length = int.from_bytes(header_length_bytes, byteorder="little", signed=False)
        header = json.loads(weight_file.read(header_length).decode("utf-8"))
    return tuple(name for name in header if name != "__metadata__")


def _normalize_peft_name(name: str) -> str:
    return re.sub(r"\.(lora_[AB])\.[^.]+\.weight$", r".\1.weight", name)


def make_runtime_lora_mapping(
    named_parameters: Dict[str, nn.Parameter],  # noqa: UP006
    adapter_weight: Path,
) -> ExternMapping:
    """Map exported runtime-LoRA parameters to their PEFT tensor names."""
    adapter_keys = tuple(_read_safetensor_keys(adapter_weight))
    normalized_keys = {name: _normalize_peft_name(name) for name in adapter_keys}
    mapping = ExternMapping()

    for mlc_name, mlc_param in named_parameters.items():
        source_suffix = mlc_param.attrs.get("peft_source_suffix")
        if source_suffix is None:
            continue
        normalized_suffix = _normalize_peft_name(source_suffix)
        candidates = [
            name
            for name, normalized in normalized_keys.items()
            if normalized.endswith(normalized_suffix)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one PEFT tensor ending in `{source_suffix}` for `{mlc_name}`, "
                f"but found {len(candidates)}."
            )
        mapping.add_mapping(
            mlc_name,
            candidates,
            functools.partial(
                _cast_runtime_lora_weight,
                dtype=mlc_param.dtype,
                scale=float(mlc_param.attrs.get("runtime_lora_scale", 1.0)),
            ),
        )

    if not mapping.param_map:
        raise ValueError("The exported model does not contain runtime LoRA parameters.")
    return mapping


__all__: tuple[str, ...] = ("make_runtime_lora_mapping", "resolve_runtime_lora_weight")
