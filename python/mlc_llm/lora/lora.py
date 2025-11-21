"""LoRA runtime/compile-time manager (Python side).

This file provides a single public helper ``set_lora`` used by the compile
and runtime entry-points to inform the rest of the python stack where LoRA
adapters live on the file-system.

For the first iteration the function only records the paths and exposes
them through a getter so that:

1. The compile pipeline can embed the information in the metadata of the
   generated package (``enable_lora=true`` and the list of adapters).
2. The server/runtime can later pick the information up and upload the
   adapter(s) via FFI.

The heavy-lifting (segment-gemm kernels, C++ LoraManager, etc.) will be
added later – this just lays the plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import tvm


# ---------------------------------------------------------------------------
# _GLOBAL_REGISTRY – simple process-wide storage
# ---------------------------------------------------------------------------

_LORA_DIRS: List[Path] = []
_UPLOAD_FUNC = None  # cached global func
_SET_DEVICE_FUNC = None  # cached global func
_INITIALISED_DEVICE = False
_LOADED_ADAPTERS: set[str] = set()


# Public exports for this module – will be extended below.
__all__: list[str] = [
    "set_lora",
    "get_registered_lora_dirs",
]


def set_lora(lora_dirs: Optional[List[Path]] = None) -> None:  # noqa: D401 – not property
    """Register LoRA adapter directories for the current process.

    Parameters
    ----------
    lora_dirs : list[Path] or None
        Paths that contain LoRA adapters (each directory must contain a
        ``lora_manifest.json``).  If *None* or empty, LoRA support is
        considered disabled.
    """

    global _LORA_DIRS  # noqa: WPS420 – deliberate global state

    if lora_dirs is None:
        _LORA_DIRS = []
    else:
        _LORA_DIRS = [Path(p).expanduser().resolve() for p in lora_dirs]


def get_registered_lora_dirs() -> List[Path]:
    """Return the list of LoRA adapters currently registered."""

    return _LORA_DIRS.copy()


def _resolve_funcs() -> None:
    """Resolve and cache the required TVM PackedFuncs."""

    global _UPLOAD_FUNC, _SET_DEVICE_FUNC  # noqa: WPS420

    if _UPLOAD_FUNC is None:
        _UPLOAD_FUNC = tvm.get_global_func("mlc.serve.UploadLora", allow_missing=True)
        if _UPLOAD_FUNC is None:  # pragma: no cover
            raise RuntimeError("UploadLora FFI symbol not found in TVM runtime.")

    if _SET_DEVICE_FUNC is None:
        _SET_DEVICE_FUNC = tvm.get_global_func("mlc.set_active_device", allow_missing=True)
        if _SET_DEVICE_FUNC is None:  # pragma: no cover
            raise RuntimeError("set_active_device FFI symbol not found in TVM runtime.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_lora(adapter_path: Path | str, *, device=None) -> None:  # type: ignore[override]
    """Load a LoRA adapter (.npz) at runtime and push to the active device.

    Parameters
    ----------
    adapter_path : str or Path
        Path to the ``.npz`` file containing LoRA delta tensors.
    device : tvm.runtime.Device, optional
        Target device for the tensors.  If *None*, we default to CPU(0).
    """

    from tvm import runtime as _rt  # local import to avoid circular deps

    _resolve_funcs()

    path = str(Path(adapter_path).expanduser().resolve())
    if path in _LOADED_ADAPTERS:
        return  # already loaded in this process

    global _INITIALISED_DEVICE  # noqa: WPS420
    if not _INITIALISED_DEVICE:
        if device is None:
            device = _rt.cpu(0)
        _SET_DEVICE_FUNC(int(device.device_type), int(device.device_id))
        _INITIALISED_DEVICE = True

    _UPLOAD_FUNC(path)
    _LOADED_ADAPTERS.add(path)


__all__.append("upload_lora")
