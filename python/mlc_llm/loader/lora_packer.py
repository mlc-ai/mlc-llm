"""Utility to convert a PEFT LoRA adapter into a runtime-friendly artifact.

The runtime path will eventually *mmap* the produced file and upload the delta
weights to GPU/CPU memory via C++ FFI.  Until that path is ready, this helper
only guarantees a stable on-disk format so the rest of the pipeline can depend
on it.

The chosen format is NumPy ``.npz`` – human-readable, portable, and can be
memory-mapped.  Each entry is saved under the key pattern::

    delta.<layer_name>  ->  (out_features, in_features)  float32 / float16

The function accepts either a *directory* produced by HuggingFace PEFT (which
contains ``adapter_model.bin`` or ``adapter_model.safetensors``) **or** a path
to that file directly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Union

import numpy as np

# Torch is an optional dependency for the core mlc-llm package but required for
# the conversion tooling.  Import lazily so most users are unaffected.
try:
    import torch
except ImportError as exc:  # pragma: no cover – CI installs torch
    raise RuntimeError(
        "The LoRA packer requires PyTorch. Install with `pip install torch`."
    ) from exc

# Safetensors is optional – fall back to torch.load if missing.
try:
    from safetensors import safe_open  # type: ignore

    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover – plenty of setups lack safetensors
    _HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Helper – read delta tensors from PEFT checkpoint
# ---------------------------------------------------------------------------


def _read_peft_adapter(file_path: Path) -> Dict[str, np.ndarray]:
    """Return a dict *name → ndarray* with LoRA delta tensors.

    The PEFT format uses keys like ``base_layer.lora_A.weight`` and
    ``base_layer.lora_B.weight``.  We combine them into a single delta matrix
    ``B @ A`` so the runtime can apply the fused formulation.
    """

    # 1. Load state-dict
    if file_path.suffix in {".bin", ".pt", ".pth"}:
        state_dict: Dict[str, torch.Tensor] = torch.load(  # type: ignore[arg-type]
            file_path, map_location="cpu"
        )
    elif file_path.suffix == ".safetensors" and _HAS_SAFETENSORS:
        state_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)  # type: ignore[assignment]
    else:  # pragma: no cover
        raise ValueError(f"Unsupported adapter file format: {file_path}")

    # 2. Group A & B pairs
    a_tensors: Dict[str, torch.Tensor] = {}
    b_tensors: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".lora_A.weight"):
            layer = key.removesuffix(".lora_A.weight")
            a_tensors[layer] = value
        elif key.endswith(".lora_B.weight"):
            layer = key.removesuffix(".lora_B.weight")
            b_tensors[layer] = value

    # 3. Compose delta = B @ A for each layer.
    deltas: Dict[str, np.ndarray] = {}
    for layer, a in a_tensors.items():
        if layer not in b_tensors:  # pragma: no cover – malformed ckpt
            raise ValueError(f"Missing lora_B for layer {layer}")
        b = b_tensors[layer]
        delta = torch.matmul(b, a)
        deltas[layer] = delta.cpu().numpy()

    return deltas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_lora_adapter(adapter_path: Union[str, Path], out_file: Union[str, Path]) -> Path:
    """Convert *adapter_path* into a ``.npz`` file stored at *out_file*.

    Parameters
    ----------
    adapter_path : str or Path
        Directory produced by PEFT **or** a direct path to the adapter file.
    out_file : str or Path
        Where to write the ``.npz`` file.  Parent directories will be created.

    Returns
    -------
    Path
        Absolute path to the written file.
    """

    adapter_path = Path(adapter_path).expanduser().resolve()
    out_file = Path(out_file).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine the actual ckpt file.
    if adapter_path.is_dir():
        # Prefer safetensors if both exist.
        for candidate in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"):
            ckpt = adapter_path / candidate
            if ckpt.exists():
                break
        else:  # pragma: no cover – directory without ckpt
            raise FileNotFoundError("No adapter checkpoint found in directory: " f"{adapter_path}")
    else:
        ckpt = adapter_path

    deltas = _read_peft_adapter(ckpt)

    # Save npz – enforce deterministic key order for reproducibility.
    savez_kwargs: Dict[str, np.ndarray] = {
        f"delta.{k}": v.astype(np.float16) for k, v in sorted(deltas.items())
    }
    np.savez(out_file, **savez_kwargs)  # type: ignore[arg-type]

    # Write manifest JSON for easy introspection (alpha defaults to 1.0, can be
    # overridden later by metadata in package).
    manifest = {
        "format": "mlc-lora-delta-v1",
        "layers": list(sorted(deltas.keys())),
        "dtype": "float16",
    }
    with out_file.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also copy over the original adapter config if present (for debugging).
    src_cfg = ckpt.with_name("adapter_config.json")
    if src_cfg.exists():
        shutil.copy(src_cfg, out_file.with_name("adapter_config.json"))

    return out_file
