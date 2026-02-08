"""LoRA (Low-Rank Adaptation) module with proper library loading."""

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Union

import tvm
from tvm.runtime import Device

# Global variables for registered LoRA directories
_registered_lora_dirs: List[str] = []


def _ensure_library_loaded():
    """Ensure the MLC-LLM library is loaded so TVM FFI functions are available."""
    try:
        # Find the compiled library
        possible_paths = [
            "/content/mlc-llm/build/libmlc_llm_module.so",
            "/content/mlc-llm/build/libmlc_llm.so",
            "./build/libmlc_llm_module.so",
            "./build/libmlc_llm.so",
        ]

        for lib_path in possible_paths:
            if os.path.exists(lib_path):
                print(f"Loading MLC-LLM library: {lib_path}")
                # Load the library to register TVM FFI functions
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                print("✓ MLC-LLM library loaded successfully")
                return True

        print("✗ No MLC-LLM library found")
        return False

    except Exception as e:
        print(f"✗ Failed to load MLC-LLM library: {e}")
        return False


def _resolve_funcs():
    """Resolve TVM FFI functions for LoRA operations."""
    # Ensure library is loaded first
    _ensure_library_loaded()

    # Try to get the functions
    upload_func = tvm.get_global_func("mlc.serve.UploadLora", allow_missing=True)
    get_delta_func = tvm.get_global_func("mlc.get_lora_delta", allow_missing=True)
    set_device_func = tvm.get_global_func("mlc.set_active_device", allow_missing=True)

    if upload_func is None:
        raise RuntimeError("UploadLora FFI symbol not found in TVM runtime.")
    if get_delta_func is None:
        raise RuntimeError("get_lora_delta FFI symbol not found in TVM runtime.")
    if set_device_func is None:
        raise RuntimeError("set_active_device FFI symbol not found in TVM runtime.")

    return upload_func, get_delta_func, set_device_func


def upload_lora(
    adapter_path: Union[str, Path], device: Optional[Device] = None, alpha: float = 1.0
) -> None:
    """Upload a LoRA adapter for use in inference.

    Args:
        adapter_path: Path to the LoRA adapter (.npz file)
        device: Target device for LoRA operations
        alpha: Scaling factor for LoRA deltas
    """
    if device is None:
        device = tvm.cpu(0)

    print(f"Uploading LoRA adapter: {adapter_path}")
    print(f"Device: {device}, Alpha: {alpha}")

    # Resolve FFI functions
    upload_func, _, set_device_func = _resolve_funcs()

    # Set the active device
    set_device_func(device.device_type, device.device_id)

    # Upload the adapter
    upload_func(str(adapter_path))

    print("✓ LoRA adapter uploaded successfully")


def get_lora_delta(param_name: str):
    """Get LoRA delta tensor for a parameter.

    Args:
        param_name: Name of the parameter to get delta for

    Returns:
        TVM NDArray containing the LoRA delta
    """
    _, get_delta_func, _ = _resolve_funcs()
    return get_delta_func(param_name)


def set_lora(adapter_path: Union[str, Path], device: Optional[Device] = None):
    """Set active LoRA adapter (alias for upload_lora)."""
    upload_lora(adapter_path, device)


def get_registered_lora_dirs() -> List[str]:
    """Get list of registered LoRA directories."""
    return _registered_lora_dirs.copy()


def register_lora_dir(directory: Union[str, Path]) -> None:
    """Register a directory containing LoRA adapters."""
    dir_str = str(directory)
    if dir_str not in _registered_lora_dirs:
        _registered_lora_dirs.append(dir_str)
        print(f"✓ Registered LoRA directory: {dir_str}")


def clear_lora_registrations() -> None:
    """Clear all registered LoRA directories."""
    global _registered_lora_dirs
    count = len(_registered_lora_dirs)
    _registered_lora_dirs.clear()
    print(f"✓ Cleared {count} LoRA registrations")
