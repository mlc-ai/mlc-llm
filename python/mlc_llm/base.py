"""Load MLC LLM library and _ffi_api functions."""

import ctypes
import os
import sys

import tvm
import tvm.base

from . import libinfo

SKIP_LOADING_MLCLLM_SO = os.environ.get("SKIP_LOADING_MLCLLM_SO", "0")


def _load_mlc_llm_lib():
    """Load MLC LLM lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    # pylint: disable=protected-access
    lib_name = "mlc_llm" if tvm.base._RUNTIME_ONLY else "mlc_llm_module"
    # pylint: enable=protected-access
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


@tvm.register_func("mlc.debug_cuda_profiler_start")
def _debug_cuda_profiler_start() -> None:
    """Start cuda profiler."""
    import cuda  # pylint: disable=import-outside-toplevel
    import cuda.cudart  # pylint: disable=import-outside-toplevel

    cuda.cudart.cudaProfilerStart()  # pylint: disable=c-extension-no-member


@tvm.register_func("mlc.debug_cuda_profiler_stop")
def _debug_cuda_profiler_stop() -> None:
    """Stop cuda profiler."""
    import cuda  # pylint: disable=import-outside-toplevel
    import cuda.cudart  # pylint: disable=import-outside-toplevel

    cuda.cudart.cudaProfilerStop()  # pylint: disable=c-extension-no-member


# only load once here
if SKIP_LOADING_MLCLLM_SO == "0":
    _LIB, _LIB_PATH = _load_mlc_llm_lib()
