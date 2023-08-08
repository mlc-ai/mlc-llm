"""Load MLC LLM library and _ffi_api functions."""

import ctypes
import os
import sys

import tvm
import tvm._ffi.base

from . import libinfo


def _load_mlc_llm_lib():
    """Load mlc llm lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib_name = "mlc_llm" if tvm._ffi.base._RUNTIME_ONLY else "mlc_llm_module"
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


# only load once here
if os.environ.get("SKIP_LOADING_MLCLLM_SO", "0") == "0":
    _LIB, _LIB_PATH = _load_mlc_llm_lib()


def get_delta_message(curr_message: str, new_message: str) -> str:
    r"""Given the current message and the new message, compute the delta message
    (the newly generated part, the diff of the new message from the current message).

    Parameters
    ----------
    curr_message : str
        The message generated in the previous round.
    new_message : str
        The message generated in the new round.

    Returns
    -------
    delta_message : str
        The diff of the new message from the current message (the newly generated part).
    """
    f_get_delta_message = tvm.get_global_func("mlc.get_delta_message")
    return f_get_delta_message(curr_message, new_message)
