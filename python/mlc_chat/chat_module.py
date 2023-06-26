"""Python runtime for MLC chat."""
#! pylint: disable=unused-import
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


_LIB, _LIB_PATH = _load_mlc_llm_lib()


def quantization_keys():
    return [
        "q3f16_0",
        "q4f16_0",
        "q4f32_0",
        "q8f16_0",
        "q8f32_0",
        "q0f16",
        "q0f32",
    ]


class ChatModule:
    def __init__(self, target: str = "cuda", device_id: int = 0):
        r"""Initialize a chat module.

        Parameters
        ----------
        target : str
            The target device type.
        device_id : int
            The device id.
        """
        fcreate = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate is not None
        if target == "cuda":
            self.device = tvm.cuda(device_id)
        elif target == "metal":
            self.device = tvm.metal(device_id)
        elif target == "vulkan":
            self.device = tvm.vulkan(device_id)
        else:
            raise ValueError("device type not supported yet")
        device_type = self.device.device_type
        chat_mod = fcreate(device_type, device_id)

        self.reload_func = chat_mod["reload"]
        self.prefill_func = chat_mod["prefill"]
        self.decode_func = chat_mod["decode"]
        self.stopped_func = chat_mod["stopped"]
        self.get_message_func = chat_mod["get_message"]
        self.reset_chat_func = chat_mod["reset_chat"]
        self.runtime_stats_text_func = chat_mod["runtime_stats_text"]
        self.reset_runtime_stats_func = chat_mod["reset_runtime_stats"]
        self.evaluate_func = chat_mod["evaluate"]
        self.get_role0 = chat_mod["get_role0"]
        self.get_role1 = chat_mod["get_role1"]

    def reload(self, lib: str, model_path: str):
        r"""Reload the chat module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        """
        self.reload_func(lib, model_path)

    def prefill(self, input: str):
        r"""Run prefill stage for a given input and decode the first output token.

        Parameters
        ----------
        input : str
            The user input string.
        """
        self.prefill_func(input)

    def decode(self):
        r"""Decode the next token, the decoding result is stored in a buffer and
        can be retrieved by :func:`get_message`.
        """
        self.decode_func()

    def stopped(self) -> bool:
        r"""Check if the stop condition is met for the current round.

        Returns
        -------
        stopped : bool
        """
        return self.stopped_func() != 0

    def get_message(self) -> str:
        r"""Get the output message in the current round.

        Returns
        -------
        message : str

        Note
        ----
        This function returns the message that corresponds to
        all the tokens decoded so far.
        """
        return self.get_message_func()

    def reset_chat(self):
        r"""Reset the chat session and clear all chat history.

        Note
        ----
        The model remains the same after :func:`reset_chat`.
        To reload module, please use :func:`reload` instead.
        """
        self.reset_chat_func()

    def runtime_stats_text(self) -> str:
        r"""Get the runtime stats text (encoding speed and decoding speed).

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        return self.runtime_stats_text_func()

    def reset_runtime_stats(self):
        r"""Reset the runtime stats."""
        self.reset_runtime_stats_func()

    def evaluate(self):
        self.evaluate_func()
