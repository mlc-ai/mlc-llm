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


def supported_models():
    return set(["vicuna-v1-7b", "RedPajama-INCITE-Chat-3B-v1"])


def quantization_keys():
    return ["q3f16_0", "q4f16_0", "q4f32_0", "q0f32", "q0f16"]


class ChatModule:
    def __init__(self, target="cuda", device_id=0):
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

    def reload(self, lib, model_path):
        self.reload_func(lib, model_path)

    def prefill(self, input):
        self.prefill_func(input)

    def decode(self):
        self.decode_func()

    def stopped(self):
        return self.stopped_func() != 0

    def get_message(self):
        return self.get_message_func()

    def reset_chat(self):
        self.reset_chat_func()

    def runtime_stats_text(self):
        return self.runtime_stats_text_func()

    def reset_runtime_stats(self):
        self.reset_runtime_stats_func()

    def evaluate(self):
        self.evaluate_func()
