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
        self.has_embed_func = chat_mod["has_embed"]
        self.embed_func = chat_mod["embed"]
        self.prefill_func = chat_mod["prefill"]
        self.prefill_with_embed_func = chat_mod["prefill_with_embed"]
        self.decode_func = chat_mod["decode"]
        self.stopped_func = chat_mod["stopped"]
        self.get_message_func = chat_mod["get_message"]
        self.reset_chat_func = chat_mod["reset_chat"]
        self.runtime_stats_text_func = chat_mod["runtime_stats_text"]
        self.reset_runtime_stats_func = chat_mod["reset_runtime_stats"]
        self.evaluate_func = chat_mod["evaluate"]
        self.get_role0 = chat_mod["get_role0"]
        self.get_role1 = chat_mod["get_role1"]
        self.process_system_prompts_func = chat_mod["process_system_prompts"]

    def reload(self, lib, model_path, app_config_json=""):
        self.reload_func(lib, model_path, app_config_json)

    def has_embed(self):
        return self.has_embed_func()

    def embed(self, text_input):
        return self.embed_func(text_input)

    def prefill(self, input):
        self.prefill_func(input)

    def prefill_with_embed(self, embedding):
        self.prefill_with_embed_func(embedding)

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

    def process_system_prompts(self):
        self.process_system_prompts_func()
