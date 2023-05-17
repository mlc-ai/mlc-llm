"""Python runtime for MLC chat."""

import os

import tvm


def load_llm_chat():
    import ctypes

    return ctypes.CDLL("/root/mlc-llm/build/libmlc_llm_module.so")


MLC_LLM = load_llm_chat()


class LLMChatModule:
    def __init__(self, model_path, target="cuda"):
        fcreate = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate is not None
        model_lib_path = model_path.split("/")[-1] + "-" + target + ".so"
        lib = tvm.runtime.load_module(os.path.join(model_path, model_lib_path))
        assert lib is not None
        params_path = os.path.join(model_path, "params")
        if target == "cuda":
            device_type = tvm.cuda().device_type
        elif target == "metal":
            device_type = tvm.metal(0).device_type
        else:
            raise ValueError("device type not supported yet")

        self.chat_mod = fcreate(lib, model_path, params_path, device_type, 0)
        self.encode_func = self.chat_mod["encode"]
        self.decode_func = self.chat_mod["decode"]
        self.stopped_func = self.chat_mod["stopped"]
        self.get_message_func = self.chat_mod["get_message"]
        self.reset_chat_func = self.chat_mod["reset_chat"]
        self.runtime_stats_text_func = self.chat_mod["reset_runtime_stats"]
        self.evaluate_func = self.chat_mod["evaluate"]

    def encode(self):
        pass

    def decode(self):
        pass

    def get_message(self):
        pass

    def stopped(self):
        pass

    def reset_chat(self):
        pass

    def runtime_stats_text(self):
        pass

    def evaluate(self):
        pass


def from_llm_dylib(path):
    """returns LLMChatModule object"""
