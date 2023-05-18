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

        tokenizer_path = os.path.join(model_path, "params")
        params_path = os.path.join(model_path, "params")

        if target == "cuda":
            device_type = tvm.cuda().device_type
        elif target == "metal":
            device_type = tvm.metal(0).device_type
        elif target == "vulkan":
            device_type = tvm.vulkan(0).device_type
        else:
            raise ValueError("device type not supported yet")

        self.chat_mod = fcreate(lib, tokenizer_path, params_path, device_type, 0)

        self.init_chat_func = self.chat_mod["init_chat"]
        self.encode_func = self.chat_mod["encode"]
        self.decode_func = self.chat_mod["decode"]
        self.stopped_func = self.chat_mod["stopped"]
        self.get_message_func = self.chat_mod["get_message"]
        self.reset_chat_func = self.chat_mod["reset_chat"]
        self.runtime_stats_text_func = self.chat_mod["reset_runtime_stats"]
        self.evaluate_func = self.chat_mod["evaluate"]

    def init_chat(self):
        model = "vicuna"
        conv_template = "vicuna_v1.1"
        max_gen_len = 512 + 256
        temperature = 0.7
        top_p = 0.95
        stream_interval = 1
        max_window_size = 512 + 256
        mean_gen_len = 128
        shift_fill_factor = 0.2
        self.init_chat_func(model, conv_template, max_gen_len, temperature, top_p, stream_interval, max_window_size, mean_gen_len, shift_fill_factor)

    def encode(self, prompt):
        self.encode_func(prompt)

    def decode(self):
        self.decode_func()

    def get_message(self):
        return self.get_message_func()

    def stopped(self):
        return self.stopped_func() != 0

    def reset_chat(self):
        self.reset_chat_func()

    def runtime_stats_text(self):
        return self.runtime_stats_text_func()

    def evaluate(self):
        self.evaluate_func()


def from_llm_dylib(path):
    """returns LLMChatModule object"""
