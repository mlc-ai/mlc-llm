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

        tokenizer_path = model_path
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
        self.runtime_stats_text_func = self.chat_mod["runtime_stats_text"]
        self.evaluate_func = self.chat_mod["evaluate"]

    # pylint: disable=attribute-defined-outside-init
    def init_chat(self, model="vicuna", conv_template="vicuna_v1.1"):
        self.model = model
        self.conv_template = conv_template
        self.max_gen_len = 512 + 256
        self.temperature = 0.7
        self.top_p = 0.95
        self.stream_interval = 1
        self.max_window_size = 512 + 256
        self.mean_gen_len = 128
        self.shift_fill_factor = 0.2
        self.init_chat_func(
            self.model,
            self.conv_template,
            self.max_gen_len,
            self.temperature,
            self.top_p,
            self.stream_interval,
            self.max_window_size,
            self.mean_gen_len,
            self.shift_fill_factor,
        )

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


def from_llm_dylib(path, target="cuda"):
    return LLMChatModule(path, target)
