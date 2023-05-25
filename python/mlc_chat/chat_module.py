"""Python runtime for MLC chat."""

import ctypes

import tvm


def load_llm_chat(mlc_lib_path):
    return ctypes.CDLL(mlc_lib_path)


def supported_models():
    return set(["vicuna-v1-7b", "RedPajama-INCITE-Chat-3B-v1"])


def quantization_keys():
    return ["q3f16_0", "q4f16_0", "q4f32_0", "q0f32", "q0f16"]


class LLMChatModule:
    def __init__(self, mlc_lib_path, target="cuda", device_id=0):
        load_llm_chat(mlc_lib_path)
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
