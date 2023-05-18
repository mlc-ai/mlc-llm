"""Python runtime for MLC chat."""


import os

import tvm


def load_llm_chat(mlc_lib_path):
    import ctypes

    return ctypes.CDLL(mlc_lib_path)


class LLMChatModule:
    def __init__(self, mlc_lib_path, model_path, target="cuda", device_id=0):
        load_llm_chat(mlc_lib_path)
        fcreate = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate is not None
        model_lib_name = model_path.split("/")[-1] + "-" + target + ".so"
        lib = tvm.runtime.load_module(os.path.join(model_path, model_lib_name))
        assert lib is not None
        if target == "cuda":
            device_type = tvm.cuda(device_id).device_type
        elif target == "metal":
            device_type = tvm.metal(device_id).device_type
        elif target == "vulkan":
            device_type = tvm.vulkan(device_id).device_type
        else:
            raise ValueError("device type not supported yet")

        chat_mod = fcreate(device_type, device_id)
        self.reload = chat_mod["reload"]
        self.reload(lib, os.path.join(model_path, "params"))
        self.encode_func = chat_mod["encode"]
        self.decode_func = chat_mod["decode"]
        self.stopped_func = chat_mod["stopped"]
        self.get_message_func = chat_mod["get_message"]
        self.reset_chat_func = chat_mod["reset_chat"]
        self.runtime_stats_text_func = chat_mod["runtime_stats_text"]
        self.evaluate_func = chat_mod["evaluate"]
        self.get_role0 = chat_mod["get_role0"]
        self.get_role1 = chat_mod["get_role1"]

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
