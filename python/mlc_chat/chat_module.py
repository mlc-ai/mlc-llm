"""Python runtime for MLC chat."""

import tvm


class LLMChatModule:
    def __init__(self):
        self.encode_func = None
        self.decode_func = None
        self.stopped_func = None
        self.get_message_func = None
        self.reset_chat_func = None
        self.runtime_stats_text_func = None

        # TODO: why is the function not registered?
        fcreate = tvm._ffi.get_global_func("mlc.llm_chat_create")

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
