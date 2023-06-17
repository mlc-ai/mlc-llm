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
    """
    A Python interface for the llm chat module.

    Attributes
    ----------
    target : str
        The device target for the chat module (default `cuda`)
    device_id : str
        The device id for the chat module (default `0`)
    """
    def __init__(
        self,
        target: str = "cuda",
        device_id: int = 0
    ):
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
        self.embed_func = chat_mod["embed"]
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

    def reload(
        self,
        lib: str,
        model_path: str
    ):
        """
        Reloads the compiled model and parameters specified at the input paths.

        Parameters
        ----------
        lib : str
            The path to the compiled model library
        model_path: str
            The path to the model parameters
        """
        self.reload_func(lib, model_path)

    def embed(
        self,
        input: str
    ):
        """
        Get the embeddings for a specified input prompt.

        Parameters
        ----------
        input : str
            The input prompt for which to generate embeddings

        Returns
        -------
        List[float]
            The embedding for the specified input prompt
        """
        return self.embed_func(input)

    def prefill(
        self,
        input: str
    ):
        """
        Encode and prefill the model using the specified input prompt.

        Parameters
        ----------
        input : str
            The input prompt which should be used to encode and prefill the model
        """
        self.prefill_func(input)

    def decode(self):
        """
        Decode the model output.
        """
        self.decode_func()

    def stopped(self):
        """
        Check if the model has stopped output.

        Returns
        -------
        bool
            A boolean indicating whether the model output has stopped
        """
        return self.stopped_func() != 0

    def get_message(self):
        """
        Get the latest output from the model. Should be called after `decode`.

        Returns
        -------
        str
            The latest output of the model
        """
        return self.get_message_func()

    def reset_chat(self):
        """
        Reset the state of the model. This will remove all history.
        """
        self.reset_chat_func()

    def runtime_stats_text(self):
        """
        Get some runtime statistics of the model, including encode and decode speed.

        Returns
        -------
        str
            The runtime statistics of the model
        """
        return self.runtime_stats_text_func()

    def reset_runtime_stats(self):
        """
        Reset the runtime statistics of the model.
        """
        self.reset_runtime_stats_func()

    def evaluate(self):
        """
        Run an evaluation of the model pipeline.
        """
        self.evaluate_func()
