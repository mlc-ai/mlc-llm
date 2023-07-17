"""Chat module for MLC chat in a standalone file, including image module for multimodal-purposes."""
#! pylint: disable=unused-import, invalid-name
import ctypes
import os
import sys

import tvm
import tvm._ffi.base

from enum import Enum
from . import libinfo


def _load_mlc_llm_lib():
    """Load mlc llm lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib_name = "mlc_llm" if tvm._ffi.base._RUNTIME_ONLY else "mlc_llm_module"
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


# only load once here
if os.environ.get("SKIP_LOADING_MLCLLM_SO", "0") == "0":
    _LIB, _LIB_PATH = _load_mlc_llm_lib()


def quantization_keys():
    return [
        "q3f16_0",
        "q4f16_0",
        "q4f16_1",
        "q4f32_0",
        "q8f16_0",
        "q0f16",
        "q0f32",
    ]


class PlaceInPrompt(Enum):
    """The place of an input message in a prompt."""

    # The input message should have role names and corresponding seperators appended both prior to it and after it,
    # making it a complete prompt.
    All = 0
    # The input message is only the beginning part of a prompt, no role name and separator should be appended after
    # the message since there will be future messages appended after the message.
    Begin = 1
    # The input message is in the middle of a prompt, nothing should be appended before or after the message.
    Middle = 2
    # The input message is the ending part of a prompt, no role name and separator should be appended prior to it
    # since the message is concatenated to some prior messages.
    End = 3


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
        if target == "cuda":
            self.device = tvm.cuda(device_id)
        elif target == "metal":
            self.device = tvm.metal(device_id)
        elif target == "vulkan":
            self.device = tvm.vulkan(device_id)
        else:
            raise ValueError("device type not supported yet")
        device_type = self.device.device_type

        fcreate_chat_mod = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate_chat_mod is not None
        chat_mod = fcreate_chat_mod(device_type, device_id)
        fcreate_image_mod = tvm.get_global_func("mlc.llm_image_module_create")
        assert fcreate_image_mod is not None
        image_mod = fcreate_image_mod(device_type, device_id)

        # chat module related functions
        self.reload_func = chat_mod["reload"]
        self.prefill_func = chat_mod["prefill"]
        self.embed_func = chat_mod["embed"]
        self.prefill_with_embed_func = chat_mod["prefill_with_embed"]
        self.decode_func = chat_mod["decode"]
        self.stopped_func = chat_mod["stopped"]
        self.get_message_func = chat_mod["get_message"]
        self.reset_chat_func = chat_mod["reset_chat"]
        self.runtime_stats_text_func = chat_mod["runtime_stats_text"]
        self.reset_runtime_stats_func = chat_mod["reset_runtime_stats"]
        self.process_system_prompts_func = chat_mod["process_system_prompts"]
        self.evaluate_func = chat_mod["evaluate"]
        self.get_role0 = chat_mod["get_role0"]
        self.get_role1 = chat_mod["get_role1"]
        # image module related functions
        self.image_reload_func = image_mod["reload"]
        self.image_embed_func = image_mod["embed"]
        self.image_reset_func = image_mod["reset"]
        self.image_runtime_stats_text_func = image_mod["runtime_stats_text"]
        self.image_reset_runtime_stats_func = image_mod["reset_runtime_stats"]

    def reload(self, lib: str, model_path: str, app_config_json: str = ""):
        r"""Reload the chat module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        app_config_json: str
            The partial config that is used to partially override the model configuration.
        """
        self.reload_func(lib, model_path, app_config_json)

    def prefill(
        self,
        input: str,
        decode_next_token: bool = True,
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
    ):
        r"""Run prefill stage for a given input and optionally decode the first output token.
        User can decide where to place the input in the prompt.

        Parameters
        ----------
        input : str
            The user input string.
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt.
        """
        self.prefill_func(input, decode_next_token, place_in_prompt.value)

    def embed(
        self,
        input: str,
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
    ):
        r"""Given a text input, get the embedding of the tokenized prompt.
        User can decide where to place the input in the prompt.

        Parameters
        ----------
        input : str
            The user input string.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt.
        """
        return self.embed_func(input, place_in_prompt.value)

    def prefill_with_embed(
        self, embedding: tvm.runtime.NDArray, decode_next_token: bool = True
    ):
        r"""Given an embedding, run the prefill stage and optionally decode the first output token.

        Parameters
        ----------
        embedding : tvm.runtime.NDArray
            The embedding of user input.
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        """
        self.prefill_with_embed_func(embedding, decode_next_token)

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

    def process_system_prompts(self):
        r"""Pre-process by prefilling the system prompts, running prior to any user input."""
        self.process_system_prompts_func()

    def evaluate(self):
        self.evaluate_func()

    def reload_image_module(self, lib: str, model_path: str):
        r"""Reload the image module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        """
        self.reload_func(lib, model_path)

    def reset_image_module(self):
        r"""Reset the image module, clear its performance record.

        Note
        ----
        The model remains the same after :func:`reset_image_module`.
        To reload module, please use :func:`reload` instead.
        """
        self.reset_image_module_func()

    def get_image_embedding(
        self,
        image: tvm.runtime.NDArray,
    ):
        r"""Given an image of type NDArray, get the embedding of the image.

        Parameters
        ----------
        image : tvm.runtime.NDArray
            The user uploaded image.
        """
        return self.embed_func(image)

    def image_module_runtime_stats_text(self) -> str:
        r"""Get the runtime stats text (image encoding speed).

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        return self.runtime_stats_text_func()

    def reset_image_module_runtime_stats(self):
        r"""Reset the runtime stats."""
        self.reset_runtime_stats_func()
