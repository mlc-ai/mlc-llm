"""The MLC LLM Serving Engine."""
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import tvm
from transformers import AutoTokenizer  # pylint: disable=import-error
from tvm.runtime import Device

from mlc_chat.support.auto_device import detect_device

from ..chat_module import _get_chat_config, _get_lib_module_path, _get_model_path
from . import data
from .config import GenerationConfig, KVCacheConfig
from .request import Request


@dataclass
class ModelInfo:
    """The model info dataclass.

    Parameters
    ----------
    model : str
        The identifier of the input model.
        It may be a compiled model's id (e.g., "Llama-2-7b-chat-hf-q4f16_1"),
        or a full path to a model directory
        (e.g., "dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1")

    device : str
        The device where to run the model.
        It can be "auto", "device_name" (e.g., "cuda") or
        "device_name:device_id" (e.g., "cuda:1").

    lib_path : Optional[str]
        The compiled library of the model.
        When specified, it is a path to the model library,
        e.g., "dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    """

    model: str
    device: Device = "auto"  # type: ignore
    lib_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = detect_device(self.device)
        assert isinstance(self.device, Device)


def _create_tvm_module(creator: str, ffi_funcs: Sequence[str]) -> Dict[str, Callable]:
    """Internal method to create a module."""
    module = tvm.get_global_func(creator, allow_missing=False)()
    return {key: module[key] for key in ffi_funcs}


class Engine:
    """The Python interface of request serving engine for MLC LLM.

    The engine can run one or multiple LLM models internally for
    text generation. Usually, when there are multiple models,
    speculative inference will be activated, where the first model
    (index 0) is the main "large model" that has better generation
    quality, and all other models are "small" models that used for
    speculation.

    The engine receives requests from the "add_request" method. For
    an given request, the engine will keep generating new tokens for
    the request until finish (under certain criterion). After finish,
    the engine will return the generation result through the callback
    function provided by the request.

    Parameters
    ----------
    models : Union[ModelInfo, List[ModelInfo]]
        One or a list of model info (specifying which models to load and
        which device to load to) to launch the engine.

    kv_cache_config : KVCacheConfig
        The configuration of the paged KV cache.
    """

    def __init__(
        self,
        models: Union[ModelInfo, List[ModelInfo]],
        kv_cache_config: KVCacheConfig,
    ):
        max_single_sequence_length = int(1e9)
        tokenizer_path: Optional[str] = None

        def _convert_model_info(model: ModelInfo) -> List[Any]:
            nonlocal max_single_sequence_length, tokenizer_path

            device = model.device
            model_path, config_file_path = _get_model_path(model.model)
            chat_config = _get_chat_config(config_file_path, user_chat_config=None)
            if chat_config.max_window_size:
                max_single_sequence_length = min(
                    max_single_sequence_length,
                    chat_config.max_window_size,
                )
            if tokenizer_path is None:
                tokenizer_path = model_path
            lib_path = _get_lib_module_path(
                model=model.model,
                model_path=model_path,
                chat_config=chat_config,
                model_lib_path=model.lib_path,
                device_name=device.MASK2STR[device.device_type],
                config_file_path=config_file_path,
            )
            return [lib_path, model_path, device.device_type, device.device_id]

        if isinstance(models, list):
            model_args: List[Any] = sum(
                (_convert_model_info(model) for model in models),
                start=[],
            )
        else:
            model_args = _convert_model_info(models)
        self._ffi = _create_tvm_module(
            "mlc.serve.create_engine",
            ffi_funcs=["init", "add_request", "abort", "step", "stats", "reset"],
        )
        self._ffi["init"](
            max_single_sequence_length,
            tokenizer_path,
            kv_cache_config.asjson(),
            *model_args,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate(
        self,
        prompts: Union[str, List[str], List[int], List[List[int]]],
        generation_config: Union[GenerationConfig, List[GenerationConfig]],
    ) -> List[str]:
        """Generate texts for a list of input prompts.
        Each prompt can be a string or a list of token ids.
        The generation for each prompt is independent.
        Return the generation results, one for each prompt.

        Parameters
        ----------
        prompts : Union[str, List[str], List[int], List[List[int]]]
            One or a list of input prompts for text generation.
            Each prompt can be a string or a list of token ids.

        generation_config : Union[GenerationConfig, List[GenerationConfig]]
            The generation config for each requests.
            If the it is a single GenerationConfig instance,
            this config will be shared by all the prompts.
            Otherwise, one generation config is required for every
            prompt.

        Returns
        -------
        results : List[str]
            The text generation results, one string for each input prompt.
        """
        if isinstance(prompts, str):
            # `prompts` is a single string.
            prompts = [prompts]
        else:
            assert isinstance(prompts, list), (
                "Input `prompts` is expected to be a string, a list of "
                "str, a list of token ids or multiple lists of token ids."
            )
            if len(prompts) == 0:
                return []
            if isinstance(prompts[0], int):
                # `prompts` is a list of token ids
                prompts = [prompts]  # type: ignore

        num_requests = len(prompts)
        if not isinstance(generation_config, list):
            generation_config = [generation_config] * num_requests

        assert (
            len(generation_config) == num_requests
        ), "Number of generation config and number of prompts mismatch"

        num_finished_requests = 0
        outputs: List[List[int]] = [[] for _ in range(num_requests)]

        # Define the callback function for request generation results
        def fcallback(
            request_id: str,
            token_data: data.TokenData,
            finished: bool,  # pylint: disable=unused-argument
        ):
            nonlocal num_finished_requests
            outputs[int(request_id)] += token_data.token_ids
            if finished:
                num_finished_requests += 1

        # Add requests to engine.
        for req_id, (prompt, generation_cfg) in enumerate(zip(prompts, generation_config)):
            input_data = (
                data.TextData(prompt)
                if isinstance(prompt, str)
                else data.TokenData(prompt)  # type: ignore
            )
            self.add_request(
                Request(
                    request_id=str(req_id),
                    inputs=input_data,
                    generation_config=generation_cfg,
                    fcallback=fcallback,
                )
            )

        while num_finished_requests != num_requests:
            self.step()

        output_strs = []
        for output in outputs:
            output_strs.append(self.detokenize(output))
        return output_strs

    def add_request(self, request: Request) -> None:
        """Add a new request to the engine.

        Parameters
        ----------
        request : Request
            The request to add.
        """
        self._ffi["add_request"](request)

    def abort_request(self, request_id: str) -> None:
        """Abort the generation of the request corresponding to the input request id.

        Parameters
        ----------
        request_id : str
            The unique id of the request to abort.
        """
        self._ffi["abort"](request_id)

    def step(self) -> None:
        """The main function that the engine takes a step of action.

        At each step, the engine may decide to
        - run prefill for one (or more) requests,
        - run one-step decode for the all existing requests
        ...

        In the end of certain actions (e.g., decode), the engine will
        check if any request has finished, and will return the
        generation results for those finished requests.
        """
        self._ffi["step"]()

    def reset(self) -> None:
        """Reset the engine, clean up all running data and statistics."""
        self._ffi["reset"]()

    def stats(self) -> Dict[str, float]:
        """The engine runtime statistics.
        We collect the following entries:
        - single token prefill latency (s/tok): avg latency of processing one token in prefill
        - single token decode latency (s/tok): avg latency of processing one token in decode
        - engine time for prefill (sec)
        - engine time for decode (sec)
        - total number of processed tokens in prefill.
        - total number of processed tokens in decode.
        """
        stats_json_str = self._ffi["stats"]()
        return json.loads(stats_json_str)

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize the given token ids to strings with the tokenizer
        that backs the engine.

        Parameters
        ----------
        token_ids : List[int]
            The token ids to decode.

        Returns
        -------
        output : str
            The detokenized text string.
        """
        return self._tokenizer.decode(token_ids)
