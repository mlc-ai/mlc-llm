"""The MLC LLM Serving Engine."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import tvm
from tvm.runtime import Device

from mlc_chat.serve import data
from mlc_chat.support import logging
from mlc_chat.support.auto_device import detect_device
from mlc_chat.support.style import green

from ..chat_module import _get_chat_config, _get_lib_module_path, _get_model_path
from ..streamer import TextStreamer
from ..tokenizer import Tokenizer
from . import data
from .config import EngineMode, GenerationConfig, KVCacheConfig
from .event_trace_recorder import EventTraceRecorder
from .request import Request

logging.enable_logging()
logger = logging.getLogger(__name__)


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

    model_lib_path : str
        The path to the compiled library of the model.
        E.g., "dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    """

    model: str
    model_lib_path: str
    device: Device = "auto"  # type: ignore

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = detect_device(self.device)
        assert isinstance(self.device, Device)


def _create_tvm_module(
    creator: str, ffi_funcs: Sequence[str], creator_args: Optional[List[Any]] = None
) -> Dict[str, Callable]:
    """Internal method to create a module."""
    if creator_args is None:
        creator_args = []
    module = tvm.get_global_func(creator, allow_missing=False)(*creator_args)
    return {key: module[key] for key in ffi_funcs}


def _process_model_args(
    models: List[ModelInfo],
) -> Tuple[List[Any], List[str], str, int, int, Optional[str]]:
    """Process the input ModelInfo to get the engine initialization arguments."""
    max_single_sequence_length = int(1e9)
    prefill_chunk_size = int(1e9)
    tokenizer_path: Optional[str] = None
    conv_template_name: Optional[str] = None
    config_file_paths: List[str] = []

    def _convert_model_info(model: ModelInfo) -> List[Any]:
        nonlocal max_single_sequence_length, prefill_chunk_size, tokenizer_path, conv_template_name

        device = model.device
        model_path, config_file_path = _get_model_path(model.model)
        config_file_paths.append(config_file_path)
        chat_config = _get_chat_config(config_file_path, user_chat_config=None)
        if chat_config.context_window_size:
            max_single_sequence_length = min(
                max_single_sequence_length,
                chat_config.context_window_size,
            )
        if chat_config.prefill_chunk_size:
            prefill_chunk_size = min(prefill_chunk_size, chat_config.prefill_chunk_size)
        if tokenizer_path is None:
            tokenizer_path = model_path
        if conv_template_name is None:
            conv_template_name = chat_config.conv_template
        # Try look up model library, and do JIT compile if model library not found.
        try:
            model_lib_path = _get_lib_module_path(
                model=model.model,
                model_path=model_path,
                chat_config=chat_config,
                model_lib_path=model.model_lib_path,
                device_name=device.MASK2STR[device.device_type],
                config_file_path=config_file_path,
            )
        except FileNotFoundError:
            from mlc_chat.interface import (  # pylint: disable=import-outside-toplevel
                jit,
            )

            model_lib_path = str(
                jit.jit(
                    model_path=Path(model_path),
                    chat_config=asdict(chat_config),
                    device=device,
                )
            )
        return [model_lib_path, model_path, device.device_type, device.device_id]

    model_args: List[Any] = sum(
        (_convert_model_info(model) for model in models),
        start=[],
    )

    return (
        model_args,
        config_file_paths,
        tokenizer_path,
        max_single_sequence_length,
        prefill_chunk_size,
        conv_template_name,
    )


def _estimate_max_total_sequence_length(  # pylint: disable=too-many-locals
    models: List[ModelInfo], config_file_paths: List[str], max_num_sequence: int
) -> int:
    """Estimate the max total sequence length (capacity) of the KV cache."""
    assert len(models) != 0

    kv_bytes_per_token = 0
    kv_aux_workspace_bytes = 0
    model_workspace_bytes = 0
    logit_processor_workspace_bytes = 0
    params_bytes = 0
    temp_func_bytes = 0

    for model, config_file_path in zip(models, config_file_paths):
        # Read metadata for the parameter size and the temporary memory size.
        cmd = [
            sys.executable,
            "-m",
            "mlc_chat.cli.model_metadata",
            model.model_lib_path,
            "--print-memory-usage-in-json",
            "--mlc-chat-config",
            config_file_path,
        ]
        usage_str = subprocess.check_output(cmd, universal_newlines=True)
        usage_json = json.loads(usage_str)
        params_bytes += usage_json["params_bytes"]
        temp_func_bytes = max(temp_func_bytes, usage_json["temp_func_bytes"])

        cmd = [
            sys.executable,
            "-m",
            "mlc_chat.cli.model_metadata",
            model.model_lib_path,
            "--print-kv-cache-metadata-in-json",
        ]
        kv_cache_metadata_str = subprocess.check_output(cmd, universal_newlines=True)
        kv_cache_metadata = json.loads(kv_cache_metadata_str)

        # Read model config and compute the kv size per token.
        with open(config_file_path, mode="rt", encoding="utf-8") as file:
            json_object = json.load(file)
            model_config = json_object["model_config"]
            vocab_size = model_config["vocab_size"]
            prefill_chunk_size = model_config["prefill_chunk_size"]
            num_layers = kv_cache_metadata["num_hidden_layers"]
            head_dim = kv_cache_metadata["head_dim"]
            num_qo_heads = kv_cache_metadata["num_attention_heads"]
            num_kv_heads = kv_cache_metadata["num_key_value_heads"]
            hidden_size = head_dim * num_qo_heads
        kv_bytes_per_token += head_dim * num_kv_heads * num_layers * 4 + 1.25
        kv_aux_workspace_bytes += (
            (max_num_sequence + 1) * 88
            + prefill_chunk_size * (num_qo_heads + 1) * 8
            + prefill_chunk_size * head_dim * (num_qo_heads + num_kv_heads) * 4
            + 48 * 1024 * 1024
        )
        model_workspace_bytes += (
            prefill_chunk_size * 4
            + max_num_sequence * 4
            + (prefill_chunk_size * 2 + max_num_sequence) * hidden_size * 2
        )
        logit_processor_workspace_bytes += (
            max_num_sequence * 20 + max_num_sequence * vocab_size * 16.125
        )

    # Get single-card GPU size.
    gpu_size_bytes = os.environ.get("MLC_GPU_SIZE_BYTES", default=None)
    if gpu_size_bytes is None:
        gpu_size_bytes = models[0].device.total_global_memory
        if gpu_size_bytes is None:
            raise ValueError(
                "Cannot read total GPU global memory from device. "
                'Please the GPU memory size in bytes through "MLC_GPU_SIZE_BYTES" env variable.'
            )

    max_total_sequence_length = int(
        (
            int(gpu_size_bytes) * 0.90
            - params_bytes
            - temp_func_bytes
            - kv_aux_workspace_bytes
            - model_workspace_bytes
            - logit_processor_workspace_bytes
        )
        / kv_bytes_per_token
    )
    assert max_total_sequence_length > 0, (
        "Cannot estimate KV cache capacity. "
        f"The model weight size {params_bytes} may be larger than GPU memory size {gpu_size_bytes}"
    )

    total_size = (
        params_bytes
        + temp_func_bytes
        + kv_aux_workspace_bytes
        + model_workspace_bytes
        + logit_processor_workspace_bytes
        + kv_bytes_per_token * max_total_sequence_length
    )
    logger.info(
        "%s: %d.",
        green('Estimated KVCacheConfig "max_total_sequence_length"'),
        max_total_sequence_length,
    )
    logger.info(
        "%s: %.2f MB (Parameters: %.2f MB. KVCache: %.2f MB. Temporary buffer: %.2f MB)",
        green("Estimated total single GPU memory usage"),
        total_size / 1024 / 1024,
        params_bytes / 1024 / 1024,
        (kv_bytes_per_token * max_total_sequence_length + kv_aux_workspace_bytes) / 1024 / 1024,
        (model_workspace_bytes + logit_processor_workspace_bytes + temp_func_bytes) / 1024 / 1024,
    )
    return int(max_total_sequence_length)


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

    request_stream_callback : Optional[Callable[[str, data.TokenData, Optional[str]], None]]
        The provided callback function to handle the generation
        output. It has the signature of `(str, data.TokenData, bool) -> None`,
        where
        - the first string is the request id,
        - the TokenData contains the generated **delta** token ids since
        the last invocation of the callback on the specific request,
        - the optional string value denotes the finish reason if the
        generation of the request is finished, or None if it has not finished.

        The callback function is optional at construction, but it needs to
        be set before the engine executing requests. This can be done via
        the `set_request_stream_callback` method. Otherwise, the engine will raise
        exception.

    engine_mode : Optional[EngineMode]
        The Engine execution mode.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        models: Union[ModelInfo, List[ModelInfo]],
        kv_cache_config: KVCacheConfig,
        engine_mode: Optional[EngineMode] = None,
        request_stream_callback: Optional[Callable[[List[data.RequestStreamOutput]], None]] = None,
        enable_tracing: bool = False,
    ):
        if isinstance(models, ModelInfo):
            models = [models]
        (
            model_args,
            config_file_paths,
            tokenizer_path,
            self.max_single_sequence_length,
            prefill_chunk_size,
            self.conv_template_name,
        ) = _process_model_args(models)
        self._ffi = _create_tvm_module(
            "mlc.serve.create_engine",
            ffi_funcs=[
                "init",
                "add_request",
                "abort_request",
                "step",
                "stats",
                "reset",
                "get_request_stream_callback",
                "set_request_stream_callback",
            ],
        )
        self.trace_recorder = EventTraceRecorder() if enable_tracing else None

        if kv_cache_config.max_total_sequence_length is None:
            kv_cache_config.max_total_sequence_length = _estimate_max_total_sequence_length(
                models, config_file_paths, kv_cache_config.max_num_sequence
            )
        if kv_cache_config.prefill_chunk_size is None:
            kv_cache_config.prefill_chunk_size = prefill_chunk_size
        elif kv_cache_config.prefill_chunk_size > prefill_chunk_size:
            raise ValueError(
                f"The specified prefill chunk size {kv_cache_config.prefill_chunk_size} is "
                f"larger than the maximum prefill chunk size {prefill_chunk_size} supported by "
                "models. Please specify a smaller prefill chunk size."
            )

        if engine_mode is None:
            # The default engine mode: non-speculative
            engine_mode = EngineMode()

        self._ffi["init"](
            self.max_single_sequence_length,
            tokenizer_path,
            kv_cache_config.asjson(),
            engine_mode.asjson(),
            request_stream_callback,
            self.trace_recorder,
            *model_args,
        )
        self.tokenizer = Tokenizer(tokenizer_path)

    def generate(  # pylint: disable=too-many-locals
        self,
        prompts: Union[str, List[str], List[int], List[List[int]]],
        generation_config: Union[GenerationConfig, List[GenerationConfig]],
    ) -> Tuple[List[List[str]], List[Optional[List[List[str]]]]]:
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
        output_text : List[List[str]]
            The text generation results, one list of strings for each input prompt.
            The length of each list is the parallel generation `n` in
            generation config.

        output_logprobs_str : List[Optional[List[List[str]]]]
            The logprob strings of each token for each input prompt, or None
            if an input prompt does not require logprobs.
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
                return [], []
            if isinstance(prompts[0], int):
                # `prompts` is a list of token ids
                prompts = [prompts]  # type: ignore

        num_requests = len(prompts)
        if not isinstance(generation_config, list):
            generation_config = [generation_config] * num_requests

        assert (
            len(generation_config) == num_requests
        ), "Number of generation config and number of prompts mismatch"

        num_finished_generations = 0
        output_texts: List[List[str]] = []
        output_logprobs_str: List[Optional[List[List[str]]]] = []
        text_streamers: List[List[TextStreamer]] = []
        for i in range(num_requests):
            output_texts.append([])
            output_logprobs_str.append([] if generation_config[i].logprobs else None)
            text_streamers.append([])
            for _ in range(generation_config[i].n):
                output_texts[i].append("")
                text_streamers[i].append(TextStreamer(self.tokenizer))
                if output_logprobs_str[i] is not None:
                    output_logprobs_str[i].append([])

        num_total_generations = sum(cfg.n for cfg in generation_config)

        # Save a copy of the original function callback since `generate`
        # overrides the callback function.
        # The original callback will be set back later on.
        original_callback = self._ffi["get_request_stream_callback"]()

        # Define the callback function for request generation results
        def request_stream_callback(delta_outputs: List[data.RequestStreamOutput]):
            nonlocal num_finished_generations
            for delta_output in delta_outputs:
                request_id, stream_outputs = delta_output.unpack()
                rid = int(request_id)

                assert len(stream_outputs) == generation_config[rid].n
                for i, (stream_output, text_streamer) in enumerate(
                    zip(stream_outputs, text_streamers[rid])
                ):
                    if output_logprobs_str[rid] is not None:
                        assert stream_output.delta_logprob_json_strs is not None
                        output_logprobs_str[rid][i] += stream_output.delta_logprob_json_strs

                    delta_text = (
                        text_streamer.put(stream_output.delta_token_ids)
                        if len(stream_output.delta_token_ids) > 0
                        else ""
                    )
                    if stream_output.finish_reason is not None:
                        delta_text += text_streamer.finish()

                    output_texts[rid][i] += delta_text
                    if stream_output.finish_reason is not None:
                        num_finished_generations += 1

        # Override the callback function in engine.
        self._ffi["set_request_stream_callback"](request_stream_callback)

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
                )
            )

        while num_finished_generations != num_total_generations:
            self.step()

        # Restore the callback function in engine.
        self._ffi["set_request_stream_callback"](original_callback)
        return output_texts, output_logprobs_str

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
        self._ffi["abort_request"](request_id)

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
