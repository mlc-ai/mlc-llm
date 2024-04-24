"""The MLC LLM Serving engine base class."""

# pylint: disable=too-many-lines

import ast
import asyncio
import json
import queue
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import tvm
from tvm.runtime import Device

from mlc_llm.chat_module import _get_chat_config, _get_lib_module_path, _get_model_path
from mlc_llm.protocol import openai_api_protocol, protocol_utils
from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.serve import data, engine_utils
from mlc_llm.serve.config import EngineConfig, GenerationConfig, SpeculativeMode
from mlc_llm.serve.event_trace_recorder import EventTraceRecorder
from mlc_llm.streamer import TextStreamer
from mlc_llm.support import logging
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green
from mlc_llm.tokenizer import Tokenizer

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

    model_lib_path : Optional[str]
        The path to the compiled library of the model.
        E.g., "dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    """

    model: str
    model_lib_path: Optional[str] = None


def _parse_models(
    model: str, model_lib_path: Optional[str], additional_models: Optional[List[str]]
) -> List[ModelInfo]:
    """Parse the specified model paths and model lib paths.
    Return a list of ModelInfo, which is a wrapper class of the model path + lib path.

    Each additional model is expected to follow the format of either
    "{MODEL_PATH}" or "{MODEL_PATH}:{MODEL_LIB_PATH}".
    """
    models = [ModelInfo(model, model_lib_path)]
    if additional_models is not None:
        for additional_model in additional_models:
            splits = additional_model.split(":", maxsplit=1)
            if len(splits) == 2:
                models.append(ModelInfo(splits[0], splits[1]))
            else:
                models.append(ModelInfo(splits[0]))
    return models


def _process_model_args(
    models: List[ModelInfo], device: tvm.runtime.Device
) -> Tuple[List[Tuple[str, str]], List[str], Conversation]:
    """Process the input ModelInfo to get the engine initialization arguments."""
    conversation: Optional[Conversation] = None
    config_file_paths: List[str] = []

    def _convert_model_info(model: ModelInfo) -> Tuple[str, str]:
        nonlocal conversation

        model_path, config_file_path = _get_model_path(model.model)
        config_file_paths.append(config_file_path)
        chat_config = _get_chat_config(config_file_path, user_chat_config=None)
        if conversation is None:
            assert isinstance(chat_config.conv_template, Conversation)
            conversation = chat_config.conv_template

        if model.model_lib_path is not None:
            # do model lib search if the model lib path is provided
            # error out if file not found
            model_lib_path = _get_lib_module_path(
                model=model.model,
                model_path=model_path,
                chat_config=chat_config,
                model_lib_path=model.model_lib_path,
                device_name=device.MASK2STR[device.device_type],
                config_file_path=config_file_path,
            )
        else:
            # TODO(mlc-team) add logging information
            # Run jit if model_lib_path is not provided
            from mlc_llm.interface import jit  # pylint: disable=import-outside-toplevel

            model_lib_path = str(
                jit.jit(
                    model_path=Path(model_path),
                    chat_config=asdict(chat_config),
                    device=device,
                )
            )
        return model_path, model_lib_path

    model_args: List[Tuple[str, str]] = [_convert_model_info(model) for model in models]

    assert conversation is not None
    return model_args, config_file_paths, conversation


def _estimate_mem_usage_and_max_total_sequence_length(  # pylint: disable=too-many-locals,too-many-arguments
    models: List[ModelInfo],
    device: tvm.runtime.Device,
    model_config_paths: List[str],
    model_config_dicts: List[Dict[str, Any]],
    max_num_sequence: int,
    gpu_memory_utilization: Optional[float],
) -> Tuple[float, float, float, float, float, int]:
    """Estimate the memory usage and the max total sequence length (capacity)
    that the KV cache can support.
    """
    assert len(models) != 0

    kv_bytes_per_token = 0
    kv_aux_workspace_bytes = 0
    model_workspace_bytes = 0
    logit_processor_workspace_bytes = 0
    params_bytes = 0
    temp_func_bytes = 0

    for model, model_config_path, model_config_dict in zip(
        models, model_config_paths, model_config_dicts
    ):
        # Read metadata for the parameter size and the temporary memory size.
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm.cli.model_metadata",
            model.model_lib_path,
            "--print-memory-usage-in-json",
            "--mlc-chat-config",
            model_config_path,
        ]
        usage_str = subprocess.check_output(cmd, universal_newlines=True)
        usage_json = json.loads(usage_str)
        params_bytes += usage_json["params_bytes"]
        temp_func_bytes = max(temp_func_bytes, usage_json["temp_func_bytes"])

        cmd = [
            sys.executable,
            "-m",
            "mlc_llm.cli.model_metadata",
            model.model_lib_path,
            "--print-kv-cache-metadata-in-json",
        ]
        kv_cache_metadata_str = subprocess.check_output(cmd, universal_newlines=True)
        kv_cache_metadata = json.loads(kv_cache_metadata_str)

        # Read model config and compute the kv size per token.
        model_config = model_config_dict["model_config"]
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
    gpu_size_bytes = device.total_global_memory
    if gpu_size_bytes is None:
        raise ValueError("Cannot read total GPU global memory from device.")
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.85

    model_max_total_sequence_length = int(
        (
            int(gpu_size_bytes) * gpu_memory_utilization
            - params_bytes
            - temp_func_bytes
            - kv_aux_workspace_bytes
            - model_workspace_bytes
            - logit_processor_workspace_bytes
        )
        / kv_bytes_per_token
    )
    if model_max_total_sequence_length <= 0:
        raise ValueError(
            f"The model weight size {params_bytes} may be larger than available GPU memory "
            f"size {gpu_size_bytes * gpu_memory_utilization} bytes."
        )

    if device.device_type == Device.kDLMetal:
        # NOTE: Metal runtime has severe performance issues with large buffers.
        # To work around the issue, we limit the KV cache capacity to 32768.
        model_max_total_sequence_length = min(model_max_total_sequence_length, 32768)

    total_mem_usage_except_kv_cache = (
        params_bytes
        + temp_func_bytes
        + kv_aux_workspace_bytes
        + model_workspace_bytes
        + logit_processor_workspace_bytes
    )
    return (
        total_mem_usage_except_kv_cache,
        params_bytes,
        kv_bytes_per_token,
        kv_aux_workspace_bytes,
        model_workspace_bytes + logit_processor_workspace_bytes + temp_func_bytes,
        int(model_max_total_sequence_length),
    )


def _get_model_config_limit(model_config_dicts: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Read the model config dictionaries, and return the maximum single
    sequence length the models can support, the maximum prefill chunk
    size the models can support, and the max batch size the models can support.

    Returns
    -------
    model_max_single_sequence_length : int
        The maximum single sequence length the models can support.
    model_max_prefill_chunk_size : int
        The maximum prefill chunk size the models can support.
    model_max_batch_size : int
        The max batch size the models can support.
    """
    model_max_single_sequence_length = int(1e9)
    model_max_prefill_chunk_size = int(1e9)
    model_max_batch_size = int(1e9)
    for i, config in enumerate(model_config_dicts):
        runtime_context_window_size = config["context_window_size"]
        compile_time_context_window_size = config["model_config"]["context_window_size"]
        if runtime_context_window_size > compile_time_context_window_size:
            raise ValueError(
                f"Model {i}'s runtime context window size ({runtime_context_window_size}) is "
                "larger than the context window size used at compile time "
                f"({compile_time_context_window_size})"
            )
        if runtime_context_window_size == -1 and compile_time_context_window_size != -1:
            raise ValueError(
                f"Model {i}'s runtime context window size (infinite) is "
                "larger than the context window size used at compile time "
                f"({compile_time_context_window_size})"
            )
        if runtime_context_window_size != -1:
            model_max_single_sequence_length = min(
                model_max_single_sequence_length, runtime_context_window_size
            )

        runtime_prefill_chunk_size = config["prefill_chunk_size"]
        compile_time_prefill_chunk_size = config["model_config"]["prefill_chunk_size"]
        if runtime_prefill_chunk_size > compile_time_prefill_chunk_size:
            raise ValueError(
                f"Model {i}'s runtime prefill chunk size ({runtime_prefill_chunk_size}) is "
                "larger than the prefill chunk size used at compile time "
                f"({compile_time_prefill_chunk_size})"
            )
        model_max_prefill_chunk_size = min(model_max_prefill_chunk_size, runtime_prefill_chunk_size)

        model_max_batch_size = min(model_max_batch_size, config["model_config"]["max_batch_size"])

    assert model_max_prefill_chunk_size != int(1e9)
    assert model_max_batch_size != int(1e9)
    return model_max_single_sequence_length, model_max_prefill_chunk_size, model_max_batch_size


def _infer_kv_cache_config(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    mode: Literal["local", "interactive", "server"],
    max_batch_size: Optional[int],
    max_total_sequence_length: Optional[int],
    prefill_chunk_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    models: List[ModelInfo],
    device: tvm.runtime.Device,
    model_config_dicts: List[Dict[str, Any]],
    model_config_paths: List[str],
) -> Tuple[int, int, int, int]:
    """Initialize the KV cache config with user input and GPU memory usage estimation.
    The returned four integers are:
    - max_batch_size
    - max_total_sequence_length
    - prefill_chunk_size
    - model_max_single_sequence_length
    """
    (
        model_max_single_sequence_length,
        model_max_prefill_chunk_size,
        model_max_batch_size,
    ) = _get_model_config_limit(model_config_dicts)

    def infer_args_under_mode(
        mode: Literal["local", "interactive", "server"],
        max_batch_size: Optional[int],
        max_total_sequence_length: Optional[int],
        prefill_chunk_size: Optional[int],
    ) -> Tuple[Tuple[int, int, int], List[float]]:
        logging_msg = ""
        # - max_batch_size
        if max_batch_size is None:
            max_batch_size = (
                min(4, model_max_batch_size)
                if mode == "local"
                else (1 if mode == "interactive" else model_max_batch_size)
            )
            logging_msg += f"max batch size is set to {max_batch_size}, "
        else:
            logging_msg += f"max batch size {max_batch_size} is specified by user, "
        # - infer the maximum total sequence length that can fit GPU memory.
        (
            total_mem_usage_except_kv_cache,
            model_params_bytes,
            kv_bytes_per_token,
            kv_aux_workspace_bytes,
            temp_workspace_bytes,
            model_max_total_sequence_length,
        ) = _estimate_mem_usage_and_max_total_sequence_length(
            models,
            device,
            model_config_paths,
            model_config_dicts,
            max_batch_size,
            gpu_memory_utilization,
        )
        # - max_total_sequence_length
        if max_total_sequence_length is None:
            if mode == "local":
                max_total_sequence_length = min(
                    model_max_total_sequence_length, model_max_single_sequence_length, 8192
                )
            elif mode == "interactive":
                max_total_sequence_length = min(
                    model_max_total_sequence_length, model_max_single_sequence_length
                )
            else:
                max_total_sequence_length = min(
                    model_max_total_sequence_length,
                    max_batch_size * model_max_single_sequence_length,
                )
            logging_msg += f"max KV cache token capacity is set to {max_total_sequence_length}, "
        else:
            logging_msg += (
                f"max KV cache token capacity {max_total_sequence_length} is specified by user. "
            )
        # - prefill_chunk_size
        if prefill_chunk_size is None:
            if mode in ["local", "interactive"]:
                prefill_chunk_size = min(
                    model_max_prefill_chunk_size,
                    model_max_total_sequence_length,
                    model_max_single_sequence_length,
                )
            else:
                prefill_chunk_size = model_max_prefill_chunk_size
            logging_msg += f"prefill chunk size is set to {prefill_chunk_size}. "
        else:
            logging_msg += f"prefill chunk size {prefill_chunk_size} is specified by user. "

        if mode == "local":
            logging_msg += (
                "We choose small max batch size and KV cache capacity to use less GPU memory."
            )
        elif mode == "interactive":
            logging_msg += "We fix max batch size to 1 for interactive single sequence use."
        else:
            logging_msg += (
                "We use as much GPU memory as possible (within the"
                " limit of gpu_memory_utilization)."
            )
        logger.info('Under mode "%s", %s', mode, logging_msg)

        # - Construct the KV cache config
        # - Estimate total GPU memory usage on single GPU.
        return (max_batch_size, max_total_sequence_length, prefill_chunk_size), [
            total_mem_usage_except_kv_cache + max_total_sequence_length * kv_bytes_per_token,
            model_params_bytes,
            kv_bytes_per_token * max_total_sequence_length + kv_aux_workspace_bytes,
            temp_workspace_bytes,
        ]

    # - Infer KV cache config and estimate memory usage for each mode.
    local_kv_cache_config, local_mem_usage_list = infer_args_under_mode(
        "local", max_batch_size, max_total_sequence_length, prefill_chunk_size
    )
    interactive_kv_cache_config, interactive_mem_usage_list = infer_args_under_mode(
        "interactive", max_batch_size, max_total_sequence_length, prefill_chunk_size
    )
    server_kv_cache_config, server_mem_usage_list = infer_args_under_mode(
        "server", max_batch_size, max_total_sequence_length, prefill_chunk_size
    )

    # - Select the config based on the actual mode.
    if mode == "local":
        kv_cache_config = local_kv_cache_config
        mem_usage_list = local_mem_usage_list
    elif mode == "interactive":
        kv_cache_config = interactive_kv_cache_config
        mem_usage_list = interactive_mem_usage_list
    else:
        kv_cache_config = server_kv_cache_config
        mem_usage_list = server_mem_usage_list

    logger.info(
        'The actual engine mode is "%s". So max batch size is %s, '
        "max KV cache token capacity is %s, prefill chunk size is %s.",
        green(mode),
        green(str(kv_cache_config[0])),
        green(str(kv_cache_config[1])),
        green(str(kv_cache_config[2])),
    )

    logger.info(
        "%s: %.2f MB (Parameters: %.2f MB. KVCache: %.2f MB. Temporary buffer: %.2f MB). "
        "The actual usage might be slightly larger than the estimated number.",
        green("Estimated total single GPU memory usage"),
        *list(mem_usage / 1024 / 1024 for mem_usage in mem_usage_list),
    )
    # - Final messages
    override_msg = "Please override the arguments if you have particular values to set."
    if mode in ["local", "interactive"]:
        logger.info(
            'Please switch to mode "server" if you want to use more GPU memory '
            "and support more concurrent requests. %s",
            override_msg,
        )
    else:
        logger.info(
            'Please switch to mode "local" or "interactive" if you want to use less GPU memory '
            "or do not have many concurrent requests to process. %s",
            override_msg,
        )

    return *kv_cache_config, model_max_single_sequence_length


@dataclass
class CallbackStreamOutput:
    """The output of MLCEngine._generate and AsyncMLCEngine._generate

    Attributes
    ----------
    delta_text : str
        The delta text generated since the last output.

    num_delta_tokens : int
        The number of delta tokens generated since the last output.

    delta_logprob_json_strs : Optional[List[str]]
        The list of logprob JSON strings since the last output,
        or None if the request does not require logprobs.

    finish_reason : Optional[str]
        The finish reason of the request, or None if unfinished.
    """

    delta_text: str
    num_delta_tokens: int
    delta_logprob_json_strs: Optional[List[str]]
    finish_reason: Optional[str]


class AsyncRequestStream:
    """The asynchronous stream for requests in AsyncMLCEngine.

    Each request has its own unique stream.
    The stream exposes the method `push` for engine to push new generated
    delta text to the stream, and the method `finish` for engine to mark
    the finish of generation.

    The stream implements `__aiter__` and `__anext__`, which the engine
    can use to iterates all the generated tokens in order asynchronously.
    """

    # The asynchronous queue to hold elements of either a list of
    # CallbackStreamOutput or an exception.
    if sys.version_info >= (3, 9):
        _queue: asyncio.Queue[  # pylint: disable=unsubscriptable-object
            Union[List[CallbackStreamOutput], Exception]
        ]
    else:
        _queue: asyncio.Queue
    # The finish flag.
    _finished: bool

    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._finished = False

    def push(self, item_or_exception: Union[List[CallbackStreamOutput], Exception]) -> None:
        """Push a new token to the stream."""
        if self._finished:
            # No new item is expected after finish.
            self._queue.put_nowait(
                RuntimeError(
                    "The request has already finished. "
                    "The stream is not supposed to accept new items."
                )
            )
            return
        self._queue.put_nowait(item_or_exception)

    def finish(self) -> None:
        """Mark the finish of the generation in the stream."""
        self._queue.put_nowait(StopIteration())
        self._finished = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> List[CallbackStreamOutput]:
        result = await self._queue.get()
        if isinstance(result, StopIteration):
            raise StopAsyncIteration
        if isinstance(result, Exception):
            raise result
        return result


class EngineState:
    """The engine states that the request stream callback function may use.

    This class is used for both AsyncMLCEngine and MLCEngine.
    AsyncMLCEngine uses the fields and methods starting with "async",
    and MLCEngine uses the ones starting with "sync".

    - For AsyncMLCEngine, the state contains an asynchronous event loop,
    the streamers and the number of unfinished generations for each request
    being processed.
    - For MLCEngine, the state contains a callback output blocking queue,
    the text streamers and the number of unfinished requests.

    We use this state class to avoid the callback function from capturing
    the AsyncMLCEngine.

    The state also optionally maintains an event trace recorder, which can
    provide Chrome tracing when enabled.
    """

    trace_recorder = None
    # States used for AsyncMLCEngine
    async_event_loop: Optional[asyncio.AbstractEventLoop] = None
    async_streamers: Dict[str, Tuple[AsyncRequestStream, List[TextStreamer]]] = {}
    async_num_unfinished_generations: Dict[str, int] = {}
    # States used for MLCEngine
    sync_output_queue: queue.Queue = queue.Queue()
    sync_text_streamers: List[TextStreamer] = []
    sync_num_unfinished_generations: int = 0

    def __init__(self, enable_tracing: bool) -> None:
        """Constructor."""
        if enable_tracing:
            self.trace_recorder = EventTraceRecorder()

    def record_event(self, request_id: str, event: str) -> None:
        """Record a event for the input request in the trace
        recorder when the recorder exists.

        Parameters
        ----------
        request_id : str
            The subject request of the event.

        event : str
            The event in a string name.
            It can have one of the following patterns:
            - "start xxx", which marks the start of event "xxx",
            - "finish xxx", which marks the finish of event "xxx",
            - "yyy", which marks the instant event "yyy".
            The "starts" and "finishes" will be automatically paired in the trace recorder.
        """
        if self.trace_recorder is None:
            return
        self.trace_recorder.add_event(request_id, event)

    def get_request_stream_callback(
        self, kind: Literal["async", "sync"]
    ) -> Callable[[List[data.RequestStreamOutput]], None]:
        """Construct a callback function and return.

        The callback function has signature
        "Callable[[List[data.RequestStreamOutput]], None]",
        whose input is a list of "data.RequestStreamOutput".
        Each "data.RequestStreamOutput" is the delta output of a request,
        generated from the engine.
        """

        f_callback = (
            self._async_request_stream_callback
            if kind == "async"
            else self._sync_request_stream_callback
        )

        def _callback(delta_outputs: List[data.RequestStreamOutput]) -> None:
            f_callback(delta_outputs)

        return _callback

    def async_lazy_init_event_loop(self) -> None:
        """Lazily set the asyncio event loop so that the event
        loop is the main driving event loop of the process.
        """
        if self.async_event_loop is None:
            self.async_event_loop = asyncio.get_event_loop()

    def _async_request_stream_callback(self, delta_outputs: List[data.RequestStreamOutput]) -> None:
        """The request stream callback function for AsyncMLCEngine to stream back
        the request generation results.

        Note
        ----
        This callback function uses `call_soon_threadsafe` in asyncio to
        schedule the invocation in the event loop, so that the underlying
        callback logic will be executed asynchronously in the future rather
        than right now.
        """

        # Schedule a callback run in the event loop without executing right now.
        # NOTE: This function causes GIL during execution.
        self.async_event_loop.call_soon_threadsafe(
            self._async_request_stream_callback_impl, delta_outputs
        )

    def _async_request_stream_callback_impl(
        self, delta_outputs: List[data.RequestStreamOutput]
    ) -> None:
        """The underlying implementation of request stream callback for AsyncMLCEngine."""
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            streamers = self.async_streamers.get(request_id, None)
            if streamers is None:
                continue

            self.record_event(request_id, event="start callback")
            stream, text_streamers = streamers
            outputs = []
            for stream_output, text_streamer in zip(stream_outputs, text_streamers):
                self.record_event(request_id, event="start detokenization")
                delta_text = (
                    text_streamer.put(stream_output.delta_token_ids)
                    if len(stream_output.delta_token_ids) > 0
                    else ""
                )
                if stream_output.finish_reason is not None:
                    delta_text += text_streamer.finish()
                self.record_event(request_id, event="finish detokenization")

                outputs.append(
                    CallbackStreamOutput(
                        delta_text=delta_text,
                        num_delta_tokens=len(stream_output.delta_token_ids),
                        delta_logprob_json_strs=stream_output.delta_logprob_json_strs,
                        finish_reason=stream_output.finish_reason,
                    )
                )
                if stream_output.finish_reason is not None:
                    self.async_num_unfinished_generations[request_id] -= 1

            # Push new delta text to the stream.
            stream.push(outputs)
            if self.async_num_unfinished_generations[request_id] == 0:
                stream.finish()
                self.async_streamers.pop(request_id, None)
                self.async_num_unfinished_generations.pop(request_id, None)
            self.record_event(request_id, event="finish callback")

    def _sync_request_stream_callback(self, delta_outputs: List[data.RequestStreamOutput]) -> None:
        """The request stream callback function for MLCEngine to stream back
        the request generation results.
        """
        # Put the delta outputs to the queue in the unblocking way.
        self.sync_output_queue.put_nowait(delta_outputs)


class MLCEngineBase:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """The base engine class, which implements common functions that
    are shared by MLCEngine and AsyncMLCEngine.

    This class wraps a threaded engine that runs on a standalone
    thread inside and streams back the delta generated results via
    callback functions. The internal threaded engine keeps running an
    loop that drives the engine.

    MLCEngine and AsyncMLCEngine inherits this MLCEngineBase class, and implements
    their own methods to process the delta generated results received
    from callback functions and yield the processed delta results in
    the forms of standard API protocols.

    Checkout subclasses AsyncMLCEngine/MLCEngine for the docstring of constructor parameters.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        kind: Literal["async", "sync"],
        model: str,
        device: Union[str, tvm.runtime.Device],
        model_lib_path: Optional[str],
        mode: Literal["local", "interactive", "server"],
        additional_models: Optional[List[str]],
        max_batch_size: Optional[int],
        max_total_sequence_length: Optional[int],
        prefill_chunk_size: Optional[int],
        gpu_memory_utilization: Optional[float],
        speculative_mode: SpeculativeMode,
        spec_draft_length: int,
        enable_tracing: bool,
    ) -> None:
        # - Initialize model loading info.
        models = _parse_models(model, model_lib_path, additional_models)
        if isinstance(device, str):
            device = detect_device(device)
        assert isinstance(device, Device)
        (
            model_args,
            model_config_paths,
            self.conv_template,
        ) = _process_model_args(models, device)

        # - Load the raw model config into dict
        self.model_config_dicts = []
        for i, model_info in enumerate(models):
            model_info.model_lib_path = model_args[i][1]
            with open(model_config_paths[i], "r", encoding="utf-8") as file:
                self.model_config_dicts.append(json.load(file))

        # - Decide the KV cache config based on mode and user input.
        (
            max_batch_size,
            max_total_sequence_length,
            prefill_chunk_size,
            max_single_sequence_length,
        ) = _infer_kv_cache_config(
            mode,
            max_batch_size,
            max_total_sequence_length,
            prefill_chunk_size,
            gpu_memory_utilization,
            models,
            device,
            self.model_config_dicts,
            model_config_paths,
        )
        self.max_input_sequence_length = min(max_single_sequence_length, max_total_sequence_length)

        # - Initialize engine state and engine.
        self.state = EngineState(enable_tracing)
        module = tvm.get_global_func("mlc.serve.create_threaded_engine", allow_missing=False)()
        self._ffi = {
            key: module[key]
            for key in [
                "add_request",
                "abort_request",
                "run_background_loop",
                "run_background_stream_back_loop",
                "reload",
                "init_background_engine",
                "exit_background_loop",
                "debug_call_func_on_all_worker",
            ]
        }
        self.tokenizer = Tokenizer(model_args[0][0])
        self._ffi["init_background_engine"](
            self.state.get_request_stream_callback(kind),
            self.state.trace_recorder,
        )
        self._ffi["reload"](
            EngineConfig(
                model=model_args[0][0],
                model_lib_path=model_args[0][1],
                additional_models=[model_arg[0] for model_arg in model_args[1:]],
                additional_model_lib_paths=[model_arg[1] for model_arg in model_args[1:]],
                device=device,
                kv_cache_page_size=16,
                max_num_sequence=max_batch_size,
                max_total_sequence_length=max_total_sequence_length,
                max_single_sequence_length=max_single_sequence_length,
                prefill_chunk_size=prefill_chunk_size,
                speculative_mode=speculative_mode,
                spec_draft_length=spec_draft_length,
            )
        )

        def _background_loop():
            self._ffi["run_background_loop"]()

        def _background_stream_back_loop():
            self._ffi["run_background_stream_back_loop"]()

        # - Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(target=_background_loop)
        self._background_stream_back_loop_thread: threading.Thread = threading.Thread(
            target=_background_stream_back_loop
        )
        self._background_loop_thread.start()
        self._background_stream_back_loop_thread.start()
        self._terminated = False

    def terminate(self):
        """Terminate the engine."""
        self._terminated = True
        self._ffi["exit_background_loop"]()
        self._background_loop_thread.join()
        self._background_stream_back_loop_thread.join()

    def _debug_call_func_on_all_worker(self, func_name: str) -> None:
        """Call the given global function on all workers. Only for debug purpose."""
        self._ffi["debug_call_func_on_all_worker"](func_name)


def process_chat_completion_request(  # pylint: disable=too-many-arguments
    request: openai_api_protocol.ChatCompletionRequest,
    request_id: str,
    engine_state: EngineState,
    model_config: Dict[str, Any],
    f_tokenize: Callable[[str], List[int]],
    max_input_sequence_length: int,
    conv_template: Conversation,
) -> Tuple[List[Union[List[int], data.Data]], GenerationConfig, bool, int]:
    """Process the given ChatCompletionRequest, apply request validity
    checks, and return the processed prompts, and other info.

    Parameters
    ----------
    request : openai_api_protocol.ChatCompletionRequest
        The request to be processed and checked.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    model_config : Dict[str, Any]
        The model configuration dictionary.

    f_tokenize : Callable[[str], List[int]]
        The tokenizer encode function.

    max_input_sequence_length : int
        The maximum allowed total prompt length.

    conv_template : Conversation
        The conversation template of the model.

    Returns
    -------
    prompts : List[Union[List[int], data.Data]]
        The prompts, in a list.
        Each element is a list of token ids or a "data.Data" instance.

    generation_cfg : GenerationConfig
        The generation config of the request got from the input request.

    use_function_calling : bool
        A boolean flag indicating if the request uses function call.

    prompt_length : int
        The total prompt length.
    """
    engine_state.record_event(request_id, event="receive request")
    # - Check if unsupported arguments are specified.
    engine_utils.check_unsupported_fields(request)

    # - Process messages and update the conversation template in three steps:
    #   i. Check the message validity.
    #  ii. Add the input messages to the conversation template.
    # iii. Add the additional message for the assistant.
    request.check_message_validity()
    # - Check for function calling usage and update the conversation template
    request.check_function_call_usage(conv_template)

    for message in request.messages:
        role = message.role
        content = message.content
        if role == "system":
            assert isinstance(content, str)
            conv_template.system_message = content if content is not None else ""
            continue
        assert role != "tool", "Internal error: tool role."
        conv_template.messages.append((role, content))
    conv_template.messages.append(("assistant", None))

    # - Get the prompt from template, and encode to token ids.
    # - Check prompt length
    engine_state.record_event(request_id, event="start tokenization")
    prompts = engine_utils.process_prompts(  # type: ignore
        conv_template.as_prompt(model_config), f_tokenize
    )
    engine_state.record_event(request_id, event="finish tokenization")

    if conv_template.system_prefix_token_ids is not None:
        if isinstance(prompts[0], list):
            prompts[0] = conv_template.system_prefix_token_ids + prompts[0]
        else:
            prompts.insert(0, conv_template.system_prefix_token_ids)
    prompt_length = engine_utils.check_and_get_prompts_length(prompts, max_input_sequence_length)

    # Process generation config. Create request id.
    generation_cfg = protocol_utils.get_generation_config(
        request,
        model_config,
        extra_stop_token_ids=conv_template.stop_token_ids,
        extra_stop_str=conv_template.stop_str,
    )
    return prompts, generation_cfg, conv_template.use_function_calling, prompt_length


def process_chat_completion_stream_output(  # pylint: disable=too-many-arguments
    delta_outputs: List[CallbackStreamOutput],
    request_id: str,
    engine_state: EngineState,
    model: str,
    generation_cfg: GenerationConfig,
    use_function_calling: bool,
    prompt_length: int,
    finish_reasons: List[Optional[str]],
    num_completion_tokens: int,
) -> Tuple[Optional[openai_api_protocol.ChatCompletionStreamResponse], int]:
    """Process the delta outputs of a single request of ChatCompletion,
    convert the delta output to ChatCompletionStreamResponse and return.

    Parameters
    ----------
    delta_outputs : List[CallbackStreamOutput]
        The delta outputs of a request.
        The list length is the number of parallel generation specified by "n".
        Each element corresponds to a generation.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    model : str
        The requested model.

    generation_cfg : GenerationConfig
        The generation config of the request.

    use_function_calling : bool
        A boolean flag indicating if the request uses function call.

    prompt_length : int
        The total prompt length.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    num_completion_tokens : int
        The number of total completion tokens so far.

    Returns
    -------
    response : Optional[openai_api_protocol.ChatCompletionStreamResponse]
        The converted OpenAI API ChatCompletionStreamResponse instance.
        It can be none when there is no content.

    num_completion_tokens : int
        The updated number of total completion tokens.
        It is sum of the input number and the number of new completion tokens
        from the given delta outputs.
    """
    assert len(delta_outputs) == generation_cfg.n
    choices = []
    num_new_completion_tokens = 0
    for i, delta_output in enumerate(delta_outputs):
        finish_reason_updated = False
        num_new_completion_tokens += delta_output.num_delta_tokens
        if delta_output.finish_reason is not None and finish_reasons[i] is None:
            finish_reasons[i] = (
                delta_output.finish_reason if not use_function_calling else "tool_calls"
            )
            finish_reason_updated = True
        if not finish_reason_updated and delta_output.delta_text == "":
            # Ignore empty delta text when finish reason is not updated.
            engine_state.record_event(request_id, event="skip empty delta text")
            continue

        choices.append(
            openai_api_protocol.ChatCompletionStreamResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                delta=openai_api_protocol.ChatCompletionMessage(
                    content=delta_output.delta_text, role="assistant"
                ),
                logprobs=(
                    openai_api_protocol.LogProbs(
                        content=[
                            openai_api_protocol.LogProbsContent.model_validate_json(
                                logprob_json_str
                            )
                            for logprob_json_str in delta_output.delta_logprob_json_strs
                        ]
                    )
                    if delta_output.delta_logprob_json_strs is not None
                    else None
                ),
            )
        )

    if len(choices) == 0 and num_new_completion_tokens == 0:
        # Skip return when there is no delta output and no number of completion tokens.
        return None, num_completion_tokens
    num_completion_tokens += num_new_completion_tokens
    response = openai_api_protocol.ChatCompletionStreamResponse(
        id=request_id,
        choices=choices,
        model=model,
        system_fingerprint="",
        usage=openai_api_protocol.UsageInfo(
            prompt_tokens=prompt_length,
            completion_tokens=num_completion_tokens,
        ),
    )
    engine_state.record_event(request_id, event="yield delta output")
    return response, num_completion_tokens


def process_completion_request(  # pylint: disable=too-many-arguments
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    engine_state: EngineState,
    model_config: Dict[str, Any],
    tokenizer: Tokenizer,
    max_input_sequence_length: int,
) -> Tuple[List[int], GenerationConfig, int, Optional[openai_api_protocol.CompletionResponse]]:
    """Process the given CompletionRequest, apply request validity
    checks, and return the processed prompts, and other info.

    Parameters
    ----------
    request : openai_api_protocol.CompletionRequest
        The request to be processed and checked.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    tokenizer : Tokenizer
        The tokenizer instance of the model.

    max_input_sequence_length : int
        The maximum allowed total prompt length.

    Returns
    -------
    prompt : List[int]
        The prompt in a list of token ids.

    generation_cfg : GenerationConfig
        The generation config of the request got from the input request.

    prompt_length : int
        The total prompt length.

    echo_response : Optional[openai_api_protocol.CompletionResponse]
        The CompletionResponse of the echoing part, when argument "echo"
        of the input request is specified.
    """
    engine_state.record_event(request_id, event="receive request")
    # - Check if unsupported arguments are specified.
    engine_utils.check_unsupported_fields(request)

    # - Process prompt and check validity.
    engine_state.record_event(request_id, event="start tokenization")
    prompts = engine_utils.process_prompts(request.prompt, tokenizer.encode)
    engine_state.record_event(request_id, event="finish tokenization")
    prompt_length = engine_utils.check_and_get_prompts_length(prompts, max_input_sequence_length)
    prompt = prompts[0]
    assert isinstance(prompt, list)

    # Process generation config. Create request id.
    generation_cfg = protocol_utils.get_generation_config(request, model_config)

    # - Echo back the prompt.
    echo_response = None
    if request.echo:
        text = tokenizer.decode(prompt)
        response = openai_api_protocol.CompletionResponse(
            id=request_id,
            choices=[
                openai_api_protocol.CompletionResponseChoice(index=i, text=text)
                for i in range(generation_cfg.n)
            ],
            model=request.model,
            usage=openai_api_protocol.UsageInfo(
                prompt_tokens=prompt_length,
                completion_tokens=0,
            ),
        )
        echo_response = response
    return prompt, generation_cfg, prompt_length, echo_response


def process_completion_stream_output(  # pylint: disable=too-many-arguments
    delta_outputs: List[CallbackStreamOutput],
    request_id: str,
    engine_state: EngineState,
    model: str,
    generation_cfg: GenerationConfig,
    prompt_length: int,
    finish_reasons: List[Optional[str]],
    num_completion_tokens: int,
) -> Tuple[Optional[openai_api_protocol.CompletionResponse], int]:
    """Process the delta outputs of a single request of Completion,
    convert the delta output to CompletionResponse and return.

    Parameters
    ----------
    delta_outputs : List[CallbackStreamOutput]
        The delta outputs of a request.
        The list length is the number of parallel generation specified by "n".
        Each element corresponds to a generation.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    model : str
        The requested model.

    generation_cfg : GenerationConfig
        The generation config of the request.

    prompt_length : int
        The total prompt length.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    num_completion_tokens : int
        The number of total completion tokens so far.

    Returns
    -------
    response : Optional[openai_api_protocol.CompletionResponse]
        The converted OpenAI API CompletionResponse instance.
        It can be none when there is no content.

    num_completion_tokens : int
        The updated number of total completion tokens.
        It is sum of the input number and the number of new completion tokens
        from the given delta outputs.
    """
    assert len(delta_outputs) == generation_cfg.n
    choices = []
    num_new_completion_tokens = 0
    for i, delta_output in enumerate(delta_outputs):
        finish_reason_updated = False
        if delta_output.finish_reason is not None and finish_reasons[i] is None:
            finish_reasons[i] = delta_output.finish_reason
            finish_reason_updated = True
        num_new_completion_tokens += delta_output.num_delta_tokens
        if not finish_reason_updated and delta_output.delta_text == "":
            # Ignore empty delta text when finish reason is not updated.
            continue

        choices.append(
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                text=delta_output.delta_text,
                logprobs=(
                    openai_api_protocol.LogProbs(
                        content=[
                            openai_api_protocol.LogProbsContent.model_validate_json(
                                logprob_json_str
                            )
                            for logprob_json_str in delta_output.delta_logprob_json_strs
                        ]
                    )
                    if delta_output.delta_logprob_json_strs is not None
                    else None
                ),
            )
        )

    if len(choices) == 0 and num_new_completion_tokens == 0:
        # Skip return when there is no delta output and no number of completion tokens.
        return None, num_completion_tokens
    num_completion_tokens += num_new_completion_tokens
    response = openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=choices,
        model=model,
        usage=openai_api_protocol.UsageInfo(
            prompt_tokens=prompt_length,
            completion_tokens=num_completion_tokens,
        ),
    )
    engine_state.record_event(request_id, event="yield delta output")
    return response, num_completion_tokens


def create_completion_suffix_response(
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    prompt_length: int,
    finish_reasons: List[Optional[str]],
    num_completion_tokens: int,
) -> Optional[openai_api_protocol.CompletionResponse]:
    """Create the suffix response of Completion request
    when the request requires suffix.

    Parameters
    ----------
    request : openai_api_protocol.CompletionRequest
        The request whose suffix response if to be created.

    request_id : str
        The id of the request.

    prompt_length : int
        The total prompt length.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    num_completion_tokens : int
        The number of total completion tokens so far.

    Returns
    -------
    suffix_response : Optional[openai_api_protocol.CompletionResponse]
        The created OpenAI API CompletionResponse instance for the suffix.
        Or None if the request does not require suffix.
    """
    # - Echo the suffix.
    if request.suffix is None:
        return None
    assert all(finish_reason is not None for finish_reason in finish_reasons)
    response = openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=[
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=request.suffix,
            )
            for i, finish_reason in enumerate(finish_reasons)
        ],
        model=request.model,
        usage=openai_api_protocol.UsageInfo(
            prompt_tokens=prompt_length,
            completion_tokens=num_completion_tokens,
        ),
    )
    return response


def convert_function_str_to_json(stringified_calls: str) -> List[Union[Dict, None]]:
    """Convert a (possibly list) of function call string to a list of json objects.
    Return None for invalid function call string."""

    def parse_function_call(call_str: str):
        node = ast.parse(call_str, mode="eval")
        call_node = node.body
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            arguments = {}
            for keyword in call_node.keywords:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            return {"name": name, "arguments": arguments}
        return None

    if (
        stringified_calls[0] == "[" and stringified_calls[-1] == "]"
    ):  # hacky way to check if string list
        calls = ast.literal_eval(stringified_calls)
    else:
        calls = [stringified_calls]
    function_calls_json = [parse_function_call(call_str) for call_str in calls]
    return function_calls_json


def process_function_call_output(
    output_texts: List[str], finish_reasons: List[str]
) -> Tuple[bool, List[List[openai_api_protocol.ChatToolCall]]]:
    """Process the potential function call results outputted by model,
    according to the finish reasons.
    Return whether the output has function call, and the list of tool calls.
    """
    n = len(output_texts)
    tool_calls_list: List[List[openai_api_protocol.ChatToolCall]] = [[] for _ in range(n)]
    use_function_calling = any(finish_reason == "tool_calls" for finish_reason in finish_reasons)
    if use_function_calling:
        for i, output_text in enumerate(output_texts):
            try:
                fn_json_list = convert_function_str_to_json(output_text)
            except (SyntaxError, ValueError):
                output_text = "Got an invalid function call output from model"
                finish_reasons[i] = "error"
            else:
                tool_calls_list[i] = [
                    openai_api_protocol.ChatToolCall(
                        type="function",
                        function=openai_api_protocol.ChatFunctionCall(
                            name=fn_json_obj["name"], arguments=fn_json_obj["arguments"]
                        ),
                    )
                    for fn_json_obj in fn_json_list
                    if fn_json_obj is not None
                ]
                if len(tool_calls_list[i]) == 0:
                    output_texts[i] = "Got an invalid function call output from model"
                    finish_reasons[i] = "error"
                else:
                    finish_reasons[i] = "tool_calls"
    return use_function_calling, tool_calls_list


def wrap_chat_completion_response(  # pylint: disable=too-many-arguments
    request_id: str,
    model: str,
    output_texts: List[str],
    finish_reasons: List[str],
    tool_calls_list: List[List[openai_api_protocol.ChatToolCall]],
    logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]],
    use_function_calling: bool,
    num_prompt_tokens: int,
    num_completion_tokens: int,
) -> openai_api_protocol.ChatCompletionResponse:
    """Wrap the non-streaming chat completion results to ChatCompletionResponse instance."""
    return openai_api_protocol.ChatCompletionResponse(
        id=request_id,
        choices=[
            openai_api_protocol.ChatCompletionResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                message=(
                    openai_api_protocol.ChatCompletionMessage(role="assistant", content=output_text)
                    if not use_function_calling or finish_reason == "error"
                    else openai_api_protocol.ChatCompletionMessage(
                        role="assistant", tool_calls=tool_calls
                    )
                ),
                logprobs=(
                    openai_api_protocol.LogProbs(content=logprob_results[i])
                    if logprob_results is not None
                    else None
                ),
            )
            for i, (output_text, finish_reason, tool_calls) in enumerate(
                zip(output_texts, finish_reasons, tool_calls_list)
            )
        ],
        model=model,
        system_fingerprint="",
        usage=openai_api_protocol.UsageInfo(
            prompt_tokens=num_prompt_tokens, completion_tokens=num_completion_tokens
        ),
    )


def wrap_completion_response(  # pylint: disable=too-many-arguments
    request_id: str,
    model: str,
    output_texts: List[str],
    finish_reasons: List[str],
    logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]],
    num_prompt_tokens: int,
    num_completion_tokens: int,
) -> openai_api_protocol.CompletionResponse:
    """Wrap the non-streaming completion results to CompletionResponse instance."""
    return openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=[
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=output_text,
                logprobs=(
                    openai_api_protocol.LogProbs(content=logprob_results[i])
                    if logprob_results is not None
                    else None
                ),
            )
            for i, (output_text, finish_reason) in enumerate(zip(output_texts, finish_reasons))
        ],
        model=model,
        usage=openai_api_protocol.UsageInfo(
            prompt_tokens=num_prompt_tokens, completion_tokens=num_completion_tokens
        ),
    )
