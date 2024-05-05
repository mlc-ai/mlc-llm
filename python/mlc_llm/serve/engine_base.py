"""The MLC LLM Serving engine base class."""

# pylint: disable=too-many-lines

import ast
import asyncio
import json
import queue
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
from mlc_llm.serve.config import EngineConfig, GenerationConfig
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

    model_lib : Optional[str]
        The path to the compiled library of the model.
        E.g., "dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    """

    model: str
    model_lib: Optional[str] = None


def _parse_models(
    model: str, model_lib: Optional[str], additional_models: Optional[List[str]]
) -> List[ModelInfo]:
    """Parse the specified model paths and model libs.
    Return a list of ModelInfo, which is a wrapper class of the model path + lib path.

    Each additional model is expected to follow the format of either
    "{MODEL_PATH}" or "{MODEL_PATH}:{MODEL_LIB}".
    """
    models = [ModelInfo(model, model_lib)]
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

        if model.model_lib is not None:
            # do model lib search if the model lib is provided
            # error out if file not found
            model_lib = _get_lib_module_path(
                model=model.model,
                model_path=model_path,
                chat_config=chat_config,
                model_lib=model.model_lib,
                device_name=device.MASK2STR[device.device_type],
                config_file_path=config_file_path,
            )
        else:
            # TODO(mlc-team) add logging information
            # Run jit if model_lib is not provided
            from mlc_llm.interface import jit  # pylint: disable=import-outside-toplevel

            model_lib = str(
                jit.jit(
                    model_path=Path(model_path),
                    chat_config=asdict(chat_config),
                    device=device,
                )
            )
        return model_path, model_lib

    model_args: List[Tuple[str, str]] = [_convert_model_info(model) for model in models]

    assert conversation is not None
    return model_args, config_file_paths, conversation


def _print_engine_mode_logging_msg(mode: Literal["local", "interactive", "server"]) -> None:
    """Print the logging info for engine mode selection."""
    if mode == "local":
        logger.info(
            "The selected engine mode is %s. "
            "We choose small max batch size and KV cache capacity to use less GPU memory.",
            green(mode),
        )
    elif mode == "interactive":
        logger.info(
            "The selected engine mode is %s. "
            "We fix max batch size to 1 for interactive single sequence use.",
            green(mode),
        )
    else:
        logger.info(
            "The selected engine mode is %s. "
            "We use as much GPU memory as possible (within the limit "
            "of gpu_memory_utilization).",
            green(mode),
        )

    if mode != "local":
        logger.info(
            "If you have low concurrent requests and want to use less GPU memory, "
            'please select mode "local".'
        )
    if mode != "interactive":
        logger.info(
            "If you don't have concurrent requests and only use the engine interactively, "
            'please select mode "interactive".'
        )
    if mode != "server":
        logger.info(
            "If you have high concurrent requests and want to maximize the GPU memory utilization, "
            'please select mode "server".'
        )


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
        model_lib: Optional[str],
        mode: Literal["local", "interactive", "server"],
        additional_models: Optional[List[str]],
        max_batch_size: Optional[int],
        max_total_sequence_length: Optional[int],
        prefill_chunk_size: Optional[int],
        max_history_size: Optional[int],
        gpu_memory_utilization: Optional[float],
        speculative_mode: Literal["disable", "small_draft", "eagle"],
        spec_draft_length: int,
        enable_tracing: bool,
        verbose: bool,
    ) -> None:
        # - Initialize model loading info.
        models = _parse_models(model, model_lib, additional_models)
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
            model_info.model_lib = model_args[i][1]
            with open(model_config_paths[i], "r", encoding="utf-8") as file:
                self.model_config_dicts.append(json.load(file))

        # - Print logging info for regarding the mode selection.
        if verbose:
            _print_engine_mode_logging_msg(mode)

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
                "init_threaded_engine",
                "exit_background_loop",
                "get_default_generation_config",
                "get_complete_engine_config",
                "stats",
                "debug_call_func_on_all_worker",
            ]
        }
        self.tokenizer = Tokenizer(model_args[0][0])
        self._ffi["init_threaded_engine"](
            device,
            self.state.get_request_stream_callback(kind),
            self.state.trace_recorder,
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

        self._ffi["reload"](
            EngineConfig(
                model=model_args[0][0],
                model_lib=model_args[0][1],
                additional_models=[model_arg[0] for model_arg in model_args[1:]],
                additional_model_libs=[model_arg[1] for model_arg in model_args[1:]],
                mode=mode,
                gpu_memory_utilization=gpu_memory_utilization,
                kv_cache_page_size=16,
                max_num_sequence=max_batch_size,
                max_total_sequence_length=max_total_sequence_length,
                prefill_chunk_size=prefill_chunk_size,
                max_history_size=max_history_size,
                speculative_mode=speculative_mode,
                spec_draft_length=spec_draft_length,
                verbose=verbose,
            ).asjson()
        )
        self.default_generation_cfg_json_str: str = self._ffi["get_default_generation_config"]()
        self.engine_config = EngineConfig.from_json(self._ffi["get_complete_engine_config"]())
        self.max_input_sequence_length = min(
            self.engine_config.max_single_sequence_length,
            self.engine_config.max_total_sequence_length,
        )

    def terminate(self):
        """Terminate the engine."""
        self._terminated = True
        self._ffi["exit_background_loop"]()
        self._background_loop_thread.join()
        self._background_stream_back_loop_thread.join()

    def _debug_call_func_on_all_worker(self, func_name: str) -> None:
        """Call the given global function on all workers. Only for debug purpose."""
        self._ffi["debug_call_func_on_all_worker"](func_name)

    def stats(self):
        """Get the engine stats."""
        return self._ffi["stats"]()


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


def process_completion_request(
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    engine_state: EngineState,
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
    generation_cfg = protocol_utils.get_generation_config(request)

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
