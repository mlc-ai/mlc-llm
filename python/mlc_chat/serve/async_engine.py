"""The MLC LLM Asynchronous Serving Engine.
Acknowledgment: Part of the code was adapted from the vLLM project.
"""
import asyncio
import sys
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import tvm

from ..streamer import StopStringHandler, TextStreamer
from ..tokenizer import Tokenizer
from . import data
from .config import EngineMode, GenerationConfig, KVCacheConfig
from .engine import ModelInfo, _estimate_max_total_sequence_length, _process_model_args
from .event_trace_recorder import EventTraceRecorder
from .request import Request, RequestStreamOutput


class AsyncRequestStream:
    """The asynchronous stream for requests.

    Each request has its own unique stream.
    The stream exposes the method `push` for engine to push new generated
    delta text to the stream, and the method `finish` for engine to mark
    the finish of generation.

    The stream implements `__aiter__` and `__anext__`, which the engine
    can use to iterates all the generated tokens in order asynchronously.
    """

    # The asynchronous queue to hold elements of
    # - either a tuple of (str, int, Optional[str]), denoting the
    #   delta output text, the number of delta tokens, the optional
    #   finish reason respectively,
    # - or an exception.
    if sys.version_info >= (3, 9):
        _queue: asyncio.Queue[  # pylint: disable=unsubscriptable-object
            Union[Tuple[str, int, Optional[str]], Exception]
        ]
    else:
        _queue: asyncio.Queue
    # The finish flag.
    _finished: bool

    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._finished = False

    def push(self, item_or_exception: Union[Tuple[str, int, Optional[str]], Exception]) -> None:
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

    async def __anext__(self) -> Tuple[str, int, Optional[str]]:
        result = await self._queue.get()
        if isinstance(result, StopIteration):
            raise StopAsyncIteration
        if isinstance(result, Exception):
            raise result
        return result


class AsyncThreadedEngine:  # pylint: disable=too-many-instance-attributes
    """The asynchronous engine for generate text asynchronously,
    backed by ThreadedEngine.

    This class wraps a synchronous threaded engine that runs on
    a standalone thread inside, and exports the asynchronous `generate`
    method as the main text generation interface, which yields the
    generated tokens. The internal threaded engine keeps running an
    event loop that drives the engine.

    Parameters
    ----------
    models : Union[ModelInfo, List[ModelInfo]]
        One or a list of model info (specifying which models to load and
        which device to load to) to launch the engine.

    kv_cache_config : KVCacheConfig
        The configuration of the paged KV cache.

    engine_mode : Optional[EngineMode]
        The Engine execution mode.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(
        self,
        models: Union[ModelInfo, List[ModelInfo]],
        kv_cache_config: KVCacheConfig,
        engine_mode: Optional[EngineMode] = None,
        enable_tracing: bool = False,
    ) -> None:
        if isinstance(models, ModelInfo):
            models = [models]
        (
            model_args,
            config_file_paths,
            tokenizer_path,
            self.max_single_sequence_length,
            self.conv_template_name,
        ) = _process_model_args(models)
        self.trace_recorder = EventTraceRecorder() if enable_tracing else None

        if kv_cache_config.max_total_sequence_length is None:
            kv_cache_config.max_total_sequence_length = _estimate_max_total_sequence_length(
                models, config_file_paths
            )

        module = tvm.get_global_func("mlc.serve.create_threaded_engine", allow_missing=False)()
        self._ffi = {
            key: module[key]
            for key in [
                "add_request",
                "abort_request",
                "run_background_loop",
                "init_background_engine",
                "exit_background_loop",
            ]
        }
        self.tokenizer = Tokenizer(tokenizer_path)
        if engine_mode is None:
            # The default engine mode: non-speculative
            engine_mode = EngineMode()

        # The mapping from request ids to request asynchronous stream.
        self._request_tools: Dict[
            str, Tuple[AsyncRequestStream, TextStreamer, StopStringHandler]
        ] = {}

        def _background_loop():
            self._ffi["init_background_engine"](
                self.max_single_sequence_length,
                tokenizer_path,
                kv_cache_config.asjson(),
                engine_mode.asjson(),
                self._request_stream_callback,
                self.trace_recorder,
                *model_args,
            )
            self._ffi["run_background_loop"]()

        # Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(target=_background_loop)
        self._background_loop_thread.start()
        # The main thread request handling asyncio event loop, which will
        # be lazily initialized.
        self._async_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._terminated = False

    def terminate(self):
        """Terminate the engine."""
        self._terminated = True
        self._ffi["exit_background_loop"]()
        self._background_loop_thread.join()

    async def generate(
        self, prompt: Union[str, List[int]], generation_config: GenerationConfig, request_id: str
    ) -> AsyncGenerator[Tuple[str, int, str], Any]:
        """Asynchronous text generation interface.
        The method is a coroutine that streams a tuple at a time via yield.
        Each tuple is contained of
        - the delta text in type str,
        - the number of delta tokens in type int,
        - the optional finish reason in type Optional[str].

        Parameters
        ----------
        prompt : Union[str, List[int]]
            The input prompt in forms of text string or a list of token ids.

        generation_config : GenerationConfig
            The generation config of the request.

        request_id : str
            The unique identifier (in string) or this generation request.
        """
        if self._terminated:
            raise ValueError("The AsyncThreadedEngine has terminated.")
        if self._async_event_loop is None:
            # Lazily set the asyncio event loop so that the event
            # loop is the main driving event loop of the process.
            self._async_event_loop = asyncio.get_event_loop()

        # Create the request with the given id, input data, generation
        # config and the created callback.
        input_data = data.TextData(prompt) if isinstance(prompt, str) else data.TokenData(prompt)
        request = Request(request_id, input_data, generation_config)

        # Create the unique stream of the request.
        stream = AsyncRequestStream()
        if request_id in self._request_tools:
            # Report error in the stream if the request id already exists.
            stream.push(
                RuntimeError(
                    f'The request id "{request_id} already exists. '
                    'Please make sure the request id is unique."'
                )
            )
        else:
            # Record the stream in the tracker
            self._request_tools[request_id] = (
                stream,
                TextStreamer(self.tokenizer),
                StopStringHandler(generation_config.stop_strs),
            )
            self._ffi["add_request"](request)

        # Iterate the stream asynchronously and yield the token.
        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:  # pylint: disable=broad-exception-caught
            await self.abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameter
        ---------
        request_id : str
            The id of the request to abort.
        """
        self._abort(request_id)

    def _abort(self, request_id: str):
        """Internal implementation of request abortion."""
        self._request_tools.pop(request_id, None)
        self._ffi["abort_request"](request_id)

    def _request_stream_callback(self, delta_outputs: List[RequestStreamOutput]) -> None:
        """The request stream callback function for engine to stream back
        the request generation results.

        Parameters
        ----------
        delta_outputs : List[RequestStreamOutput]
            The delta output of each requests.
            Check out RequestStreamOutput for the fields of the outputs.

        Note
        ----
        This callback function uses `call_soon_threadsafe` in asyncio to
        schedule the invocation in the event loop, so that the underlying
        callback logic will be executed asynchronously in the future rather
        than right now.
        """
        # Schedule a callback run in the event loop without executing right now.
        # NOTE: This function causes GIL during execution.
        self._async_event_loop.call_soon_threadsafe(
            self._request_stream_callback_impl, delta_outputs
        )

    def _request_stream_callback_impl(self, delta_outputs: List[RequestStreamOutput]) -> None:
        """The underlying implementation of request stream callback."""
        for delta_output in delta_outputs:
            request_id, delta_tokens, finish_reason = delta_output.unpack()
            tools = self._request_tools.get(request_id, None)
            if tools is None:
                continue

            self.record_event(request_id, event="start callback")
            stream, text_streamer, stop_handler = tools

            self.record_event(request_id, event="start detokenization")
            delta_token_ids = delta_tokens.token_ids
            delta_text = stop_handler.put(text_streamer.put(delta_token_ids))
            if stop_handler.stop_triggered:
                finish_reason = "stop"
                self._abort(request_id)
            elif finish_reason is not None:
                delta_text += stop_handler.put(text_streamer.finish())
                if stop_handler.stop_triggered:
                    finish_reason = "stop"
                    self._abort(request_id)
                else:
                    delta_text += stop_handler.finish()
            self.record_event(request_id, event="finish detokenization")

            # Push new delta text to the stream.
            stream.push((delta_text, len(delta_token_ids), finish_reason))
            if finish_reason is not None:
                stream.finish()
                self._request_tools.pop(request_id, None)
            self.record_event(request_id, event="finish callback")

    def record_event(self, request_id: str, event: str) -> None:
        """Record a event for the the input request in the trace
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
