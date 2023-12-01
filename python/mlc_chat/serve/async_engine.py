"""The MLC LLM Asynchronous Serving Engine.
Acknowledgment: Part of the code was adapted from the vLLM project.
"""
import asyncio
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from ..streamer import StopStringHandler, TextStreamer
from ..tokenizer import Tokenizer
from . import data
from .config import GenerationConfig, KVCacheConfig
from .engine import Engine, ModelInfo, _create_tvm_module, _process_model_args
from .request import Request


class AsyncEngineDeadError(RuntimeError):
    """The error class of asynchronous engine."""


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    msg = "Task finished unexpectedly. This should never happen!"
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(msg) from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        raise exc


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
    _queue: asyncio.Queue[Union[Tuple[str, int, Optional[str]], Exception]]
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


class AsyncRequestTracker:
    """The request trackers which manages the asynchronous stream
    of each request, and the queues of requests to add to or abort
    from the engine.
    """

    # The mapping from request ids to request asynchronous stream.
    _request_tools: Dict[str, Tuple[AsyncRequestStream, TextStreamer, StopStringHandler]]
    # The queue of requests to add into the engine.
    _requests_to_add: asyncio.Queue[Request]
    # The queue of requests (identified by request id) to abort from the engine.
    _requests_to_abort: asyncio.Queue[str]
    # The event of "there are new requests to add", used to avoid engine
    # from cyclically pull and waiting for new requests.
    _new_request_event: asyncio.Event

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._request_tools = {}
        self._requests_to_add = asyncio.Queue()
        self._requests_to_abort = asyncio.Queue()
        self._new_request_event = asyncio.Event()
        self._tokenizer = tokenizer

    def add_request(
        self, request_id: str, input_data: data.Data, generation_cfg: GenerationConfig
    ) -> AsyncRequestStream:
        """Add a new request to the tracker for engine to pull.

        Parameters
        ----------
        request_id : str
            The unique id of the request.

        input_data : data.Data
            The input data of the request.

        generation_cfg : GenerationConfig
            The generation config of the request.

        Returns
        -------
        stream : AsyncRequestStream
            The asynchronous stream of the request for iteratively
            fetch the generated tokens asynchronously.
        """
        # Create the unique stream of the request.
        stream = AsyncRequestStream()
        # Report error if the request id already exists.
        if request_id in self._request_tools:
            stream.push(
                RuntimeError(
                    f'The request id "{request_id} already exists. '
                    'Please make sure the request id is unique."'
                )
            )
            return stream

        # Create the request with the given id, input data, generation
        # config and the created callback.
        request = Request(request_id, input_data, generation_cfg)
        # Record the stream in the tracker
        self._request_tools[request_id] = (
            stream,
            TextStreamer(self._tokenizer),
            StopStringHandler(generation_cfg.stop_strs),
        )
        # Push the request to the `requests_to_add` queue for the engine
        # to pull in the next step.
        self._requests_to_add.put_nowait(request)
        # Set `new_request_event`, so that the engine waiting for new
        # requests can be waken up.
        self._new_request_event.set()
        return stream

    def abort_request(self, request_id: str) -> None:
        """Abort a request to the tracker for engine to pull.

        Parameters
        ----------
        request_id : str
            The id of the request to abort.
        """
        self._requests_to_abort.put_nowait(request_id)

    def finish_request(self, request_id: str) -> None:
        """The input request has finished. Remove its stream from the tracker."""
        self._request_tools.pop(request_id, None)

    def get_requests_to_add_and_abort(self) -> Tuple[List[Request], List[str]]:
        """Fetch the requests to add to or abort from the engine.

        Returns
        -------
        requests_to_add : List[Request]
            The requests to add to the engine.

        requests_to_abort : List[str]
            The requests (identified by request id) to abort from the engine.
        """
        requests_to_add = []
        requests_to_abort = []

        while not self._requests_to_add.empty():
            requests_to_add.append(self._requests_to_add.get_nowait())
        while not self._requests_to_abort.empty():
            request_id = self._requests_to_abort.get_nowait()
            requests_to_abort.append(request_id)
            # Remove the request from the tracker.
            self._request_tools.pop(request_id, None)

        # Reset the event since all new requests are fetched in this method.
        self._new_request_event.clear()

        return requests_to_add, requests_to_abort

    def get_request_tools(
        self, request_id: str
    ) -> Tuple[AsyncRequestStream, TextStreamer, StopStringHandler]:
        """Fetch the stream, text streamer and stop handler of the input request.
        The request is expected to have its tools in the tracker.
        """
        return self._request_tools[request_id]

    async def wait_for_new_requests(self):
        """Wait for new requests to add."""
        await self._new_request_event.wait()

    @property
    def is_empty(self) -> bool:
        """Check if there is any request streams existing."""
        return len(self._request_tools) == 0


class AsyncEngine:
    """The asynchronous engine for generate text asynchronously.

    This class wraps a synchronous engine inside and exports the
    asynchronous `generate` method as the main text generation
    interface, which yields the generated tokens. The asynchronous
    engine runs a background loop that drives the synchronous engine.

    Parameters
    ----------
    models : Union[ModelInfo, List[ModelInfo]]
        One or a list of model info (specifying which models to load and
        which device to load to) to launch the engine.

    kv_cache_config : KVCacheConfig
        The configuration of the paged KV cache.
    """

    def __init__(
        self, models: Union[ModelInfo, List[ModelInfo]], kv_cache_config: KVCacheConfig
    ) -> None:
        self._background_engine = Engine(models, kv_cache_config, self._request_stream_callback)
        self.tokenizer = self._background_engine.tokenizer
        self.max_single_sequence_length = self._background_engine.max_single_sequence_length

        self._request_tracker = AsyncRequestTracker(self.tokenizer)
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self._background_loop: Optional[asyncio.Future] = None
        self._terminated: bool = False

    def terminate(self) -> None:
        """Terminate the engine."""
        self._terminated = True
        if not self._background_loop.done():
            self._background_loop_unshielded.cancel()

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
            raise ValueError("The AsyncEngine has terminated.")
        if not self.is_running:
            # Lazily start the background loop so that the engine loop is
            # handled the correct async event loop.
            self.start_background_loop()

        input_data = data.TextData(prompt) if isinstance(prompt, str) else data.TokenData(prompt)
        # Add the request to the tracker and get the stream.
        stream = self._request_tracker.add_request(request_id, input_data, generation_config)
        # Iterate the stream asynchronously and yield the token.
        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:  # pylint: disable=broad-exception-caught
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameter
        ---------
        request_id : str
            The id of the request to abort.
        """
        assert self.is_running
        return self._abort(request_id)

    def start_background_loop(self) -> None:
        """Start the background loop that drives the underlying
        synchronous engine.
        """
        if self.is_running:
            raise RuntimeError("Background loop is already running.")

        # start with event loop that drives the engine
        # use create_task so we do not have to await it
        self._background_loop_unshielded = asyncio.get_event_loop().create_task(
            self._run_engine_loop()
        )
        # when we are done
        self._background_loop_unshielded.add_done_callback(_raise_exception_on_finish)
        self._background_loop = asyncio.shield(self._background_loop_unshielded)

    @property
    def is_running(self) -> bool:
        """Check if the background loop is running."""
        return self._background_loop is not None and not self._background_loop.done()

    async def _run_engine_loop(self) -> None:
        """Internal asynchronous coroutine that drives the engine."""
        while True:
            if self._request_tracker.is_empty:
                # When there is no available requests, wait for new requests.
                await self._request_tracker.wait_for_new_requests()
            self._engine_step()
            await asyncio.sleep(0)

    def _request_stream_callback(
        self, request_id: str, delta_tokens: data.TokenData, finish_reason: Optional[str]
    ) -> None:
        """The request stream callback function for engine to stream back
        the request generation results.

        Parameters
        ----------
        request_id : str
            The id of the request that the function is invoked for.

        delta_tokens : data.TokenData
            The new generated tokens since the last callback invocation
            for the input request.

        finish_reason : Optional[str]
            The finish reason of the request when it is finished,
            of None if the request has not finished yet.

        Note
        ----
        This callback function uses `call_soon_threadsafe` in asyncio to
        schedule the invocation in the event loop, so that the underlying
        callback logic will be executed asynchronously in the future rather
        than right now.
        """
        # Schedule a callback run in the event loop without executing right now.
        asyncio.get_event_loop().call_soon_threadsafe(
            self._request_stream_callback_impl, request_id, delta_tokens, finish_reason
        )

    def _request_stream_callback_impl(
        self, request_id: str, delta_tokens: data.TokenData, finish_reason: Optional[str]
    ) -> None:
        """The underlying implementation of request stream callback."""
        stream, text_streamer, stop_handler = self._request_tracker.get_request_tools(request_id)
        finish_reason = str(finish_reason) if finish_reason is not None else None

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

        # Push new delta text to the stream.
        stream.push((delta_text, len(delta_token_ids), finish_reason))
        if finish_reason is not None:
            stream.finish()
            # Remove the request from the tracker.
            self._request_tracker.finish_request(request_id)

    def _engine_step(self) -> None:
        """Internal implementation of asynchronous engine behavior at each time step."""
        # Fetch the requests to add/abort from the tracker queues.
        # Apply the request add and abortion.
        requests_to_add, requests_to_abort = self._request_tracker.get_requests_to_add_and_abort()
        for request in requests_to_add:
            self._background_engine.add_request(request)
        for request_id in requests_to_abort:
            self._background_engine.abort_request(request_id)

        # Take an engine step.
        self._background_engine.step()

    def _abort(self, request_id: str):
        """Internal implementation of request abortion."""
        # Notify the request tracker about the abortion.
        self._request_tracker.abort_request(request_id)


class AsyncThreadedEngine:
    """
    The asynchronous engine backed by ThreadedEngine: the only
    difference between AsyncEngine and AsyncThreadedEngine is that
    AsyncEngine is backed by a normal Engine, while AsyncThreadedEngine
    is backed by ThreadedEngine.

    See AsyncEngine for full documentation.
    """

    def __init__(
        self, models: Union[ModelInfo, List[ModelInfo]], kv_cache_config: KVCacheConfig
    ) -> None:
        model_args, tokenizer_path, self.max_single_sequence_length = _process_model_args(models)
        self._ffi = _create_tvm_module(
            "mlc.serve.create_threaded_engine",
            ffi_funcs=[
                "add_request",
                "abort_request",
                "run_background_loop",
                "exit_background_loop",
            ],
            creator_args=[
                self.max_single_sequence_length,
                tokenizer_path,
                kv_cache_config.asjson(),
                self._request_stream_callback,
                *model_args,
            ],
        )
        self.tokenizer = Tokenizer(tokenizer_path)

        # The mapping from request ids to request asynchronous stream.
        self._request_tools: Dict[
            str, Tuple[AsyncRequestStream, TextStreamer, StopStringHandler]
        ] = {}
        # Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(
            target=self._ffi["run_background_loop"]
        )
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

    def _request_stream_callback(
        self, request_id: str, delta_tokens: data.TokenData, finish_reason: Optional[str]
    ) -> None:
        """The request stream callback function for engine to stream back
        the request generation results.

        Parameters
        ----------
        request_id : str
            The id of the request that the function is invoked for.

        delta_tokens : data.TokenData
            The new generated tokens since the last callback invocation
            for the input request.

        finish_reason : Optional[str]
            The finish reason of the request when it is finished,
            of None if the request has not finished yet.

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
            self._request_stream_callback_impl, request_id, delta_tokens, finish_reason
        )

    def _request_stream_callback_impl(
        self, request_id: str, delta_tokens: data.TokenData, finish_reason: Optional[str]
    ) -> None:
        """The underlying implementation of request stream callback."""
        tools = self._request_tools.get(request_id, None)
        if tools is None:
            return
        stream, text_streamer, stop_handler = tools
        finish_reason = str(finish_reason) if finish_reason is not None else None

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

        # Push new delta text to the stream.
        stream.push((delta_text, len(delta_token_ids), finish_reason))
        if finish_reason is not None:
            stream.finish()
            self._request_tools.pop(request_id, None)
