"""The MLC LLM Asynchronous Serving Engine.
Acknowledgment: Part of the code was adapted from the vLLM project.
"""
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from . import data
from .config import GenerationConfig, KVCacheConfig
from .engine import Engine, ModelInfo
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
    tokens to the stream, and the method `finish` for engine to mark
    the finish of generation.

    The stream implements `__aiter__` and `__anext__`, which the engine
    can use to iterates all the generated tokens in order asynchronously.
    """

    # The asynchronous queue to hold generated tokens or exceptions.
    _queue: asyncio.Queue[Union[int, Exception]]
    # The finish flag.
    _finished: bool

    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._finished = False

    def push(self, token_or_exception: Union[int, Exception]) -> None:
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
        self._queue.put_nowait(token_or_exception)

    def finish(self) -> None:
        """Mark the finish of the generation in the stream."""
        self._queue.put_nowait(StopIteration())
        self._finished = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
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
    _request_streams: Dict[str, AsyncRequestStream]
    # The queue of requests to add into the engine.
    _requests_to_add: asyncio.Queue[Request]
    # The queue of requests (identified by request id) to abort from the engine.
    _requests_to_abort: asyncio.Queue[str]
    # The event of "there are new requests to add", used to avoid engine
    # from cyclically pull and waiting for new requests.
    _new_request_event: asyncio.Event

    def __init__(self) -> None:
        self._request_streams = {}
        self._requests_to_add = asyncio.Queue()
        self._requests_to_abort = asyncio.Queue()
        self._new_request_event = asyncio.Event()

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
        if request_id in self._request_streams:
            stream.push(
                RuntimeError(
                    f'The request id "{request_id} already exists. '
                    'Please make sure the request id is unique."'
                )
            )
            return stream

        # Create the callback function of the request, which pushes new
        # generated tokens to the stream.
        def stream_callback(rid: str, delta_tokens: data.TokenData, finished: bool) -> None:
            if rid != request_id:
                stream.push(
                    RuntimeError(
                        "Internal error: request id mismatch. "
                        f'Expect "{request_id}" but got "{rid}".'
                    )
                )
            for token_id in delta_tokens.token_ids:
                stream.push(token_id)
            if finished:
                stream.finish()
                # Remove the request from the tracker.
                self._request_streams.pop(request_id)

        # Create the request with the given id, input data, generation
        # config and the created callback.
        request = Request(request_id, input_data, generation_cfg, stream_callback)
        # Record the stream in the tracker
        self._request_streams[request_id] = stream
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
            self._request_streams.pop(request_id)

        # Reset the event since all new requests are fetched in this method.
        self._new_request_event.clear()

        return requests_to_add, requests_to_abort

    async def wait_for_new_requests(self):
        """Wait for new requests to add."""
        await self._new_request_event.wait()

    @property
    def is_empty(self) -> bool:
        """Check if there is any request streams existing."""
        return len(self._request_streams) == 0


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

    engine: Engine

    def __init__(
        self, models: Union[ModelInfo, List[ModelInfo]], kv_cache_config: KVCacheConfig
    ) -> None:
        self.engine = Engine(models, kv_cache_config)
        self._request_tracker = AsyncRequestTracker()
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self._background_loop: Optional[asyncio.Future] = None

    async def generate(
        self, prompt: Union[str, List[int]], generation_config: GenerationConfig, request_id: str
    ) -> AsyncGenerator[int, Any]:
        """Asynchronous text generation interface.
        The method is a coroutine that streams the generated tokens to
        the caller side via yield.

        Parameters
        ----------
        prompt : Union[str, List[int]]
            The input prompt in forms of text string or a list of token ids.

        generation_config : GenerationConfig
            The generation config of the request.

        request_id : str
            The unique identifier (in string) or this generation request.
        """
        input_data = data.TextData(prompt) if isinstance(prompt, str) else data.TokenData(prompt)
        # Add the request to the tracker and get the stream.
        stream = self._request_tracker.add_request(request_id, input_data, generation_config)
        # Iterate the stream asynchronously and yield the token.
        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError):  # pylint: disable=broad-exception-caught
            self._abort(request_id)

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

    def _engine_step(self) -> None:
        """Internal implementation of asynchronous engine behavior at each time step."""
        # Fetch the requests to add/abort from the tracker queues.
        # Apply the request add and abortion.
        requests_to_add, requests_to_abort = self._request_tracker.get_requests_to_add_and_abort()
        for request in requests_to_add:
            self.engine.add_request(request)
        for request_id in requests_to_abort:
            self.engine.abort_request(request_id)

        # Take an engine step.
        self.engine.step()

    def _abort(self, request_id: str):
        """Internal implementation of request abortion."""
        # Notify the request tracker about the abortion.
        self._request_tracker.abort_request(request_id)
