import asyncio
import time
import random
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
from .types import (InferenceEngine, Request, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig,
                    RequestOutput, CompletionOutput)

from .arg_utils import EngineArgs

from .sampling_params import SamplingParams
from .mlc_llm_b1_inference_engine import MlcLLMb1Engine
# logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task,
                               request_tracker: "RequestTracker") -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()

    def __contains__(self, item):
        return item in self._request_streams

    def propagate_exception(self, exc: Exception) -> None:
        """Propagate an exception to all request streams."""
        for stream in self._request_streams.values():
            stream.put(exc)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            # if verbose:
            #     logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str,
                    model_name: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            "model_name": model_name,
            **engine_add_request_kwargs
        }))
        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        # if verbose:
        #     logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests


class _AsyncLLMEngine:
    """Extension of LLMEngine to add async methods."""
    requests_output: Dict[str, RequestOutput] = {}
    engine: InferenceEngine = None

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.engine = MlcLLMb1Engine(model_config, cache_config, parallel_config, scheduler_config)

    def add(self, requests: list[Request]) -> list[str]:
        return self.engine.add(requests)

    def cancel(self, request_id: str):
        self.engine.cancel(request_id)

    async def step_async(self) -> List[RequestOutput]:
        output = self.engine.step()
        outputs = []
        for a in output.outputs:
            if self.requests_output.get(a.request_id) == None:
                ro = RequestOutput
                ro.request_id = a.request_id
                ro.outputs = [CompletionOutput]
                ro.outputs[0].text = ""
                ro.outputs[0].index = 0
                self.requests_output[ro.request_id] = ro
            ro = self.requests_output.get(a.request_id)
            ro.outputs[0].text += a.delta
            if hasattr(a,"finish_reason") and a.finish_reason:
                ro.finished = True
                ro.outputs[0].finish_reason = a.finish_reason
            else:
                ro.finished = False
            outputs.append(ro)
        return outputs


class ServingLayer:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for LLMEngine.
    """

    def __init__(self,
                 *args,
                #  log_requests: bool = True,
                #  max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        # self.log_requests = log_requests
        # self.max_log_len = max_log_len
        self.engine = _AsyncLLMEngine(*args,**kwargs)

        self.request_tracker: RequestTracker = RequestTracker()
        self.background_loop = None
        self.start_engine_loop = start_engine_loop

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self.background_loop = asyncio.get_event_loop().create_task(
            self.run_engine_loop())
        self.background_loop.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self.request_tracker))

    async def engine_step(self):
        """Kick the engine to process the waiting requests."""

        new_requests, finished_requests = (
            self.request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            r = Request
            r.request_id = new_request["request_id"]
            r.model_name = new_request["model_name"]
            r.prompt = new_request["prompt"]
            r.sampling_params = new_request["sampling_params"]
            self.engine.add([r])

        if finished_requests:
            await self._engine_abort(finished_requests)

        request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self.request_tracker.process_request_output(request_output)
                                #, verbose=self.log_requests)

    async def _engine_abort(self, request_ids: Iterable[str]):
        for r in request_ids:
            self.engine.cancel(r)

    async def run_engine_loop(self):
        while True:
            await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        model_name: str,
        prompt: Optional[str],
        sampling_params: SamplingParams
    ) -> AsyncStream:
        # if self.log_requests:
        #     shortened_prompt = prompt
        #     shortened_token_ids = prompt_token_ids
        #     if self.max_log_len is not None:
        #         if shortened_prompt is not None:
        #             shortened_prompt = shortened_prompt[:self.max_log_len]
        #         if shortened_token_ids is not None:
        #             shortened_token_ids = shortened_token_ids[:self.
        #                                                       max_log_len]
        #     logger.info(f"Received request {request_id}: "
        #                 f"prompt: {shortened_prompt!r}, "
        #                 f"sampling params: {sampling_params}, "
        #                 f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self.request_tracker.add_request(
            request_id,
            model_name,
            prompt=prompt,
            sampling_params=sampling_params)

        return stream

    async def generate(
            self,
            prompt: str,
            model_name: str,
            sampling_params: SamplingParams,
            request_id: str) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        try:
            stream = await self.add_request(request_id,
                                            model_name,
                                            prompt,
                                            sampling_params)

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self.request_tracker.abort_request(request_id)
                                           # ,verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: EngineArgs,
                         start_engine_loop: bool = True) -> "ServingLayer":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        # Create the async LLM engine.
        engine = cls(*engine_configs,
                     start_engine_loop=start_engine_loop)
        return engine
