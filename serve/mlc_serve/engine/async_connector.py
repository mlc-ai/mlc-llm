import asyncio
import structlog
from typing import AsyncIterator, Any
from concurrent.futures import ThreadPoolExecutor

from .base import (
    InferenceEngine,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    ScopedInferenceEngine,
)

from .error import EngineException, TextGenerationError

LOG = structlog.stdlib.get_logger(__name__)

ResultQueue = asyncio.Queue[RequestOutput]

class AsyncEngineConnector:
    def __init__(self, engine: InferenceEngine, engine_wait_timeout=1):
        self.engine = engine
        self.engine_wait_timeout = engine_wait_timeout
        self.engine_loop_task = None
        self.engine_loop_exception = None
        self.shutdown_event = asyncio.Event()
        self.result_queues = dict[RequestId, ResultQueue]()

    async def start(self):
        """
        Needs to be called in the thread with event loop
        """
        LOG.info("Starting AsyncEngineConnector.", engine_wait_timeout=self.engine_wait_timeout)
        if self.engine_loop_task is not None:
            return

        loop = asyncio.get_running_loop()
        should_stop_inference = False

        if isinstance(self.engine, ScopedInferenceEngine):
            await asyncio.to_thread(self.engine.start)

        def inference_loop() -> None:
            while True:
                self.engine.wait_for_request(timeout_seconds=self.engine_wait_timeout)
                if should_stop_inference:
                    return
                result = self.engine.step()
                # TODO: Use a queue here to guarantee the ordering,
                # that is, the result of next step must be dispatched after the
                # result of current step.
                asyncio.run_coroutine_threadsafe(self._dispatch_result(result), loop)

        async def wait():
            try:
                await asyncio.to_thread(inference_loop)
            except asyncio.CancelledError:
                nonlocal should_stop_inference
                should_stop_inference = True
            except Exception as e:
                LOG.exception("Error occurred while running the inference loop.")
                self.engine_loop_exception = e
                raise
            finally:
                self.shutdown_event.set()

        self.engine_loop_task = asyncio.create_task(wait())

    async def stop(self):
        self.engine_loop_task.cancel()
        await self.engine_loop_task
        if isinstance(self.engine, ScopedInferenceEngine):
            await asyncio.to_thread(self.engine.stop)

    async def generate(self, request: Request) -> AsyncIterator[RequestOutput]:
        try:
            queue = await self._add_request(request)
            while True:
                # TODO(jknight): Should make sure we are catching cancellations 
                # correctly inside this _get_queue...(...) awaitable object too
                output = await self._get_queue_item_until_stopped(queue)
                if output.error is not None:
                    raise TextGenerationError(output.error)
                yield output
                if output.is_finished:
                    return
        except asyncio.CancelledError:
            LOG.info("AsyncEngineConnector.generate iterator cancelled.", request_id=request.request_id)
            # Running this sync because `await` inside of cancellation events is problematic
            self.engine.cancel(request.request_id)
            LOG.info("AsyncEngineConnector.generate request sucessfully cancelled.", request_id=request.request_id)
            # Always re-raise CancellationErrors unless you know what you're doing.
            raise
        finally:
            LOG.info("AsyncEngineConnector.generate removing request from result queue.", request_id=request.request_id)
            self.result_queues.pop(request.request_id, None)

    async def _get_queue_item_until_stopped(self, queue: ResultQueue) -> RequestOutput:
        try:
            get_queue_task = asyncio.create_task(queue.get(), name="get_queue_task")
            wait_shutdown_task = asyncio.create_task(self.shutdown_event.wait(), name="wait_shutdown_task")

            await asyncio.wait(
                (get_queue_task, wait_shutdown_task),
                return_when=asyncio.FIRST_COMPLETED,
            )

            if wait_shutdown_task.done():
                if self.engine_loop_exception is not None:
                    raise EngineException("raised while handling previous engine loop exception") from self.engine_loop_exception
                else:
                    raise EngineException("stopped with no exception")

            wait_shutdown_task.cancel()
            return get_queue_task.result()
        except asyncio.CancelledError:
            wait_shutdown_task.cancel()
            get_queue_task.cancel()
            raise

    async def _add_request(self, request: Request) -> ResultQueue:
        if self.engine_loop_task is None:
            raise RuntimeError(
                "Inference loop is not running. Call AsyncEngineConnector.start first."
            )

        if request.request_id in self.result_queues:
            raise RuntimeError(f"Duplicate request id: {request.request_id}")

        queue = asyncio.Queue()
        self.result_queues[request.request_id] = queue

        await asyncio.to_thread(self.engine.add, [request])

        return queue

    async def _dispatch_result(self, result: InferenceStepResult) -> None:
        coroutines = []
        for item in result.outputs:
            request_id = item.request_id
            if request_id not in self.result_queues:
                LOG.warn(
                    f"Unknown request id when dispatching result: {request_id}"
                )
                continue

            queue = self.result_queues[request_id]
            coroutines.append(queue.put(item))

        await asyncio.gather(*coroutines)
