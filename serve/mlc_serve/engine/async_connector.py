import asyncio
from typing import AsyncIterator, Union

from .types import (
    InferenceEngine,
    InferenceStepResult,
    Request,
    RequestId,
    TextGenerationError,
    TextGenerationOutput,
)

TextGenerationResult = Union[TextGenerationOutput, TextGenerationError]
ResultQueue = asyncio.Queue[TextGenerationResult]


class AsyncEngineConnector:
    def __init__(self, engine: InferenceEngine, engine_wait_timeout=1):
        self.engine = engine
        self.engine_wait_timeout = engine_wait_timeout
        self.engine_loop_task = None
        self.engine_loop_exception = None
        self.shutdown_event = asyncio.Event()
        self.request_queues = dict[RequestId, ResultQueue]()

    async def start(self):
        """
        Needs to be called in the thread with event loop
        """
        if self.engine_loop_task is not None:
            return

        loop = asyncio.get_running_loop()
        should_stop_inference = False

        def inference_loop():
            while True:
                self.engine.wait_for_request(timeout_seconds=self.engine_wait_timeout)
                if should_stop_inference:
                    return
                result = self.engine.step()
                asyncio.run_coroutine_threadsafe(self._dispatch_result(result), loop)

        async def wait():
            try:
                await asyncio.to_thread(inference_loop)
            except asyncio.CancelledError:
                nonlocal should_stop_inference
                should_stop_inference = True
            except Exception as e:
                # TODO: Log
                self.engine_loop_exception = e
                raise
            finally:
                self.shutdown_event.set()

        self.engine_loop_task = asyncio.create_task(wait())

    async def stop(self):
        # TODO: make it able to restart?
        self.engine_loop_task.cancel()

    async def generate(self, request: Request) -> AsyncIterator[TextGenerationOutput]:
        try:
            queue = await self._add_request(request)
            while True:
                result = await self._get_queue_item_until_stopped(queue)
                if isinstance(result, TextGenerationOutput):
                    yield result
                    if result.finish_reason is not None:
                        return
                elif isinstance(result, TextGenerationError):
                    # TODO: rethink about the error handling here
                    raise result
                else:
                    raise RuntimeError(f"Unknown result type {type(result)}")
        except asyncio.CancelledError:
            asyncio.to_thread(self.engine.cancel, request.request_id)
        finally:
            self.request_queues.pop(request.request_id, None)

    async def _get_queue_item_until_stopped(
        self, queue: ResultQueue
    ) -> TextGenerationResult:
        get_queue_task = asyncio.create_task(queue.get())
        wait_shutdown_task = asyncio.create_task(self.shutdown_event.wait())

        await asyncio.wait(
            (get_queue_task, wait_shutdown_task),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if wait_shutdown_task.done():
            if self.engine_loop_exception is not None:
                raise RuntimeError(
                    f"Engine loop raised exception: {self.engine_loop_exception}"
                )
            else:
                raise RuntimeError("Engine stopped")

        return get_queue_task.result()

    async def _add_request(self, request: Request) -> ResultQueue:
        if self.engine_loop_task is None:
            raise RuntimeError(
                "Inference loop is not running. Call AsyncEngineConnector.start first."
            )
        if request.request_id in self.request_queues:
            raise RuntimeError(f"Duplicate request id: {request.request_id}")

        queue = asyncio.Queue()
        self.request_queues[request.request_id] = queue

        await asyncio.to_thread(self.engine.add, [request])

        return queue

    async def _dispatch_result(self, result: InferenceStepResult):
        coroutines = []
        for item in result.outputs + result.errors:
            request_id = item.request_id
            if request_id not in self.request_queues:
                # TODO: handle error here
                continue

            queue = self.request_queues[request_id]
            coroutines.append(queue.put(item))

        await asyncio.gather(*coroutines)
