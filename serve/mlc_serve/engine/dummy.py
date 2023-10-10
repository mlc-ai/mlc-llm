from threading import Event, Lock
from typing import Dict
from uuid import uuid4

from .types import InferenceStepResult, Request, RequestId, TextGenerationOutput


class DummyInferenceEngine:
    def __init__(self):
        self.queue_lock = Lock()
        self.has_requests = Event()
        self.request_queue: Dict[RequestId, int] = {}

    def add(self, requests: list[Request]) -> list[RequestId]:
        ids = []
        requests_to_add = {}

        for r in requests:
            request_id = str(uuid4())
            ids.append(request_id)
            requests_to_add[request_id] = 5

        with self.queue_lock:
            self.request_queue.update(requests_to_add)
            self.has_requests.set()

        return ids

    def cancel(self, request_id: RequestId):
        """
        Cancel the generation of a request.
        """
        with self.queue_lock:
            del self.request_queue[request_id]
            if not self.request_queue:
                self.has_requests.clear()

    def step(self) -> InferenceStepResult:
        """
        InferenceResult contains the next token for processed results,
        and indicates whether the generation for a request is finished.

        It's up to the InferenceEngine to choose which requests
        to work on, while it should be guaranteed that all requests will be
        processed eventually.

        If the engine has no requests in the queue, `step` will block until there is
        request coming in.
        """
        result = InferenceStepResult(outputs=[], errors=[])

        self.has_requests.wait()

        with self.queue_lock:
            for request_id, n in list(self.request_queue.items()):
                result.outputs.append(
                    TextGenerationOutput(
                        request_id=request_id,
                        delta=" test",
                        finish_reason="length" if n == 1 else None,
                    )
                )
                if n == 1:
                    del self.request_queue[request_id]
                else:
                    self.request_queue[request_id] -= 1

            if not self.request_queue:
                self.has_requests.clear()

        return result
