from threading import Condition, Lock
from typing import Dict
from uuid import uuid4

from .types import InferenceStepResult, Request, RequestId, TextGenerationOutput


class DummyInferenceEngine:
    def __init__(self):
        self.queue_lock = Lock()
        self.has_new_requests = Condition(self.queue_lock)
        self.request_queue: Dict[RequestId, int] = {}

    def add(self, requests: list[Request]) -> list[RequestId]:
        ids = []
        requests_to_add = {}

        for r in requests:
            ids.append(r.request_id)
            if r.stopping_criteria.max_tokens is not None:
                requests_to_add[r.request_id] = r.stopping_criteria.max_tokens
            else:
                requests_to_add[r.request_id] = 5

        with self.queue_lock:
            self.request_queue.update(requests_to_add)
            self.has_new_requests.notify_all()

        return ids

    def cancel(self, request_id: RequestId):
        """
        Cancel the generation of a request.
        """
        with self.queue_lock:
            del self.request_queue[request_id]
            if not self.request_queue:
                self.has_requests.clear()

    def wait_for_request(self, timeout_seconds=None):
        with self.queue_lock:
            self.has_new_requests.wait_for(
                lambda: len(self.request_queue) > 0, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        result = InferenceStepResult(outputs=[], errors=[])

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

        return result
