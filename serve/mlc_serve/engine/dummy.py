from threading import Condition, Lock
from typing import Dict

from .base import (
    FinishReason,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    SequenceOutput,
)


class DummyInferenceEngine:
    def __init__(self):
        self.queue_lock = Lock()
        self.has_new_requests = Condition(self.queue_lock)
        self.request_queue: Dict[RequestId, int] = {}

    def add(self, requests: list[Request]):
        ids = []
        requests_to_add = {}

        for req in requests:
            assert req.num_sequences == 1, "Only one generated sequence allowed for now"
            ids.append(req.request_id)
            if req.stopping_criteria.max_tokens is not None:
                requests_to_add[req.request_id] = req.stopping_criteria.max_tokens
            else:
                requests_to_add[req.request_id] = 5

        with self.queue_lock:
            self.request_queue.update(requests_to_add)
            self.has_new_requests.notify_all()

    def cancel(self, request_id: RequestId):
        """
        Cancel the generation of a request.
        """
        with self.queue_lock:
            del self.request_queue[request_id]

    def wait_for_request(self, timeout_seconds=None):
        with self.queue_lock:
            self.has_new_requests.wait_for(
                lambda: len(self.request_queue) > 0, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        result = InferenceStepResult(outputs=[])

        with self.queue_lock:
            for request_id, remaining_tokens in list(self.request_queue.items()):
                result.outputs.append(
                    RequestOutput(
                        request_id=request_id,
                        sequences=[
                            SequenceOutput(
                                index=0,
                                delta=" test" if remaining_tokens > 0 else None,
                                finish_reason=FinishReason.Length
                                if remaining_tokens == 0
                                else None,
                            )
                        ],
                    )
                )
                if remaining_tokens == 0:
                    del self.request_queue[request_id]
                else:
                    self.request_queue[request_id] -= 1

        return result
