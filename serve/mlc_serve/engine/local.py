"""
A implementation of InferenceEngine that executes in the current process.
"""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Lock
from uuid import uuid4

from .types import (
    InferenceEngine,
    InferenceStepResult,
    ModelExecutor,
    Request,
    RequestId,
    SamplingParams,
    SequenceGenerationRequest,
    SequenceGenerationResponse,
    StoppingCriteria,
    TextGenerationError,
    TextGenerationOutput,
    Tokenizer,
)


@dataclass
class RequestState:
    request_id: RequestId
    token_ids: list[int]
    output_text: str
    prompt_len: int
    next_start_position: int
    sampling_params: SamplingParams
    stopping_criteria: StoppingCriteria


class LocalProcessInferenceEngine(InferenceEngine):
    def __init__(self, executor: ModelExecutor, tokenizer: Tokenizer):
        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)
        self.requests_to_be_cancelled = set[RequestId]()

        self.current_batch = dict[RequestId, RequestState]()

        self.executor = executor
        self.tokenizer = tokenizer

    def add(self, requests: list[Request]):
        if not requests:
            return []

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            state = self._get_new_request_state(req)
            new_request_states.append(state)

        with self.queue_lock:
            self.queue.extend(new_request_states)
            self.has_new_requests.notify_all()

        return [s.request_id for s in new_request_states]

    def cancel(self, request_id: RequestId):
        with self.queue_lock:
            # TODO: consider iterating throught the queue to find if request id exist
            # Otherwise cancel a request that's already finished will leave request_id
            # in the `requests_to_be_cancelled` set forever.
            self.requests_to_be_cancelled.add(request_id)

    def wait_for_request(self, timeout_seconds=None):
        with self.queue_lock:
            self.has_new_requests.wait_for(
                self._has_request_to_process, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        outputs = list[TextGenerationOutput]()
        errors = list[TextGenerationError]()
        result = InferenceStepResult(outputs=outputs, errors=errors)

        previous_requests_to_be_cancelled = set(self.requests_to_be_cancelled)
        self._adjust_batch()

        for request_id in previous_requests_to_be_cancelled:
            if request_id not in self.requests_to_be_cancelled:
                outputs.append(
                    TextGenerationOutput(
                        request_id=request_id,
                        delta="",
                        finish_reason="cancelled",
                    )
                )

        if not self.current_batch:
            return result

        requests = [
            SequenceGenerationRequest(
                request_id=state.request_id,
                token_ids=state.token_ids,
                start_position=state.next_start_position,
                sampling_params=state.sampling_params,
            )
            for state in self.current_batch.values()
        ]

        for req in requests:
            if req.start_position > 0:
                self.executor.extend(
                    req.request_id, len(req.token_ids) - req.start_position
                )
        responses = self.executor.generate(requests)

        for res in responses:
            if res.error is not None:
                errors.append(
                    TextGenerationError(res.request_id, "GenerationError", res.error)
                )
                del self.current_batch[res.request_id]
                continue

            state = self.current_batch[res.request_id]
            state.next_start_position = len(state.token_ids)
            new_token_ids = res.token_ids
            is_ended = False
            for i, token_id in enumerate(new_token_ids):
                if token_id == self.tokenizer.eos_token_id:
                    new_token_ids = new_token_ids[:i]
                    is_ended = True
            state.token_ids.extend(new_token_ids)

            delta = self._decode_last_output(state)
            state.output_text += delta

            output = TextGenerationOutput(res.request_id, delta)
            if is_ended:
                output.finish_reason = "stop"
            if self._should_stop_by_length(state):
                output.finish_reason = "length"

            if output.finish_reason is not None:
                self.current_batch.pop(state.request_id)
                self.executor.free(state.request_id)

            outputs.append(output)

        return result

    def _adjust_batch(self):
        with self.queue_lock:
            for request_id in list(self.requests_to_be_cancelled):
                if request_id in self.current_batch:
                    state = self.current_batch.pop(request_id)
                    self.executor.free(state.request_id)
                    self.requests_to_be_cancelled.remove(request_id)

            while self.executor.get_max_new_tokens() < 1:
                request_to_remove = min(
                    self.current_batch.values(), key=lambda s: len(s.token_ids)
                )
                del self.current_batch[request_to_remove.request_id]
                self.executor.free(request_to_remove.request_id)
                self.queue.appendleft(request_to_remove)

            self._discard_cancelled_requests_from_queue()

            if not self._should_process_new_request():
                return

            # TODO: make this 15 into config
            # and consider the max cache size of the executor
            while self.queue and self.executor.get_max_new_tokens() > 15:
                state = self.queue[0]
                num_tokens = len(state.token_ids)
                if self.executor.get_free_space() <= 1.5 * num_tokens:
                    break

                self.queue.popleft()
                self.executor.allocate(state.request_id, num_tokens)
                self.current_batch[state.request_id] = state

                self._discard_cancelled_requests_from_queue()

    def _should_process_new_request(self):
        return self.executor.get_free_space() * 1.6 > self.executor.get_kv_cache_size()

    def _has_request_to_process(self) -> bool:
        return self.queue or self.current_batch

    def _discard_cancelled_requests_from_queue(self):
        """
        Requires the self.queue_lock to be held before calling this function.
        """
        while self.queue and self.queue[0].request_id in self.requests_to_be_cancelled:
            state = self.queue.popleft()
            self.requests_to_be_cancelled.remove(state.request_id)

    def _get_new_request_state(self, request: Request) -> RequestState:
        prompt_tokens = self.tokenizer.encode(request.prompt)

        return RequestState(
            request_id=request.request_id,
            token_ids=prompt_tokens,
            prompt_len=len(prompt_tokens),
            next_start_position=0,
            sampling_params=request.sampling_params,
            stopping_criteria=request.stopping_criteria,
            output_text="",
        )

    def _decode_last_output(self, state: RequestState) -> str:
        prefix_idx = max(0, state.next_start_position - 6)
        if prefix_idx == 0:
            return self.tokenizer.decode(state.token_ids)

        prefix = self.tokenizer.decode(
            state.token_ids[prefix_idx : state.next_start_position],
            skip_special_tokens=True,
        )
        full = self.tokenizer.decode(
            state.token_ids[prefix_idx:], skip_special_tokens=True
        )

        return full[len(prefix) :]

    def _should_stop_by_length(self, state: RequestState) -> bool:
        # TODO: put to config
        max_tokens = 4096
        if state.stopping_criteria.max_tokens is not None:
            max_tokens = min(max_tokens, state.stopping_criteria.max_tokens)

        return len(state.token_ids) - state.prompt_len >= max_tokens
