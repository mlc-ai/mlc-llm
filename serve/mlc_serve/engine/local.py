"""
A implementation of InferenceEngine that executes in the current process.
"""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Lock
from uuid import uuid4

from .base import (
    DebugOptions,
    FinishReason,
    InferenceEngine,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    SamplingParams,
    SequenceOutput,
    StoppingCriteria,
)
from .model_module import DecodeRequest, ModelModule, PrefillRequest, SequenceId


@dataclass
class RequestState:
    request_id: RequestId
    token_ids: list[int]
    output_text: str
    prompt_len: int
    next_start_position: int
    sampling_params: SamplingParams
    stopping_criteria: StoppingCriteria
    debug_options: DebugOptions
    is_ended: bool = False


class LocalProcessInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model_module: ModelModule,
        max_batched_tokens: int = 2560,
        min_decode_steps: int = 32,
    ):
        self.text_generator = model_module.text_generator
        self.tokenizer = model_module.tokenizer
        self.conversation_template = model_module.conversation_template
        self.cache_manager = model_module.cache_manager

        self.max_batched_tokens = max_batched_tokens
        self.min_decode_steps = min(
            self.cache_manager.get_kv_cache_size(), min_decode_steps
        )

        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)
        self.requests_to_be_cancelled = set[RequestId]()

        self.current_batch = dict[RequestId, RequestState]()

    def add(self, requests: list[Request]):
        if not requests:
            return []

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            if req.num_sequences > 1:
                raise RuntimeError("num_sequences > 1 is not supported for now")
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

    def wait_for_request(self, timeout_seconds=None) -> bool:
        with self.queue_lock:
            self.has_new_requests.wait_for(
                self._has_request_to_process, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        outputs = list[RequestOutput]()
        result = InferenceStepResult(outputs=outputs)

        # TODO: consolidate into a single function
        for state in list(self.current_batch.values()):
            finish_reason = None
            if state.is_ended:
                finish_reason = FinishReason.Stop
            if self._should_stop_by_length(state):
                finish_reason = FinishReason.Length

            if finish_reason is not None:
                outputs.append(
                    RequestOutput(
                        state.request_id,
                        [SequenceOutput(0, finish_reason=finish_reason)],
                    )
                )
                self.current_batch.pop(state.request_id)
                self.cache_manager.free(SequenceId(state.request_id, 0))

        previous_requests_to_be_cancelled = set(self.requests_to_be_cancelled)
        self._adjust_batch()

        for request_id in previous_requests_to_be_cancelled:
            if request_id not in self.requests_to_be_cancelled:
                outputs.append(
                    RequestOutput(
                        request_id=request_id,
                        sequences=[
                            # TODO: support multi-sequence
                            SequenceOutput(0, finish_reason=FinishReason.Cancelled)
                        ],
                    )
                )

        if not self.current_batch:
            return result

        requests = []
        for state in self.current_batch.values():
            if state.next_start_position == 0:
                requests.append(
                    PrefillRequest(
                        request_id=state.request_id,
                        token_ids=state.token_ids,
                        num_sequence=1,
                        sampling_params=state.sampling_params,
                    )
                )
            else:
                seq_id = SequenceId(state.request_id, 0)
                requests.append(
                    DecodeRequest(
                        sequence_id=seq_id,
                        token_ids=state.token_ids,
                        sampling_params=state.sampling_params,
                    )
                )
                self.cache_manager.extend(
                    seq_id, len(state.token_ids) - state.next_start_position
                )

        results = self.text_generator.generate(requests, self.cache_manager.get_cache())

        for res in results:
            # For now we only support single sequence per request
            request_id = res.sequence_id.request_id
            if res.error is not None:
                del self.current_batch[request_id]
                self.cache_manager.free(res.sequence_id)
                outputs.append(
                    RequestOutput(res.sequence_id.request_id, error=res.error)
                )
                continue

            state = self.current_batch[request_id]
            state.next_start_position = len(state.token_ids)
            new_token_ids = res.generated_tokens
            for i, token_id in enumerate(new_token_ids):
                if (
                    token_id == self.tokenizer.eos_token_id
                    and not state.debug_options.ignore_eos
                ):
                    new_token_ids = new_token_ids[:i]
                    state.is_ended = True
            state.token_ids.extend(new_token_ids)

            delta = self._decode_last_output(state)
            state.output_text += delta

            outputs.append(
                RequestOutput(request_id, sequences=[SequenceOutput(0, delta=delta)])
            )

        return result

    def _adjust_batch(self):
        with self.queue_lock:
            for request_id in list(self.requests_to_be_cancelled):
                if request_id in self.current_batch:
                    state = self.current_batch.pop(request_id)
                    self.cache_manager.free(state.request_id)
                    self.requests_to_be_cancelled.remove(request_id)

            while self.cache_manager.get_max_new_tokens() < 1:
                request_to_remove = min(
                    self.current_batch.values(), key=lambda s: len(s.token_ids)
                )
                del self.current_batch[request_to_remove.request_id]
                self.cache_manager.free(SequenceId(request_to_remove.request_id, 0))
                self.queue.appendleft(request_to_remove)

            self._discard_cancelled_requests_from_queue()

            num_batched_tokens = sum(
                len(state.token_ids) for state in self.current_batch.values()
            )
            while self.queue:
                if self.cache_manager.get_max_new_tokens() < self.min_decode_steps:
                    # stop adding request if there isn't enough space to do a certain steps of decoding.
                    break
                state = self.queue[0]
                num_tokens = len(state.token_ids)
                num_batched_tokens += num_tokens
                if num_batched_tokens > self.max_batched_tokens > 0:
                    break
                if self.cache_manager.get_free_space() <= 1.5 * num_tokens:
                    break

                self.queue.popleft()
                self.cache_manager.allocate(state.request_id, num_tokens)
                self.current_batch[state.request_id] = state

                self._discard_cancelled_requests_from_queue()

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
        if request.debug_options.prompt is not None:
            prompt = request.debug_options.prompt
        else:
            prompt = self.conversation_template.apply(request.messages)

        prompt_tokens = self.tokenizer.encode(prompt)

        return RequestState(
            request_id=request.request_id,
            token_ids=prompt_tokens,
            prompt_len=len(prompt_tokens),
            next_start_position=0,
            sampling_params=request.sampling_params,
            stopping_criteria=request.stopping_criteria,
            debug_options=request.debug_options,
            output_text="",
        )

    def _decode_last_output(self, state: RequestState) -> str:
        if len(state.output_text):
            prefix_idx = max(0, state.next_start_position - 6)
        else:
            prefix_idx = state.next_start_position

        if prefix_idx == 0:
            return self.tokenizer.decode(state.token_ids)

        prefix = self.tokenizer.decode(
            state.token_ids[prefix_idx : state.next_start_position]
        )
        full = self.tokenizer.decode(state.token_ids[prefix_idx:])

        return full[len(prefix) :]

    def _should_stop_by_length(self, state: RequestState) -> bool:
        # TODO: put to config
        max_tokens = 4096
        if state.stopping_criteria.max_tokens is not None:
            max_tokens = min(max_tokens, state.stopping_criteria.max_tokens)

        return len(state.token_ids) - state.prompt_len >= max_tokens
