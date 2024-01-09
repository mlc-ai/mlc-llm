"""
A implementation of InferenceEngine that executes in the current process.
"""

import logging
from collections import defaultdict
from typing import Set, Dict

from .base import (
    FinishReason,
    InferenceEngine,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    SequenceOutput,
    ValidationError,
)
from .engine_common import (
    should_stop_by_length,
    get_new_request_state,
    get_requests_to_process,
    update_sequence,
    EngineBase,
)
from .model_module import (
    ModelModule,
)

logger = logging.getLogger(__name__)


class SynchronousInferenceEngine(InferenceEngine, EngineBase):
    """
    A implementation of InferenceEngine that does inference synchronously in the current thread
    when `step` is called.
    """

    # ID and num_sequences for canceled requests
    requests_to_be_cancelled: Set[RequestId]
    num_sequences_per_requests: Dict[RequestId, int]

    def __init__(
        self,
        model_module: ModelModule,
    ):
        EngineBase.__init__(self, model_module)
        self.requests_to_be_cancelled = set[RequestId]()
        self.num_sequences_per_requests = dict[RequestId, int]()

    def add(self, requests: list[Request]):
        if not requests:
            return []

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            # wrap the stop sequence with list if necessary
            if req.stopping_criteria.stop_sequences:
                if isinstance(req.stopping_criteria.stop_sequences, str):
                    req.stopping_criteria.stop_sequences = [
                        req.stopping_criteria.stop_sequences
                    ]
                assert isinstance(req.stopping_criteria.stop_sequences, list)

            state = get_new_request_state(
                req, self.conversation_template, self.tokenizer
            )
            new_request_states.append(state)
            self.num_sequences_per_requests[state.request_id] = state.num_sequences

            if state.validation_err is not None or self.check_prompt_too_long(
                state.prompt_len, state.num_sequences
            ):
                self.cancel(req.request_id)
                if state.validation_err is None:
                    state.validation_err = ValidationError(
                        "The prompt is too long for the given set of engine parameters."
                    )

        with self.queue_lock:
            self.queue.extend(new_request_states)
            self.has_new_requests.notify_all()

    def cancel(self, request_id: RequestId):
        with self.queue_lock:
            # TODO: consider iterating throught the queue to find if request id exist
            # Otherwise cancel a request that's already finished will leave request_id
            # in the `requests_to_be_cancelled` set forever.
            self.requests_to_be_cancelled.add(request_id)

    def wait_for_request(self, timeout_seconds=None) -> bool:
        with self.queue_lock:
            return self.has_new_requests.wait_for(
                self.has_pending_requests, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        logger.debug("Starting new inference step.")

        outputs = list[RequestOutput]()

        previous_requests_to_be_cancelled = set(self.requests_to_be_cancelled)
        self._adjust_batch()

        if not self.current_batch:
            if len(self.queue) > 0:
                logger.warning(
                    f"The engine has {len(self.queue)} requests to be processed in the"
                    " queue, but none of them were added to the current batch during"
                    " the execution of SyncEngine._adjust_batch"
                )

        for request_id in previous_requests_to_be_cancelled:
            if request_id not in self.requests_to_be_cancelled:
                outputs.append(
                    RequestOutput(
                        request_id=request_id,
                        sequences=[
                            SequenceOutput(i, finish_reason=FinishReason.Cancelled)
                            for i in range(self.num_sequences_per_requests[request_id])
                        ],
                    )
                )
                del self.num_sequences_per_requests[request_id]

        for request_id in list(self.requests_to_be_cancelled):
            if request_id not in previous_requests_to_be_cancelled:
                # This is for requests canceled during _adjust_batch()
                outputs.append(
                    RequestOutput(
                        request_id=request_id,
                        sequences=[
                            SequenceOutput(i, finish_reason=FinishReason.Cancelled)
                            for i in range(self.num_sequences_per_requests[request_id])
                        ],
                    )
                )
                self.requests_to_be_cancelled.remove(request_id)
                del self.num_sequences_per_requests[request_id]

        if not self.current_batch:
            return InferenceStepResult(outputs)

        requests, _, _ = get_requests_to_process(
            list(self.current_batch.values()), self.cache_manager
        )
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        logger.debug("Finished text generation.")

        valid_results = []
        failed_requests = set()

        for res in results:
            request_id = res.sequence_id.request_id
            # Report an error for a request if any of its generation sequences fails.
            if res.error is not None and request_id not in failed_requests:
                failed_requests.add(request_id)
                self.remove_request_from_batch(request_id)
                outputs.append(
                    RequestOutput(
                        request_id,
                        sequences=[],
                        error=res.error,
                    )
                )
            elif res.error is None:
                valid_results.append(res)

        seq_outputs = defaultdict(list)

        for res in valid_results:
            request_id = res.sequence_id.request_id

            if request_id in failed_requests:
                # Other sequence in the same request has errored.
                continue

            seq_index = res.sequence_id.sequence_index
            state = self.current_batch[request_id]
            gen_seq = state.generation_sequences[seq_index]
            new_token_ids = res.generated_tokens

            for i, token_id in enumerate(new_token_ids):
                if (
                    token_id == self.tokenizer.eos_token_id
                    and not state.debug_options.ignore_eos
                ):
                    new_token_ids = new_token_ids[:i]
                    gen_seq.is_finished = True
                    break

            delta = update_sequence(
                gen_seq,
                new_token_ids,
                state.prompt_token_ids,
                self.tokenizer,
                state.stopping_criteria,
            )

            if not state.is_prefilled:
                # Successfully completed a prefill request
                state.is_prefilled = True

            finish_reason = None

            if gen_seq.is_finished:
                finish_reason = FinishReason.Stop

            if should_stop_by_length(
                gen_seq,
                state.prompt_len,
                self.max_context_length,
                state.stopping_criteria.max_tokens,
            ):
                gen_seq.is_finished = True
                finish_reason = FinishReason.Length

            seq_outputs[request_id].append(
                SequenceOutput(
                    seq_index,
                    delta,
                    num_generated_tokens=len(gen_seq.generated_token_ids),
                    finish_reason=finish_reason,
                )
            )

        for request_id, out_seqs in seq_outputs.items():
            state = self.current_batch[request_id]
            outputs.append(
                RequestOutput(
                    request_id,
                    sequences=out_seqs,
                    num_prompt_tokens=state.prompt_len,
                )
            )

            if state.is_finished:
                self.current_batch.pop(state.request_id)
                self.cache_manager.free_request(state)
                del self.num_sequences_per_requests[state.request_id]

        logger.debug("Finished detokenization and output object creation.")

        return InferenceStepResult(outputs)

    def _adjust_batch(self):
        with self.queue_lock:
            for request_id in list(self.requests_to_be_cancelled):
                if request_id in self.current_batch:
                    state = self.current_batch.pop(request_id)
                    self.cache_manager.free_request(state)
                    self.requests_to_be_cancelled.remove(request_id)

            self.evict_request(
                cancell_callback=lambda request_id: self.requests_to_be_cancelled.add(
                    request_id
                )
            )

            self._discard_cancelled_requests_from_queue()

            if self.cache_manager.get_max_new_tokens() <= self.max_decode_steps:
                logger.debug(
                    "Skip growing the batch due to max_decode_steps. Decode steps: %s",
                    self.cache_manager.get_max_new_tokens(),
                )
                return

            num_new_batched_tokens = len(self.current_batch)

            while self.queue and num_new_batched_tokens is not None:
                num_new_batched_tokens = self.try_grow_batch(num_new_batched_tokens)
                self._discard_cancelled_requests_from_queue()

    def has_pending_requests(self) -> bool:
        return bool(self.queue or self.current_batch)

    def _discard_cancelled_requests_from_queue(self):
        """
        Requires the self.queue_lock to be held before calling this function.
        """
        while self.queue and self.queue[0].request_id in self.requests_to_be_cancelled:
            state = self.queue.popleft()
            self.requests_to_be_cancelled.remove(state.request_id)
