"""
A implementation of InferenceEngine that executes in the current process.
"""

import logging
from typing import Deque, Set, Dict
from collections import deque
from sre_parse import Tokenizer
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
    RequestState,
    SamplingParams,
    SequenceOutput,
    StoppingCriteria,
    check_stopping_sequences,
)
from .model_module import DecodeRequest, ModelModule, PrefillRequest, SequenceId, TextGenerator, Tokenizer as TokenizerP
from ..model.base import ModelArtifactConfig

logger = logging.getLogger(__name__)


class SynchronousInferenceEngine(InferenceEngine):
    """
    A implementation of InferenceEngine that does inference synchronously in the current thread
    when `step` is called.
    """
    text_generator: TextGenerator
    tokenizer: TokenizerP
    model_artifact_config: ModelArtifactConfig
    max_context_length: int
    max_num_batched_tokens: int
    max_decode_steps: int
    min_decode_steps: int
    prompt_allocate_ratio: float
    queue_lock: Lock
    queue: Deque[RequestState]
    has_new_requests: Condition
    requests_to_be_cancelled: Set[RequestId]
    current_batch: Dict[RequestId, RequestState]

    def __init__(
        self,
        model_module: ModelModule,
    ):
        self.text_generator = model_module.text_generator
        self.tokenizer = model_module.tokenizer
        self.conversation_template = model_module.conversation_template
        self.cache_manager = model_module.cache_manager
        self.model_artifact_config = model_module.model_artifact_config
        assert self.model_artifact_config.max_context_length, "max_context_length must not be zero"
        self.max_context_length = self.model_artifact_config.max_context_length
        self.max_num_batched_tokens = model_module.engine_config.max_num_batched_tokens
        self.max_decode_steps = min(
            self.cache_manager.get_kv_cache_size(),
            model_module.engine_config.max_decode_steps,
        )
        self.min_decode_steps = min(
            self.max_decode_steps - 1, model_module.engine_config.min_decode_steps
        )
        self.prompt_allocate_ratio = model_module.engine_config.prompt_allocate_ratio
        assert self.prompt_allocate_ratio >= 1.0

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

            # wrap the stop sequence with list if necessary
            if req.stopping_criteria.stop_sequences:
                if isinstance(req.stopping_criteria.stop_sequences, str):
                    req.stopping_criteria.stop_sequences = [
                        req.stopping_criteria.stop_sequences
                    ]
                assert isinstance(req.stopping_criteria.stop_sequences, list)

            state = self._get_new_request_state(req)
            new_request_states.append(state)

            if state.prompt_len >= self.max_context_length:
                self.cancel(req.request_id)

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
                        [
                            SequenceOutput(
                                0,
                                finish_reason=finish_reason,
                                num_generated_tokens=(
                                    len(state.token_ids) - state.prompt_len
                                ),
                            )
                        ],
                        num_prompt_tokens=state.prompt_len,
                    )
                )
                self.current_batch.pop(state.request_id)
                self.cache_manager.free(SequenceId(state.request_id, 0))

        previous_requests_to_be_cancelled = set(self.requests_to_be_cancelled)
        self._adjust_batch()
        
        if not self.current_batch:
            if len(self.queue) > 0:
                logger.warning(
                    f"The engine has {len(self.queue)} requests to be processed in the queue, but none of them were added to the current batch during the execution of SyncEngine._adjust_batch"
                )

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

        requests = self._get_requests_to_process()
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        logger.debug("Finished text generation.")

        valid_results = []

        for res in results:
            # For now we only support single sequence per request
            request_id = res.sequence_id.request_id
            if res.error is not None:
                del self.current_batch[request_id]
                self.cache_manager.free(res.sequence_id)
                outputs.append(
                    RequestOutput(
                        res.sequence_id.request_id,
                        sequences=[],
                        error=res.error,
                    )
                )
            else:
                valid_results.append(res)

        for res in valid_results:
            request_id = res.sequence_id.request_id
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
                    break

            state.token_ids.extend(new_token_ids)

            delta = self._decode_last_output(state)
            state.output_text += delta

            state.output_text, delta, state.is_ended = check_stopping_sequences(
                state.stopping_criteria, state.output_text, delta, state.is_ended
            )

            outputs.append(
                RequestOutput(
                    request_id,
                    sequences=[
                        SequenceOutput(
                            0,
                            delta=delta,
                            num_generated_tokens=(
                                len(state.token_ids) - state.prompt_len
                            ),
                        ),
                    ],
                    num_prompt_tokens=state.prompt_len,
                )
            )

        logger.debug("Finished detokenization and output object creation.")

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

            if self.cache_manager.get_max_new_tokens() <= self.max_decode_steps:
                logger.debug(
                    "Skip growing the batch due to max_decode_steps. Decode steps: %s",
                    self.cache_manager.get_max_new_tokens(),
                )
                return

            num_new_batched_tokens = len(self.current_batch)
            while self.queue:
                max_new_tokens = self.cache_manager.get_max_new_tokens()
                if max_new_tokens < self.min_decode_steps:
                    logger.debug(
                        "Stop growing the batch due to min_decode_steps. Decode steps: %s",
                        max_new_tokens,
                    )
                    # stop adding request if there isn't enough space to do a certain steps of decoding.
                    break
                state = self.queue[0]
                num_tokens = len(state.token_ids)
                num_new_batched_tokens += num_tokens
                if num_new_batched_tokens > self.max_num_batched_tokens > 0:
                    logger.debug(
                        "Stop growing the batch due to max_num_batched_tokens. Batched tokens: %s",
                        num_new_batched_tokens,
                    )
                    break
                if (
                    self.cache_manager.get_free_space()
                    <= self.prompt_allocate_ratio * num_tokens
                ):
                    logger.debug(
                        "Stop growing the batch due to not enough free space. Free: %s, Num tokens: %s",
                        self.cache_manager.get_free_space(),
                        num_tokens,
                    )
                    break

                self.queue.popleft()
                self.cache_manager.allocate(state.request_id, num_tokens)
                self.current_batch[state.request_id] = state

                self._discard_cancelled_requests_from_queue()

    def _get_requests_to_process(self):
        requests = []
        # TODO: consider having hybrid batch if the underlying attention kernel supports
        # mixing prefill and decode.
        is_prompt_batch = any(
            state.next_start_position == 0 for state in self.current_batch.values()
        )

        if is_prompt_batch:
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
            logger.debug(
                "Creating prompt batch with %s requests with %s total tokens.",
                len(requests),
                sum(len(r.token_ids) for r in requests),
            )
        else:
            for state in self.current_batch.values():
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
            logger.debug("Creating decode batch with %s requests.", len(requests))

        return requests

    def has_pending_requests(self) -> bool:
        return bool(self.queue or self.current_batch)

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
        # TODO: currently, we simply return true for both stopping reasons.
        #       in the future, we can differentiate these two.
        # this include prompt tokens and gen tokens so far
        num_context_tokens = len(state.token_ids)
        if num_context_tokens >= self.max_context_length:
            return True
        num_gen_tokens = num_context_tokens - state.prompt_len
        if (
            state.stopping_criteria.max_tokens is not None
            and num_gen_tokens >= state.stopping_criteria.max_tokens
        ):
            return True
        return False
