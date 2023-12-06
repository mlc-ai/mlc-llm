"""
The worker for StagingInferenceEngine
"""
import time
import multiprocessing
import multiprocessing.synchronize
from collections import deque
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import Callable, Optional, Union, Any, Dict, Deque, List

import structlog

from .base import FinishReason, RequestId, RequestState, ValidationError
from .metrics import PrometheusMetrics
from .metrics_labels import *
from .model_module import (
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    SequenceId,
    TextGenerator,
    Tokenizer as TokenizerP,
)
from ..model.base import ModelArtifactConfig
from ..logging_utils import configure_logging

LOG = structlog.stdlib.get_logger(__name__)


@dataclass
class ShutdownCommand:
    pass


@dataclass
class AddRequestsCommand:
    request_states: list[RequestState]


@dataclass
class CancelRequestCommand:
    request_id: RequestId


@dataclass
class StopRequestCommand:
    request_id: RequestId


GenerationLoopWorkerCommand = Union[
    ShutdownCommand, AddRequestsCommand, CancelRequestCommand
]


@dataclass
class SequenceGenerationOutput:
    id: SequenceId
    new_tokens: list[int]
    finish_reason: Optional[FinishReason] = None
    error: Optional[Union[str, ValidationError]] = None


@dataclass
class GenerationLoopWorkerOutput:
    sequences: list[SequenceGenerationOutput]
    error: Optional[str] = None


class GenerationLoopWorker:
    text_generator: TextGenerator
    cache_manager: Any
    tokenizer: TokenizerP
    model_artifact_config: ModelArtifactConfig
    max_context_length: int
    max_num_batched_tokens: int
    max_decode_steps: int
    min_decode_steps: int
    queue_lock: Lock
    queue: Deque[RequestState]
    has_new_requests: Condition
    cancelled_requests: List[RequestState]
    stopped_requests: List[RequestState]
    current_batch: Dict[RequestId, RequestState]

    def __init__(
        self,
        model_module: ModelModule,
    ):
        self.text_generator = model_module.text_generator
        self.cache_manager = model_module.cache_manager
        self.tokenizer = model_module.tokenizer
        self.model_artifact_config = model_module.model_artifact_config
        assert self.model_artifact_config.max_context_length, "must not be None"
        self.max_context_length = self.model_artifact_config.max_context_length
        self.max_num_batched_tokens = model_module.engine_config.max_num_batched_tokens
        self.max_decode_steps = min(
            self.cache_manager.get_kv_cache_size(),
            model_module.engine_config.max_decode_steps,
        )
        self.min_decode_steps = min(
            self.max_decode_steps - 1, model_module.engine_config.min_decode_steps
        )

        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)

        self.cancelled_requests = list[RequestState]()
        self.stopped_requests = list[RequestState]()

        self.current_batch = dict[RequestId, RequestState]()

        self.prom_metrics = PrometheusMetrics()
        self.inv_kv_cache_size = 1.0 / self.cache_manager.get_kv_cache_size()

    def add(self, request_states: list[RequestState]):
        LOG.debug("GenerationLoopWorker", requests_states=request_states)
        with self.queue_lock:
            # States which have been invalidated should never be added, directly
            # cancel them instead.
            valid_states = []
            kv_cache_size = self.cache_manager.get_kv_cache_size()
            max_prompt_len = min(self.max_context_length, self.max_num_batched_tokens)
            for state in request_states:
                if (
                    state.validation_err is not None
                    or state.prompt_len > max_prompt_len
                    # We make sure that the KV cache will have enough free space for this request to proceed
                    # decoding for at least self.max_decode_steps steps.
                    or (kv_cache_size - state.prompt_len) < self.max_decode_steps
                ):
                    self.cancelled_requests.append(state)
                    if state.validation_err is None:
                        state.validation_err = ValidationError(
                            "The prompt is too long for the given set of engine parameters."
                        )
                else:
                    valid_states.append(state)

            self.queue.extend(valid_states)
            self.has_new_requests.notify_all()

    def _cacnel_or_stop_request(
        self, request_id: RequestId, requests: list[RequestState]
    ):
        with self.queue_lock:
            queue_index_to_delete = None
            for i, state in enumerate(self.queue):
                if state.request_id == request_id:
                    queue_index_to_delete = i
                    requests.append(state)
                    break

            if queue_index_to_delete is not None:
                del self.queue[queue_index_to_delete]

            if request_id in self.current_batch:
                requests.append(self.current_batch[request_id])

    def cancel_request(self, request_id: RequestId):
        self._cacnel_or_stop_request(request_id, self.cancelled_requests)

    def stop_request(self, request_id: RequestId):
        self._cacnel_or_stop_request(request_id, self.stopped_requests)

    def wait_for_request(self, timeout_seconds=None):
        with self.queue_lock:
            self.has_new_requests.wait_for(
                self.has_pending_requests, timeout=timeout_seconds
            )

    def has_pending_requests(self) -> bool:
        return len(self.queue) != 0 or len(self.current_batch) != 0 or len(self.cancelled_requests) != 0

    def step(self) -> GenerationLoopWorkerOutput:
        LOG.debug("Starting new inference step.")

        outputs = list[SequenceGenerationOutput]()
        result = GenerationLoopWorkerOutput(sequences=outputs)

        # TODO: consolidate into a single function
        for state in list(self.current_batch.values()):
            finish_reason = None
            if state.is_ended:
                finish_reason = FinishReason.Stop
            if self._should_stop_by_length(state):
                finish_reason = FinishReason.Length

            if finish_reason is not None:
                outputs.append(
                    SequenceGenerationOutput(
                        # TODO: support multi-sequence
                        id=SequenceId(state.request_id, 0),
                        new_tokens=[],
                        finish_reason=finish_reason,
                    )
                )
                self._remove_request_from_batch(state.request_id)
                duration = time.time() - state.arrival_timestamp
                self.prom_metrics.histogram(E2E_LATENCY).observe(duration)

        for state in self.stopped_requests:
            outputs.append(
                SequenceGenerationOutput(
                    # TODO: support multi-sequence
                    id=SequenceId(state.request_id, 0),
                    new_tokens=[],
                    finish_reason=FinishReason.Stop,
                )
            )
            if state.request_id in self.current_batch:
                self._remove_request_from_batch(state.request_id)

        self.stopped_requests.clear()

        with self.queue_lock:
            # Hold the lock here since self.cancelled_requests is modified in add(...) as well.
            for state in self.cancelled_requests:
                err = None
                if state.validation_err:
                    err = state.validation_err

                outputs.append(
                    SequenceGenerationOutput(
                        # TODO: support multi-sequence
                        id=SequenceId(state.request_id, 0),
                        new_tokens=[],
                        finish_reason=FinishReason.Cancelled,
                        error=err,
                    )
                )
                if state.request_id in self.current_batch:
                    self._remove_request_from_batch(state.request_id)

            self.cancelled_requests.clear()

        self._adjust_batch()

        if not self.current_batch:
            if len(self.queue) > 0:
                LOG.warn(
                     f"The engine has {len(self.queue)} requests to be processed in the queue, but none of them were added to the current batch during the execution of StagingEngine._adjust_batch"
                )
            return result

        requests, is_prompt_batch = self._get_requests_to_process()
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        LOG.debug("Finished text generation.")

        for res in results:
            # For now we only support single sequence per request
            request_id = res.sequence_id.request_id
            if res.error is not None:
                self._remove_request_from_batch(request_id)
                outputs.append(
                    SequenceGenerationOutput(
                        # TODO: support multi-sequence
                        id=res.sequence_id,
                        new_tokens=[],
                        error=res.error,
                    )
                )
                continue

            state = self.current_batch[request_id]
            state.next_start_position = len(state.token_ids)
            new_tokens = res.generated_tokens

            # Need to match at the token-id level
            for i, token_id in enumerate(new_tokens):
                if (
                    token_id == self.tokenizer.eos_token_id
                    and not state.debug_options.ignore_eos
                ):
                    new_tokens = new_tokens[:i]
                    state.is_ended = True
                    break

            state.token_ids.extend(new_tokens)
            outputs.append(
                SequenceGenerationOutput(id=res.sequence_id, new_tokens=new_tokens)
            )

            if is_prompt_batch:
                ttft = time.time() - state.arrival_timestamp
                self.prom_metrics.histogram(FIRST_TOKEN_LATENCY).observe(ttft)

        self.prom_metrics.gauge(KV_CACHE_UTILIZATION).set(
            1.0 - self.cache_manager.get_free_space() * self.inv_kv_cache_size
        )

        LOG.debug("Finished state update and stopping criteria check.")

        return result

    def _adjust_batch(self):
        with self.queue_lock:
            while self.cache_manager.get_max_new_tokens() < 1:
                self.prom_metrics.counter(NUM_CACHE_EVICTONS).inc()
                request_to_remove = min(
                    self.current_batch.values(), key=lambda s: len(s.token_ids)
                )
                self._remove_request_from_batch(request_to_remove.request_id)
                self.queue.appendleft(request_to_remove)
                LOG.debug(
                    "Preempt request to free %s tokens",
                    len(request_to_remove.token_ids),
                )

            if self.cache_manager.get_max_new_tokens() <= self.max_decode_steps:
                LOG.debug(
                    "Skip growing the batch due to max_decode_steps. Decode steps: %s",
                    self.cache_manager.get_max_new_tokens(),
                )
                return

            num_new_batched_tokens = len(self.current_batch)
            while self.queue:
                max_new_tokens = self.cache_manager.get_max_new_tokens()
                if max_new_tokens < self.min_decode_steps:
                    LOG.debug(
                        "Stop growing the batch due to min_decode_steps. Decode steps: %s",
                        max_new_tokens,
                    )
                    # stop adding request if there isn't enough space to do a certain steps of decoding.
                    break
                state = self.queue[0]
                num_tokens = len(state.token_ids)
                num_new_batched_tokens += num_tokens
                # This can happen when we are recovering from cache eviction and the sum of prompt
                # and intermediate decode tokens is bigger than the biggest allowable batch size,
                # self.max_num_batched_tokens. In such cases, we need to discard the recent decode
                # tokens that cannot fit into a batch, and recompute them after we fill the cache
                # entries for the older tokens.
                if (
                    len(self.current_batch) == 0
                    and num_new_batched_tokens > self.max_num_batched_tokens
                ):
                    state.token_ids = state.token_ids[: self.max_num_batched_tokens]
                    state.next_start_position = (
                        num_new_batched_tokens
                    ) = num_tokens = self.max_num_batched_tokens
                if num_new_batched_tokens > self.max_num_batched_tokens:
                    LOG.debug(
                        "Stop growing the batch due to max_num_batched_tokens. Batched tokens: %s",
                        num_new_batched_tokens,
                    )
                    break
                # We make sure that the KV cache will have enough free space for all sequences in the batch
                # to proceed decoding for at least self.max_decode_steps steps.
                if (self.cache_manager.get_free_space() - num_tokens) / (
                    len(self.current_batch) + 1
                ) < self.max_decode_steps:
                    LOG.debug(
                        "Stop growing the batch due to not enough free space. Free: %s, Num tokens: %s",
                        self.cache_manager.get_free_space(),
                        num_tokens,
                    )
                    break

                self.queue.popleft()
                self.cache_manager.allocate(state.request_id, num_tokens)
                self.current_batch[state.request_id] = state

    def _remove_request_from_batch(self, request_id: RequestId):
        del self.current_batch[request_id]
        self.cache_manager.free(SequenceId(request_id, 0))

    def _get_requests_to_process(self):
        requests = []
        # TODO: consider having hybrid batch if the underlying attention kernel supports
        # mixing prefill and decode.
        is_prompt_batch = any(
            state.next_start_position == 0 for state in self.current_batch.values()
        )

        if is_prompt_batch:
            prefill_token_counts = 0

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
                    prefill_token_counts += len(state.token_ids)

            self.prom_metrics.histogram(BATCHED_PREFILL_TOKENS).observe(prefill_token_counts)

            LOG.debug(
                "Creating prompt batch.",
                num_requests=len(requests),
                total_tokens=prefill_token_counts,
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

            decode_token_counts = len(requests)
            self.prom_metrics.histogram(BATCHED_DECODE_TOKENS).observe(decode_token_counts)

            LOG.debug("Creating decode batch with %s requests.", decode_token_counts)

        return requests, is_prompt_batch

    def _has_request_to_process(self) -> bool:
        return len(self.queue) != 0 or len(self.current_batch) != 0

    def _should_stop_by_length(self, state: RequestState) -> bool:
        # TODO: currently, we simply return true for both stopping reasons.
        #       in the future, we can differentiate these two.
        # this include prompt tokens and gen tokens so far
        num_context_tokens = len(state.token_ids)
        assert self.model_artifact_config.max_context_length is not None
        if num_context_tokens >= self.model_artifact_config.max_context_length:
            return True
        num_gen_tokens = num_context_tokens - state.prompt_len
        if (
            state.stopping_criteria.max_tokens is not None
            and num_gen_tokens >= state.stopping_criteria.max_tokens
        ):
            return True
        return False


def run_generation_loop_worker(
    model_module_loader: Callable[..., ModelModule],
    model_module_loader_kwargs: dict,
    command_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    ready_event: multiprocessing.synchronize.Event,
    contextvars: Optional[Dict[str, Any]] = None,
    enable_json_logs = False,
    log_level="INFO",
):
    configure_logging(enable_json_logs, log_level)
    structlog.contextvars.bind_contextvars(**contextvars)

    try:
        model_module = model_module_loader(**model_module_loader_kwargs)
        worker = GenerationLoopWorker(model_module=model_module)
    except:
        LOG.exception("An error raised in model initialization.")
        return

    should_stop = False

    def handle_command():
        while True:
            cmd = command_queue.get()
            if isinstance(cmd, ShutdownCommand):
                break
            elif isinstance(cmd, AddRequestsCommand):
                worker.add(cmd.request_states)
            elif isinstance(cmd, CancelRequestCommand):
                worker.cancel_request(cmd.request_id)
            elif isinstance(cmd, StopRequestCommand):
                worker.stop_request(cmd.request_id)
            else:
                LOG.error("Unknown command type %s", type(cmd))
                break

        nonlocal should_stop
        should_stop = True

    handler_thread = Thread(
        target=handle_command, name="staging-engine-worker-command-handler"
    )
    handler_thread.start()

    ready_event.set()

    while True:
        worker.wait_for_request(timeout_seconds=1)
        if should_stop:
            return
        if not worker.has_pending_requests():
            continue

        try:
            output = worker.step()
        except Exception as exc:
            LOG.exception("Error when calling GenerationLoopWorker.step")
            output = GenerationLoopWorkerOutput(sequences=[], error=str(exc))
            result_queue.put(output)
            break

        if output.sequences:
            # result_queue should have size limit and the blocking behavior
            # of queue.put will naturally limits the tokens it generates ahead of time.
            result_queue.put(output)

    handler_thread.join()
