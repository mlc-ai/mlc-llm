"""
The worker for StagingInferenceEngine
"""
import time
import multiprocessing
import multiprocessing.synchronize
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Callable, Optional, Union, Any, Dict, List

import structlog

from .base import (
    FinishReason,
    RequestId,
    RequestState,
    ValidationError,
    SequenceId,
    GenerationSequence,
)
from .metrics import PrometheusMetrics
from .metrics_labels import *
from .model_module import (
    ModelModule,
)
from .engine_common import (
    should_stop_by_length,
    get_requests_to_process,
    EngineBase,
)
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
class StopSequenceCommand:
    sequence_id: SequenceId


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
    error: Optional[BaseException] = None


class GenerationLoopWorker(EngineBase):
    cancelled_requests: List[RequestState]
    stopped_sequences: List[SequenceId]
    stopped_sequences_lock: Lock
    sequence_map: Dict[SequenceId, GenerationSequence]
    prom_metrics: PrometheusMetrics
    inv_kv_cache_size: float

    def __init__(
        self,
        model_module: ModelModule,
    ):
        EngineBase.__init__(self, model_module)

        self.cancelled_requests = list[RequestState]()
        self.stopped_sequences = list[SequenceId]()
        self.stopped_sequences_lock = Lock()
        self.sequence_map = dict[SequenceId, GenerationSequence]()

        self.prom_metrics = PrometheusMetrics()
        self.inv_kv_cache_size = 1.0 / self.cache_manager.get_kv_cache_size()

    def add(self, request_states: list[RequestState]):
        LOG.debug("GenerationLoopWorker", requests_states=request_states)
        with self.queue_lock:
            # States which have been invalidated should never be added, directly
            # cancel them instead.
            valid_states = []
            for state in request_states:
                if state.validation_err is not None or self.check_prompt_too_long(
                    state.prompt_len, state.num_sequences
                ):
                    self.cancelled_requests.append(state)
                    if state.validation_err is None:
                        state.validation_err = ValidationError(
                            "The prompt is too long for the given set of engine"
                            " parameters."
                        )
                else:
                    valid_states.append(state)

                for seq in state.generation_sequences:
                    self.sequence_map[seq.seq_id] = seq

            self.queue.extend(valid_states)
            self.has_new_requests.notify_all()

    def cancel_request(self, request_id: RequestId):
        with self.queue_lock:
            queue_index_to_delete = None
            for i, state in enumerate(self.queue):
                if state.request_id == request_id:
                    queue_index_to_delete = i
                    self.cancelled_requests.append(state)
                    break

            if queue_index_to_delete is not None:
                del self.queue[queue_index_to_delete]

            if request_id in self.current_batch:
                self.cancelled_requests.append(self.current_batch[request_id])

    def stop_sequence(self, sequence_id: SequenceId):
        with self.stopped_sequences_lock:
            self.stopped_sequences.append(sequence_id)

    def create_aborted_outputs(
        self,
        cancelled_or_stopped_requests: List[RequestState],
        finish_reason: FinishReason,
    ):
        outputs = []
        for state in cancelled_or_stopped_requests:
            err = None
            if state.validation_err:
                err = state.validation_err

            for gen_seq in state.generation_sequences:
                outputs.append(
                    SequenceGenerationOutput(
                        id=gen_seq.seq_id,
                        new_tokens=[],
                        finish_reason=finish_reason,
                        error=err,
                    )
                )
                del self.sequence_map[gen_seq.seq_id]

            if state.request_id in self.current_batch:
                self.remove_request_from_batch(state.request_id)

        cancelled_or_stopped_requests.clear()
        return outputs

    def wait_for_request(self, timeout_seconds=None):
        with self.queue_lock:
            self.has_new_requests.wait_for(
                self.has_pending_requests, timeout=timeout_seconds
            )

    def has_pending_requests(self) -> bool:
        return bool(self.queue or self.current_batch or self.cancelled_requests)

    def step(self) -> GenerationLoopWorkerOutput:
        LOG.debug("Starting new inference step.")

        outputs = list[SequenceGenerationOutput]()
        result = GenerationLoopWorkerOutput(sequences=outputs)

        with self.stopped_sequences_lock:
            for seq_id in self.stopped_sequences:
                gen_seq = self.sequence_map[seq_id]
                if not gen_seq.is_finished:
                    gen_seq.is_finished = True

            self.stopped_sequences.clear()

        for state in list(self.current_batch.values()):
            if state.is_finished:
                for gen_seq in state.generation_sequences:
                    del self.sequence_map[gen_seq.seq_id]

                self.remove_request_from_batch(state.request_id)

                duration = time.time() - state.arrival_timestamp
                self.prom_metrics.histogram(E2E_LATENCY).observe(duration)

        with self.queue_lock:
            # Hold the lock here since self.cancelled_requests is modified in add(...) as well.
            outputs += self.create_aborted_outputs(
                self.cancelled_requests, finish_reason=FinishReason.Cancelled
            )

        self._adjust_batch()

        with self.queue_lock:
            # _adjust_batch also adds to self.cancelled_requests
            outputs += self.create_aborted_outputs(
                self.cancelled_requests, finish_reason=FinishReason.Cancelled
            )

        if not self.current_batch:
            if len(self.queue) > 0:
                LOG.warn(
                    f"The engine has {len(self.queue)} requests to be processed in the"
                    " queue, but none of them were added to the current batch during"
                    " the execution of StagingEngine._adjust_batch"
                )
            return result

        requests, is_prompt_batch = self._get_requests_to_process()
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        LOG.debug("Finished text generation.")

        failed_requests = set()

        for res in results:
            request_id = res.sequence_id.request_id

            if res.error is not None and request_id not in failed_requests:
                failed_requests.add(request_id)
                self.remove_request_from_batch(request_id)

                outputs.append(
                    SequenceGenerationOutput(
                        id=res.sequence_id,
                        new_tokens=[],
                        error=res.error,
                    )
                )
                continue

            state = self.current_batch[request_id]
            seq_index = res.sequence_id.sequence_index
            gen_seq = state.generation_sequences[seq_index]
            new_tokens = res.generated_tokens

            gen_seq.next_start_position = state.prompt_len + len(
                gen_seq.generated_token_ids
            )

            finish_reason = None

            # Need to match at the token-id level
            for i, token_id in enumerate(new_tokens):
                if (
                    token_id == self.tokenizer.eos_token_id
                    and not state.debug_options.ignore_eos
                ):
                    new_tokens = new_tokens[:i]
                    gen_seq.is_finished = True
                    finish_reason = FinishReason.Stop
                    break

            gen_seq.generated_token_ids.extend(new_tokens)

            if should_stop_by_length(
                gen_seq,
                state.prompt_len,
                self.max_context_length,
                state.stopping_criteria.max_tokens,
            ):
                gen_seq.is_finished = True
                finish_reason = FinishReason.Length

            outputs.append(
                SequenceGenerationOutput(
                    id=res.sequence_id,
                    new_tokens=new_tokens,
                    finish_reason=finish_reason,
                )
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
            num_eviction = self.evict_request(
                cancell_callback=lambda request_id: self.cancelled_requests.append(
                    self.current_batch[request_id]
                )
            )
            self.prom_metrics.counter(NUM_CACHE_EVICTONS).inc(num_eviction)

            if self.cache_manager.get_max_new_tokens() <= self.max_decode_steps:
                LOG.debug(
                    "Skip growing the batch due to max_decode_steps. Decode steps: %s",
                    self.cache_manager.get_max_new_tokens(),
                )
                return

            num_new_batched_tokens = len(self.current_batch)

            while self.queue and num_new_batched_tokens is not None:
                num_new_batched_tokens = self.try_grow_batch(num_new_batched_tokens)

    def _get_requests_to_process(self):
        requests, is_prompt_batch, token_counts = get_requests_to_process(
            self.current_batch.values(), self.cache_manager
        )

        if is_prompt_batch:
            self.prom_metrics.histogram(BATCHED_PREFILL_TOKENS).observe(token_counts)
        else:
            self.prom_metrics.histogram(BATCHED_DECODE_TOKENS).observe(token_counts)

        return requests, is_prompt_batch

    def _has_request_to_process(self) -> bool:
        return bool(self.queue or self.current_batch)


def run_generation_loop_worker(
    model_module_loader: Callable[..., ModelModule],
    model_module_loader_kwargs: dict,
    command_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    ready_event: multiprocessing.synchronize.Event,
    contextvars: Optional[Dict[str, Any]] = None,
    enable_json_logs: bool = False,
    log_level: str = "INFO",
):
    configure_logging(enable_json_logs, log_level)
    structlog.contextvars.bind_contextvars(**contextvars)

    try:
        model_module = model_module_loader(**model_module_loader_kwargs)
        LOG.info("Model is initalized.")
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
            elif isinstance(cmd, StopSequenceCommand):
                worker.stop_sequence(cmd.sequence_id)
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
            LOG.exception("Error when calling GenerationLoopWorker.step", exc=exc)
            output = GenerationLoopWorkerOutput(sequences=[], error=exc)
            result_queue.put(output)
            break

        if output.sequences:
            # result_queue should have size limit and the blocking behavior
            # of queue.put will naturally limits the tokens it generates ahead of time.
            result_queue.put(output)

    handler_thread.join()
