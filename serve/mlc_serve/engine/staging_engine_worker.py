"""
The worker for StagingInferenceEngine
"""

import logging
import multiprocessing
from collections import deque
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import Callable, Optional, Union, List

from .base import FinishReason, RequestId, RequestState, StagingInferenceEngineConfig
from .model_module import DecodeRequest, ModelModule, PrefillRequest, SequenceId

logger = logging.getLogger(__name__)


@dataclass
class ShutdownCommand:
    pass


@dataclass
class AddRequestsCommand:
    request_states: List[RequestState]


@dataclass
class CancelRequestCommand:
    request_id: RequestId


GenerationLoopWorkerCommand = Union[
    ShutdownCommand, AddRequestsCommand, CancelRequestCommand
]


@dataclass
class SequenceGenerationOutput:
    id: SequenceId
    new_tokens: List[int]
    finish_reason: Optional[FinishReason] = None
    error: Optional[str] = None


@dataclass
class GenerationLoopWorkerOutput:
    sequences: List[SequenceGenerationOutput]
    error: Optional[str] = None


class GenerationLoopWorker:
    def __init__(
        self,
        model_module: ModelModule,
        config: StagingInferenceEngineConfig,
    ):
        self.text_generator = model_module.text_generator
        self.cache_manager = model_module.cache_manager
        self.tokenizer = model_module.tokenizer

        self.max_model_tokens = config.max_model_tokens
        self.max_batched_tokens = config.max_batched_tokens
        self.max_decode_steps = min(
            self.cache_manager.get_kv_cache_size(), config.max_decode_steps
        )
        self.min_decode_steps = min(self.max_decode_steps - 1, config.min_decode_steps)
        self.prompt_allocate_ratio = config.prompt_allocate_ratio
        assert config.prompt_allocate_ratio >= 1.0

        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)

        self.cancelled_requests = list[RequestState]()

        self.current_batch = dict[RequestId, RequestState]()

    def add(self, request_states: list[RequestState]):
        with self.queue_lock:
            self.queue.extend(request_states)
            self.has_new_requests.notify_all()

    def cancel(self, request_id: RequestId):
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

    def wait_for_request(self, timeout_seconds=None) -> bool:
        with self.queue_lock:
            self.has_new_requests.wait_for(
                self.has_pending_requests, timeout=timeout_seconds
            )

    def has_pending_requests(self) -> bool:
        return self.queue or self.current_batch

    def step(self) -> GenerationLoopWorkerOutput:
        logger.debug("Starting new inference step.")

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

        for state in self.cancelled_requests:
            outputs.append(
                SequenceGenerationOutput(
                    # TODO: support multi-sequence
                    id=SequenceId(state.request_id, 0),
                    new_tokens=[],
                    finish_reason=FinishReason.Cancelled,
                )
            )
            if state.request_id in self.current_batch:
                self._remove_request_from_batch(state.request_id)

        self.cancelled_requests.clear()

        self._adjust_batch()

        if not self.current_batch:
            return result

        requests = self._get_requests_to_process()
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        logger.debug("Finished text generation.")

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

        logger.debug("Finished state update and stopping criteria check.")

        return result

    def _adjust_batch(self):
        with self.queue_lock:
            while self.cache_manager.get_max_new_tokens() < 1:
                request_to_remove = min(
                    self.current_batch.values(), key=lambda s: len(s.token_ids)
                )
                self._remove_request_from_batch(request_to_remove.request_id)
                self.queue.appendleft(request_to_remove)
                logger.debug(
                    "Preempt request to free %s tokens",
                    len(request_to_remove.token_ids),
                )

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
                if num_new_batched_tokens > self.max_batched_tokens > 0:
                    logger.debug(
                        "Stop growing the batch due to max_batched_tokens. Batched tokens: %s",
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

    def _has_request_to_process(self) -> bool:
        return self.queue or self.current_batch

    def _should_stop_by_length(self, state: RequestState) -> bool:
        if state.stopping_criteria.max_tokens is not None:
            max_tokens = min(self.max_model_tokens, state.stopping_criteria.max_tokens)

        return len(state.token_ids) - state.prompt_len >= max_tokens


def setup_logging(level):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": level,  # Set the handler's log level to DEBUG
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,  # Set the logger's log level to DEBUG
        },
        "mlc_serve.engine.staging_engine_worker": {"level": level},
    }
    logging.config.dictConfig(logging_config)


def run_generation_loop_worker(
    model_module_loader: Callable[..., ModelModule],
    model_module_loader_kwargs: dict,
    worker_kwargs: dict,
    command_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
    log_level="INFO",
):
    setup_logging(log_level)
    model_module = model_module_loader(**model_module_loader_kwargs)
    worker = GenerationLoopWorker(model_module=model_module, **worker_kwargs)

    should_stop = False

    def handle_command():
        while True:
            cmd = command_queue.get()
            if isinstance(cmd, ShutdownCommand):
                break
            elif isinstance(cmd, AddRequestsCommand):
                worker.add(cmd.request_states)
            elif isinstance(cmd, CancelRequestCommand):
                worker.cancel(cmd.request_id)
            else:
                logger.error("Unknown command type %s", type(cmd))
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
            logger.exception("Error when calling GenerationLoopWorker.step")
            output = GenerationLoopWorkerOutput(sequences=[], error=str(exc))
            result_queue.put(output)
            break

        if output.sequences:
            # result_queue should have size limit and the blocking behavior
            # of queue.put will naturally limits the tokens it generates ahead of time.
            result_queue.put(output)

    handler_thread.join()
