"""
An implementation of InferenceEngine that offloads the text generation loop to another worker process.
"""
import logging
import multiprocessing
import queue
from threading import Lock
from typing import Callable, Optional

from .base import (
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    RequestState,
    ScopedInferenceEngine,
    SequenceOutput,
    check_stopping_sequences
)
from .model_module import ModelModule, TokenizerModule
from .staging_engine_worker import (
    AddRequestsCommand,
    CancelRequestCommand,
    ShutdownCommand,
    run_generation_loop_worker,
)

logger = logging.getLogger(__name__)


class StagingInferenceEngine(ScopedInferenceEngine):
    """
    An implementation of InferenceEngine that offloads the text generation loop to another worker process,
    Text tokens are generated asynchronously from the invocation of `step`. The generation progress could be one step
    ahead of the invocation of `step`. Tokenization and detokenization is still processed synchronously
    when `step` is called.
    """

    def __init__(
        self,
        tokenizer_module: TokenizerModule,
        model_module_loader: Callable[..., ModelModule],
        model_module_loader_kwargs: dict,
    ):
        self.next_generation_output = None
        self.requests_lock = Lock()
        self.requests = dict[RequestId, RequestState]()

        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

        self.mp_context = multiprocessing.get_context("spawn")
        self.command_queue = self.mp_context.Queue()
        self.result_queue = self.mp_context.Queue(maxsize=1)
        self.ready_event = self.mp_context.Event()
        self.worker_process = self.mp_context.Process(
            target=run_generation_loop_worker,
            args=(
                model_module_loader,
                model_module_loader_kwargs,
                self.command_queue,
                self.result_queue,
                self.ready_event,
            ),
        )

    def start(self):
        self.worker_process.start()
        if not self.ready_event.wait(timeout=180):
            raise RuntimeError(
                "StagingInferenceEngine worker is not ready before timeout."
            )

    def stop(self):
        self.command_queue.put(ShutdownCommand())
        self.worker_process.join()

    def add(self, requests: list[Request]):
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            if req.num_sequences > 1:
                raise RuntimeError("num_sequences > 1 is not supported for now")

            # If the request violates the tokenization, this returns None, so skip.
            state = self._get_new_request_state(req)
            new_request_states.append(state)

        self.command_queue.put(AddRequestsCommand(request_states=new_request_states))

        with self.requests_lock:
            self.requests.update({s.request_id: s for s in new_request_states})

    def cancel(self, request_id: RequestId):
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")
        self.command_queue.put(CancelRequestCommand(request_id))

    def has_pending_requests(self) -> bool:
        with self.requests_lock:
            return len(self.requests) > 0

    def wait_for_request(self, timeout_seconds=None) -> bool:
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")

        if self.next_generation_output is not None:
            return True

        try:
            self.next_generation_output = self.result_queue.get(timeout=timeout_seconds)
            return True
        except queue.Empty:
            return False

    def step(self) -> InferenceStepResult:
        if not self._is_ready_to_serve():
            raise RuntimeError("GenerationLoopWorker process is not running")
        if not self.has_pending_requests():
            return InferenceStepResult([])

        if self.next_generation_output is None:
            generation_output = self.result_queue.get()
        else:
            generation_output = self.next_generation_output
            self.next_generation_output = None

        if generation_output.error is not None:
            raise RuntimeError(
                f"Error when calling GenerationLoopWorker: {generation_output.error}"
            )

        outputs = list[RequestOutput]()
        with self.requests_lock:
            for seq_output in generation_output.sequences:
                # TODO: support multi-sequence per request
                request_id = seq_output.id.request_id
                if request_id not in self.requests:
                    logger.warn(
                        "Unknown request %s from GenerationLoopWorkerOutput", request_id
                    )
                    continue

                state = self.requests[request_id]

                if seq_output.error is not None:
                    outputs.append(
                        RequestOutput(
                            request_id,
                            sequences=[],
                            error=seq_output.error,
                            num_prompt_tokens=state.prompt_len,
                        )
                    )
                    del self.requests[request_id]
                    continue

                state.next_start_position = len(state.token_ids)
                state.token_ids.extend(seq_output.new_tokens)

                delta = self._decode_last_output(state)
                state.output_text += delta

                state.output_text, delta, state.is_ended = check_stopping_sequences(state.stopping_criteria,
                                                                                state.output_text,
                                                                                delta,
                                                                                state.is_ended)

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
                                finish_reason=seq_output.finish_reason,
                            ),
                        ],
                        num_prompt_tokens=state.prompt_len,
                    )
                )

                if seq_output.finish_reason is not None:
                    del self.requests[request_id]

        return InferenceStepResult(outputs=outputs)

    def _is_ready_to_serve(self) -> bool:
        return self.worker_process is not None and self.worker_process.is_alive()

    def _get_new_request_state(self, request: Request) -> RequestState:
        if request.debug_options.prompt is not None:
            prompt = request.debug_options.prompt
        else:
            prompt = self.conversation_template.apply(request.messages)

        prompt_tokens = self.tokenizer.encode(prompt)

        validation_err = None
        if request.validate_tokens is not None:
            validation_err = request.validate_tokens(request, prompt_tokens)

        return RequestState(
            request_id=request.request_id,
            token_ids=prompt_tokens,
            prompt_len=len(prompt_tokens),
            next_start_position=0,
            sampling_params=request.sampling_params,
            stopping_criteria=request.stopping_criteria,
            debug_options=request.debug_options,
            output_text="",
            validation_err=validation_err,
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
