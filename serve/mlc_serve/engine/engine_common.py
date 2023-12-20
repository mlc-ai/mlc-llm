"""
Common utilites for engine classes.
"""

import time
from typing import Tuple, Deque, Dict, Optional, Union, Callable
from collections import deque
from threading import Condition, Lock

import structlog

from .base import (
    Request,
    RequestId,
    RequestState,
    GenerationSequence,
    SequenceId,
    StoppingCriteria,
)
from .model_module import (
    DecodeRequest,
    PrefillRequest,
    ConversationTemplate,
    KVCacheManager,
    ModelModule,
    TextGenerator,
    Tokenizer as TokenizerP,
)
from ..model.base import ModelArtifactConfig

LOG = structlog.stdlib.get_logger(__name__)


def get_new_request_state(
    request: Request, conversation_template: ConversationTemplate, tokenizer: TokenizerP
) -> RequestState:
    if request.debug_options.prompt is not None:
        prompt = request.debug_options.prompt
    else:
        prompt = conversation_template.apply(request.messages)

    prompt_tokens = tokenizer.encode(prompt)

    validation_err = None
    if request.validate_tokens is not None:
        validation_err = request.validate_tokens(request, prompt_tokens)

    gen_seqs = [
        GenerationSequence(
            seq_id=SequenceId(request.request_id, i),
            generated_token_ids=[],
            next_start_position=0,
            output_text="",
        )
        for i in range(request.num_sequences)
    ]

    return RequestState(
        request_id=request.request_id,
        prompt_token_ids=prompt_tokens,
        generation_sequences=gen_seqs,
        sampling_params=request.sampling_params,
        stopping_criteria=request.stopping_criteria,
        debug_options=request.debug_options,
        validation_err=validation_err,
        arrival_timestamp=time.time(),
    )


# Based on vllm: https://github.com/vllm-project/vllm/pull/984
def detokenize_incrementally(
    prompt_tokens: list[int],
    generation_sequence: GenerationSequence,
    tokenizer: TokenizerP,
    skip_special_tokens=False,
) -> str:
    new_token_id = generation_sequence.generated_token_ids[-1]

    # This is the first iteration for this sequence
    if generation_sequence.prev_tokens is None:
        # TODO(masahi): Figure out a way to remove this concat
        new_tokens = tokenizer.convert_ids_to_tokens(
            prompt_tokens + generation_sequence.generated_token_ids
        )
        output_tokens = new_tokens

        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_begin_offset = max(len(output_tokens) - 6, 0)

        if skip_special_tokens and new_token_id in tokenizer.all_special_ids:
            prefix_end_offset = max(len(output_tokens), 0)
        else:
            prefix_end_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens([new_token_id])
        output_tokens = generation_sequence.prev_tokens + new_tokens

        prefix_begin_offset = generation_sequence.prefix_begin_offset
        prefix_end_offset = generation_sequence.prefix_end_offset

    assert tokenizer.is_fast

    prefix_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_begin_offset:prefix_end_offset]
    )
    new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_begin_offset:])

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_prefix_begin_offset = prefix_end_offset
        new_prefix_end_offset = len(output_tokens)
        delta = new_text[len(prefix_text) :]
    else:
        new_prefix_begin_offset = prefix_begin_offset
        new_prefix_end_offset = prefix_end_offset
        delta = ""

    generation_sequence.prefix_begin_offset = new_prefix_begin_offset
    generation_sequence.prefix_end_offset = new_prefix_end_offset
    if generation_sequence.prev_tokens is None:
        generation_sequence.prev_tokens = new_tokens
    else:
        generation_sequence.prev_tokens.extend(new_tokens)

    return delta


def check_stopping_sequences(stopping_criteria, output_text, delta, is_ended):
    if stopping_criteria.stop_sequences:
        for t in stopping_criteria.stop_sequences:
            if t in output_text:
                # since search pattern can include only part of the new generated token,
                # we need to trim generated string
                # for example, we have "I " in the stopping criteria, previously existed
                # output_text had "I" and new coming token "am" would add space before the word
                # thus final output_text would have "I am" before verification on stop sequence
                # While eventually we need to return "I "
                if not output_text.endswith(t):
                    sub_index = output_text.find(t)
                    delta = delta[: -(len(output_text) - sub_index - len(t))]
                    output_text = output_text[: output_text.find(t) + len(t)]
                is_ended = True
                break
    return output_text, delta, is_ended


def update_sequence(
    gen_seq: GenerationSequence,
    new_token_ids: list[int],
    prompt_token_ids: list[int],
    tokenizer: TokenizerP,
    stopping_criteria: StoppingCriteria,
) -> str:
    gen_seq.next_start_position = len(prompt_token_ids) + len(
        gen_seq.generated_token_ids
    )
    gen_seq.generated_token_ids.extend(new_token_ids)
    delta = detokenize_incrementally(prompt_token_ids, gen_seq, tokenizer)
    gen_seq.output_text += delta

    gen_seq.output_text, delta, gen_seq.is_finished = check_stopping_sequences(
        stopping_criteria, gen_seq.output_text, delta, gen_seq.is_finished
    )

    return delta


def get_requests_to_process(
    current_states: list[RequestState], cache_manager: KVCacheManager
) -> Tuple[list[Union[PrefillRequest, DecodeRequest]], bool, int]:
    requests: list[Union[PrefillRequest, DecodeRequest]] = []
    # TODO: consider having hybrid batch if the underlying attention kernel supports
    # mixing prefill and decode.
    is_prompt_batch = any(
        state.generation_sequences[0].next_start_position == 0
        for state in current_states
    )

    token_counts = 0

    if is_prompt_batch:
        for state in current_states:
            if state.generation_sequences[0].next_start_position == 0:
                requests.append(
                    PrefillRequest(
                        request_id=state.request_id,
                        token_ids=state.prompt_token_ids,
                        num_sequence=state.num_sequences,
                        sampling_params=state.sampling_params,
                    )
                )

                token_counts += len(state.prompt_token_ids)

        LOG.debug(
            "Creating prompt batch.",
            num_requests=len(requests),
            total_tokens=token_counts,
        )
    else:
        for state in current_states:
            for gen_seq in state.generation_sequences:
                if not gen_seq.is_finished:
                    prompt_counts = len(state.prompt_token_ids)
                    requests.append(
                        DecodeRequest(
                            sequence_id=gen_seq.seq_id,
                            prompt_token_counts=prompt_counts,
                            token_ids=gen_seq.generated_token_ids,
                            sampling_params=state.sampling_params,
                        )
                    )
                    cache_manager.extend(
                        gen_seq.seq_id,
                        prompt_counts
                        + len(gen_seq.generated_token_ids)
                        - gen_seq.next_start_position,
                    )

        token_counts = len(requests)
        LOG.debug("Creating decode batch with %s requests.", token_counts)

    return requests, is_prompt_batch, token_counts


def should_stop_by_length(
    gen_seq: GenerationSequence,
    prompt_len: int,
    max_context_length: int,
    max_tokens: Optional[int],
) -> bool:
    # If max_tokens is None, we do not put any length restriction.
    if gen_seq.is_finished or max_tokens is None:
        return False

    num_context_tokens = prompt_len + len(gen_seq.generated_token_ids)

    if num_context_tokens >= max_context_length:
        return True

    num_gen_tokens = num_context_tokens - prompt_len

    if max_tokens and num_gen_tokens >= max_tokens:
        return True

    return False


class EngineBase:
    text_generator: TextGenerator
    tokenizer: TokenizerP
    model_artifact_config: ModelArtifactConfig
    max_context_length: int
    max_num_batched_tokens: int
    max_decode_steps: int
    min_decode_steps: int
    kv_cache_size: int
    max_prompt_len: int
    model_context_window_size: int
    queue_lock: Lock
    queue: Deque[RequestState]
    has_new_requests: Condition
    current_batch: Dict[RequestId, RequestState]

    def __init__(self, model_module: ModelModule):
        self.text_generator = model_module.text_generator
        self.tokenizer = model_module.tokenizer
        self.conversation_template = model_module.conversation_template
        self.cache_manager = model_module.cache_manager
        self.model_artifact_config = model_module.model_artifact_config
        assert (
            self.model_artifact_config.max_context_length
        ), "max_context_length must not be zero"
        self.max_context_length = self.model_artifact_config.max_context_length
        self.max_num_batched_tokens = model_module.engine_config.max_num_batched_tokens
        self.max_decode_steps = min(
            self.cache_manager.get_kv_cache_size(),
            model_module.engine_config.max_decode_steps,
        )
        self.min_decode_steps = min(
            self.max_decode_steps - 1, model_module.engine_config.min_decode_steps
        )
        self.kv_cache_size = self.cache_manager.get_kv_cache_size()
        self.max_prompt_len = min(self.max_context_length, self.max_num_batched_tokens)

        if self.model_artifact_config.sliding_window is not None:
            self.model_context_window_size = self.model_artifact_config.sliding_window
        else:
            self.model_context_window_size = self.max_context_length

        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)

        self.current_batch = dict[RequestId, RequestState]()

    def check_prompt_too_long(self, prompt_len: int, num_sequences: int = 1) -> bool:
        # We make sure that the KV cache will have enough free space for this request to proceed
        # decoding for at least self.max_decode_steps steps.
        #
        # For models using SWA, the number of consumed cache slots is upper bounded by the window
        # size. This assumes that the model implementation does not store past KV tensors beyond
        # the window into the cache.
        num_kv_slots_needed = min(prompt_len, self.model_context_window_size)
        return (
            prompt_len > self.max_prompt_len
            or (self.kv_cache_size - num_kv_slots_needed)
            < self.max_decode_steps * num_sequences
        )

    def evict_request(self, cancell_callback: Callable[[RequestId], None]) -> int:
        # Must be called with the queue lock held
        num_eviction = 0

        while self.cache_manager.get_max_new_tokens() < 1:
            num_eviction += 1

            single_sample_requests = []
            parallel_sample_requests = []

            for state in self.current_batch.values():
                if state.num_sequences == 1:
                    single_sample_requests.append(state)
                else:
                    parallel_sample_requests.append(state)

            if len(single_sample_requests) > 0:
                candidate_victims = single_sample_requests
            else:
                assert len(parallel_sample_requests) > 0
                candidate_victims = parallel_sample_requests

            request_to_remove = min(candidate_victims, key=lambda s: s.num_total_tokens)

            # TODO(masahi): Properly support evicting a multi-sequence request
            if self.current_batch[request_to_remove.request_id].num_sequences != 1:
                cancell_callback(request_to_remove.request_id)
                self.remove_request_from_batch(request_to_remove.request_id)
                LOG.warn(
                    "Preempting a multi-sequence request is currently not supported,"
                    f" cancelling request '{request_to_remove.request_id}'",
                )
                continue

            self.remove_request_from_batch(request_to_remove.request_id)
            self.queue.appendleft(request_to_remove)

            LOG.debug(
                "Preempt request to free %s tokens",
                request_to_remove.num_total_tokens,
            )

        return num_eviction

    def try_grow_batch(self, num_new_batched_tokens) -> Optional[int]:
        # Must be called with the queue lock held
        max_new_tokens = self.cache_manager.get_max_new_tokens()
        if max_new_tokens < self.min_decode_steps:
            LOG.debug(
                "Stop growing the batch due to min_decode_steps. Decode steps: %s",
                max_new_tokens,
            )
            # stop adding request if there isn't enough space to do self.min_decode_steps steps of decoding.
            return None

        state = self.queue[0]

        if state.num_sequences == 1:
            gen_seq = state.generation_sequences[0]
            num_tokens = state.prompt_len + len(gen_seq.generated_token_ids)
            num_new_batched_tokens += num_tokens
            # This can happen when we are recovering from cache eviction and the sum of prompt
            # and intermediate decode tokens is bigger than the biggest allowable batch size,
            # self.max_num_batched_tokens. In such cases, we need to discard the recent decode
            # tokens that cannot fit into a batch, and recompute them after we fill the cache
            # entries for the older tokens.
            if (
                len(self.current_batch) == 0
                and num_tokens > self.max_num_batched_tokens
            ):
                gen_seq.generated_token_ids = gen_seq.generated_token_ids[
                    : (self.max_num_batched_tokens - state.prompt_len)
                ]
                gen_seq.next_start_position = (
                    num_new_batched_tokens
                ) = num_tokens = self.max_num_batched_tokens
        else:
            # Evicting and recovering multi-sequence requests is not supported for now.
            assert all(
                gen_seq.next_start_position == 0
                for gen_seq in state.generation_sequences
            )
            num_tokens = state.prompt_len
            num_new_batched_tokens += num_tokens

        if num_new_batched_tokens > self.max_num_batched_tokens:
            LOG.debug(
                "Stop growing the batch due to max_num_batched_tokens. Batched tokens: %s",
                num_new_batched_tokens,
            )
            return None

        # We make sure that the KV cache will have enough free space for this request to proceed
        # decoding for at least self.max_decode_steps steps.
        # See the comment in check_prompt_too_long for the optimization involving the window size.
        num_kv_slots_needed = min(num_tokens, self.model_context_window_size)
        if (self.cache_manager.get_free_space() - num_kv_slots_needed) / (
            len(self.current_batch) + 1
        ) < self.max_decode_steps * state.num_sequences:
            LOG.debug(
                "Stop growing the batch due to not enough free space. Free: %s, Num tokens: %s",
                self.cache_manager.get_free_space(),
                num_tokens,
            )
            return None

        self.queue.popleft()
        # TODO parallel sampling: Need update here when evicting multi-sequence requests is supported.
        self.cache_manager.allocate(state.request_id, num_tokens, state.num_sequences)
        self.current_batch[state.request_id] = state

        return num_new_batched_tokens

    def remove_request_from_batch(self, request_id: RequestId):
        self.cache_manager.free_request(self.current_batch[request_id])
        del self.current_batch[request_id]
