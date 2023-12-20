import math
from collections import defaultdict
from typing import List, Optional

from ..engine import (
    RequestId,
    SequenceId,
    RequestState,
    get_prompt_sequence_id,
)


# The following implementation has become complicated mostly to support parallel sampling and
# SWA. vLLM uses more redundant representaion of block tables (prompt blocks are duplicated
# among all sequences, and circular buffering is realized by appending a block at index
# (pos // block_size) % block_sliding_window to the end of the block table list), but thanks
# to that parallel sampling with SWA was automatically supported when they introduced
# Mistral / SWA support.
#
# TODO(masahi): Consider adopting their representation if ours turns out to be too buggy (hopefully not)


class DecodeBlockTable:
    def __init__(
        self,
        prompt_blocks: list[int],
        num_prompt_tokens: int,
        block_size: int,
        block_sliding_window: Optional[int] = None,
        prompt_shared: bool = False,
    ):
        self.num_prompt_blocks = len(prompt_blocks)
        self.prompt_blocks = prompt_blocks  # immutable
        self.block_sliding_window = block_sliding_window
        self.prompt_shared = prompt_shared

        # Prompt blocks between [prompt_cursor, prompt_cursor_tail) are shared
        # with other sequences in a parallel-sampling request.

        if (
            self.block_sliding_window
            and self.num_prompt_blocks >= self.block_sliding_window
            and prompt_shared
        ):
            self.prompt_cursor = (
                num_prompt_tokens // block_size
            ) % block_sliding_window
            self.prompt_cursor_tail = self.prompt_cursor
        else:
            self.prompt_cursor = 0
            self.prompt_cursor_tail = self.num_prompt_blocks

        self.decode_blocks: list[int] = []

    def append(self, new_block_id: int):
        self.decode_blocks.append(new_block_id)

    def __len__(self):
        return self.num_prompt_blocks + len(self.decode_blocks)

    def __getitem__(self, index: int) -> int:
        if index == -1:
            if len(self.decode_blocks) == 0:
                return self.prompt_blocks[-1]

            return self.decode_blocks[-1]

        assert index >= 0

        if index < self.num_prompt_blocks:
            return self.prompt_blocks[index]

        return self.decode_blocks[index - self.num_prompt_blocks]

    def get_blocks(self) -> list[int]:
        if not self.block_sliding_window or not self.prompt_shared:
            return self.prompt_blocks + self.decode_blocks

        if self.prompt_cursor <= self.prompt_cursor_tail:
            return (
                self.prompt_blocks[self.prompt_cursor : self.prompt_cursor_tail]
                + self.decode_blocks
            )

        return (
            self.prompt_blocks[self.prompt_cursor :]
            + self.prompt_blocks[: self.prompt_cursor_tail]
            + self.decode_blocks
        )

    def replace_head_prompt_block_with(self, new_block):
        assert self.prompt_shared

        self.append(new_block)
        self.prompt_cursor += 1
        self.prompt_cursor %= self.num_prompt_blocks
        self.num_prompt_blocks -= 1

        if self.prompt_cursor == self.prompt_cursor_tail:
            # No more prompt blocks to be shared
            self.prompt_shared = False


class KVCache:
    def __init__(
        self,
        cache_blocks,
        block_size,
    ):
        self.cache_blocks = cache_blocks
        self.block_size = block_size

        # SequenceId -> list[int]
        self.prompt_block_tables = defaultdict(list)
        self.slot_mappings = defaultdict(list)

        # The core data structure
        self.decode_block_tables = dict[SequenceId, DecodeBlockTable]()

        # Record indices of blocks to copy after prefill in the format [src1, dst1, src2, dst2, ...]
        self.pending_copy_from_to: list[int] = []


class CacheManager:
    block_size: int = 16

    @staticmethod
    def get_cache_block_size(num_layers, num_heads, head_size):
        # Taken from vllm/worker/cache_engine.py
        key_cache_block = CacheManager.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = 2  # fp16
        return dtype_size * total

    def __init__(
        self,
        cache_blocks,  # This can be any type
        num_blocks: int,
        sliding_window: Optional[int] = None,
    ):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(cache_blocks, self.block_size)
        self.token_counts = dict[SequenceId, int]()

        if sliding_window:
            assert sliding_window % self.kv_cache.block_size == 0
            self.block_sliding_window = sliding_window // self.kv_cache.block_size
        else:
            self.block_sliding_window = None

        self.sliding_window = sliding_window

    def set_size(self, sequence_ids: List[SequenceId], target_sizes: List[int]):
        for id, size in zip(sequence_ids, target_sizes):
            num_needed_block = math.ceil(size / self.block_size)

            if self.block_sliding_window:
                num_needed_block = min(num_needed_block, self.block_sliding_window)

            if size == 0:
                if id in self.kv_cache.prompt_block_tables:
                    self.free_blocks.extend(self.kv_cache.prompt_block_tables[id])
                    del self.kv_cache.prompt_block_tables[id]
                elif id in self.kv_cache.decode_block_tables:
                    self.free_blocks.extend(
                        self.kv_cache.decode_block_tables[id].decode_blocks
                    )
                    del self.kv_cache.decode_block_tables[id]

                if id in self.kv_cache.slot_mappings:
                    del self.kv_cache.slot_mappings[id]

            elif id in self.kv_cache.decode_block_tables:
                decode_block_table = self.kv_cache.decode_block_tables[id]

                if len(decode_block_table) < num_needed_block:
                    # Need to allocate a new block for this request
                    assert len(decode_block_table) + 1 == num_needed_block
                    assert len(self.free_blocks) > 0
                    decode_block_table.append(self.free_blocks.pop())

                pos = size - 1

                def get_block_circular_index(token_pos):
                    assert self.block_sliding_window
                    return (token_pos // self.block_size) % self.block_sliding_window

                if (
                    decode_block_table.prompt_shared
                    and self.sliding_window
                    and size >= self.sliding_window
                ):
                    # Parallel sampling + SWA case
                    if decode_block_table.prompt_cursor == get_block_circular_index(
                        pos
                    ):
                        # This sequence is trying to overwrite a prompt block shared with other sequences.
                        assert (
                            len(self.free_blocks) > 0
                        ), "No more free block in the cache."

                        block_number = self.free_blocks.pop()
                        # Add a new decode block and advance the prompt cursor
                        decode_block_table.replace_head_prompt_block_with(block_number)
                    else:
                        # Write to the decode block allocated above
                        block_number = decode_block_table[-1]

                else:
                    if self.block_sliding_window:
                        index = get_block_circular_index(pos)
                    else:
                        index = -1

                    block_number = decode_block_table[index]

                block_offset = pos % self.block_size
                slot = block_number * self.block_size + block_offset
                self.kv_cache.slot_mappings[id].append(slot)

            elif id not in self.kv_cache.prompt_block_tables:
                assert (
                    len(self.free_blocks) >= num_needed_block
                ), "Not enough free blocks."

                for _ in range(num_needed_block):
                    self.kv_cache.prompt_block_tables[id].append(self.free_blocks.pop())

                for block_idx in range(math.floor(size / self.block_size)):
                    if self.block_sliding_window:
                        block_idx %= self.block_sliding_window

                    block_number = self.kv_cache.prompt_block_tables[id][block_idx]
                    slots = [
                        block_number * self.block_size + block_offset
                        for block_offset in range(self.block_size)
                    ]
                    self.kv_cache.slot_mappings[id] += slots

                for i in range(len(self.kv_cache.slot_mappings[id]), size):
                    block_idx = i // self.block_size

                    if self.block_sliding_window:
                        block_idx %= self.block_sliding_window

                    block_number = self.kv_cache.prompt_block_tables[id][block_idx]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    self.kv_cache.slot_mappings[id].append(slot)

    def get_cache(self):
        return self.kv_cache

    def allocate(self, request_id: RequestId, num_tokens: int, num_sequences: int):
        """
        Allocate cache space for a prefill request, raise error if there is no space.
        """
        prompt_seq_id = get_prompt_sequence_id(request_id)
        self.set_size([prompt_seq_id], [num_tokens])

        last_block_partially_shared = num_sequences > 1 and (
            num_tokens % self.block_size != 0
        )

        if self.sliding_window:
            last_block_partially_shared &= num_tokens < self.sliding_window

        prompt_blocks = self.kv_cache.prompt_block_tables[prompt_seq_id]
        assert prompt_blocks

        prompt_shared = num_sequences > 1

        for i in range(num_sequences):
            decode_seq_id = SequenceId(request_id, i)
            self.token_counts[decode_seq_id] = num_tokens

            if not last_block_partially_shared:
                self.kv_cache.decode_block_tables[decode_seq_id] = DecodeBlockTable(
                    prompt_blocks,
                    num_tokens,
                    self.block_size,
                    self.block_sliding_window,
                    prompt_shared,
                )
            else:
                if i < num_sequences:
                    # Need to copy the last block in self.kv_cache.block_tables[prompt_seq_id]
                    self.kv_cache.decode_block_tables[decode_seq_id] = DecodeBlockTable(
                        prompt_blocks[:-1],
                        num_tokens,
                        self.block_size,
                        self.block_sliding_window,
                        prompt_shared,
                    )
                    last_block_copy = self.free_blocks.pop()
                    self.kv_cache.decode_block_tables[decode_seq_id].append(
                        last_block_copy
                    )
                    self.kv_cache.pending_copy_from_to.extend(
                        [prompt_blocks[-1], last_block_copy]
                    )
                else:
                    # The last sequence can directly overwrite the last block without copying it,
                    # since other sequences have its own copy of the last block.
                    self.kv_cache.decode_block_tables[decode_seq_id] = DecodeBlockTable(
                        prompt_blocks,
                        num_tokens,
                        self.block_size,
                        self.block_sliding_window,
                        prompt_shared,
                    )

    def extend(self, sequence_id: SequenceId, new_tokens: int):
        """
        Extend cache space for a sequence, raise error if there is no space.
        """
        allocated = self.token_counts[sequence_id]
        self.set_size([sequence_id], [allocated + new_tokens])
        self.token_counts[sequence_id] += new_tokens

    def free(self, sequence_id: SequenceId):
        """
        Free cache space for a sequence in a request.
        """
        if sequence_id in self.token_counts:
            del self.token_counts[sequence_id]
            self.set_size([sequence_id], [0])

    def free_request(self, state: RequestState):
        """
        Free cache space for all sequences in a request.
        """
        for gen_seq in state.generation_sequences:
            self.free(gen_seq.seq_id)

        prompt_seq_id = get_prompt_sequence_id(state.request_id)
        self.set_size([prompt_seq_id], [0])

    def get_kv_cache_size(self) -> int:
        """
        Return the size of the cache, in number of tokens.
        """
        return self.num_blocks * self.block_size

    def get_free_space(self) -> int:
        """
        Get available space of the cache.
        Return number of tokens that can be allocated for a new request.

        For paged KV cache, this ignores the remaining tokens in pages allocated
        for existing sequences, since they cannot be used for the new request.
        """
        return len(self.free_blocks) * self.block_size

    def get_max_new_tokens(self) -> int:
        """
        Get the maximum number of new tokens that can be extended for
        all sequences in the cache.

        For example, if the cache size is 16 tokens, with page size 1, and
        there are 3 sequences in the cache, each of them have 3 tokens cached,
        this method should return 2.

        It should return the result of `get_kv_cache_size` if there is
        no requests in the cache.
        """
        if not self.token_counts:
            return len(self.free_blocks) * self.block_size

        free_blocks_per_sequence = len(self.free_blocks) // len(self.token_counts)
        remaining_blocks = len(self.free_blocks) - free_blocks_per_sequence * len(
            self.token_counts
        )
        # For parallel sampling, the number of shared prompt tokens is divisible
        # by self.block_size (since the remainers are copied to each sequence).
        # So the following calculation does not overcount shared prompt tokens.
        remaining_tokens_in_last_block = [
            self.block_size - (tokens - 1) % self.block_size - 1
            for tokens in self.token_counts.values()
        ]

        return (
            free_blocks_per_sequence * self.block_size
            + sorted(remaining_tokens_in_last_block)[remaining_blocks]
        )
