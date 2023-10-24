import json
import logging
import math
import os
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
from tvm.runtime import disco as di

from mlc_llm import utils
from mlc_llm.relax_model.llama import LlamaConfig

from ..engine import ChatMessage, RequestId, SamplingType
from ..engine.model_module import (
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    SequenceId,
    TextGenerationResult,
)

logger = logging.getLogger(__name__)


class KVCache:
    def __init__(
        self, num_blocks, block_size, num_layers, num_heads, head_size, disco_session
    ):
        if disco_session:
            init_cache_func = disco_session.get_global_func(
                "tvm.contrib.vllm.allocate_kv_cache"
            )
        else:
            init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

        self.cache = init_cache_func(
            head_size, num_layers, num_heads, block_size, num_blocks
        )

        self.block_tables = defaultdict(list)
        self.block_size = block_size


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
        num_blocks,
        num_layers,
        num_heads,
        head_size,
        disco_session=None,
        sliding_window=None,
    ):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(
            num_blocks, self.block_size, num_layers, num_heads, head_size, disco_session
        )
        self.allocated_tokens = dict[RequestId, int]()

        if sliding_window:
            assert sliding_window % self.kv_cache.block_size == 0
            self.block_sliding_window = sliding_window // self.kv_cache.block_size
        else:
            self.block_sliding_window = None

    def set_size(self, request_ids: List[int], target_sizes: List[int]):
        for id, size in zip(request_ids, target_sizes):
            num_needed_block = math.ceil(size / self.block_size)

            if self.block_sliding_window:
                num_needed_block = min(num_needed_block, self.block_sliding_window)

            if id in self.kv_cache.block_tables and size == 0:
                self.free_blocks.extend(self.kv_cache.block_tables[id])
                del self.kv_cache.block_tables[id]

            elif (
                id in self.kv_cache.block_tables
                and len(self.kv_cache.block_tables[id]) < num_needed_block
            ):
                # Decoding, need to allocate a new block for this request
                assert len(self.kv_cache.block_tables[id]) + 1 == num_needed_block
                self.kv_cache.block_tables[id].append(self.free_blocks.pop())

            elif id not in self.kv_cache.block_tables:
                assert (
                    len(self.free_blocks) >= num_needed_block
                ), "Not enough free blocks."

                for _ in range(num_needed_block):
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

    def get_cache(self):
        return self.kv_cache

    def allocate(self, request_id: RequestId, num_tokens: int):
        """
        Allocate cache space for request, raise error if there is no space.
        """
        self.set_size([request_id], [num_tokens])
        self.allocated_tokens[request_id] = num_tokens

    def extend(self, sequence_id: SequenceId, new_tokens: int):
        """
        Extend cache space for a sequence, raise error if there is no space.
        """
        assert sequence_id.sequence_index == 0, "multiple sequences not supported"
        request_id = sequence_id.request_id
        allocated = self.allocated_tokens[request_id]
        self.set_size([request_id], [allocated + new_tokens])
        self.allocated_tokens[request_id] += new_tokens

    def free(self, sequence_id: SequenceId):
        """
        Free cache space for a sequence or all sequences of a request.
        """
        assert sequence_id.sequence_index == 0, "multiple sequences not supported"
        request_id = sequence_id.request_id
        del self.allocated_tokens[request_id]
        self.set_size([request_id], [0])

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
        if not self.allocated_tokens:
            return len(self.free_blocks) * self.block_size

        free_blocks_per_request = len(self.free_blocks) // len(self.allocated_tokens)
        remaining_blocks = len(self.free_blocks) - free_blocks_per_request * len(
            self.allocated_tokens
        )
        remaining_tokens_in_last_block = [
            self.block_size - (tokens - 1) % self.block_size - 1
            for _, tokens in self.allocated_tokens.items()
        ]

        return (
            free_blocks_per_request * self.block_size
            + sorted(remaining_tokens_in_last_block)[remaining_blocks]
        )


def _apply_top_p_top_k(logits, top_ps, top_ks):
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


def sample(logits, sampling_params, vocab_size):
    logits = torch.from_dlpack(logits)
    # TODO: Support beam search?
    do_greedy = [p.sampling_type == SamplingType.GREEDY for p in sampling_params]
    # TODO: Support per-type batched sampling like vllm.
    assert all(do_greedy) or all([not greedy for greedy in do_greedy])

    temperatures = [p.temperature for p in sampling_params]
    if any(t != 1.0 for t in temperatures):
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits.div_(t.unsqueeze(dim=1))

    top_ps = [p.top_p for p in sampling_params]
    top_ks = [p.top_k if p.top_k != -1 else vocab_size for p in sampling_params]

    do_top_p = any(p < 1.0 for p in top_ps)
    do_top_k = any(k != vocab_size for k in top_ks)

    if do_top_p or do_top_k:
        logits = _apply_top_p_top_k(logits, top_ps, top_ks)

    if all(do_greedy):
        return torch.argmax(logits, -1).cpu().numpy()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, True).cpu().numpy()[:, 0]


def load_disco_module(artifact_path, lib_path, num_shards):
    sess = di.ThreadedSession(num_workers=num_shards)
    devices = range(num_shards)
    sess.init_ccl("nccl", *devices)
    module = sess.load_vm_module(lib_path)

    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    metadata_path = os.path.join(artifact_path, "params", "ndarray-cache.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        ndarray_cache_metadata = f.read()

    loader = loader_create(metadata_path, ndarray_cache_metadata, "", module)
    loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
    params = loader_load(loader)

    return module, params, sess


def copy_to_worker_0(sess: di.Session, host_array):
    x_array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def get_tvm_model(artifact_path, model, quantization, num_shards, dev):
    model_artifact_path = os.path.join(artifact_path, f"{model}-{quantization}-batched")
    lib_path = os.path.join(model_artifact_path, f"{model}-{quantization}-cuda.so")

    if num_shards == 1:
        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)
        params = utils.load_params(model_artifact_path, dev)
        return vm.module, params, None

    return load_disco_module(model_artifact_path, lib_path, num_shards)


class Model:
    def __init__(
        self,
        artifact_path,
        model_name,
        quant,
        vocab_size,
        num_shards,
        dev,
        sliding_window=None,
    ):
        self.mod, self.params, self.disco_session = get_tvm_model(
            artifact_path, model_name, quant, num_shards, dev
        )
        self.dev = dev
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window

        if sliding_window:
            self.block_sliding_window = sliding_window // CacheManager.block_size
        else:
            self.block_sliding_window = None

    def get_used_memory(self):
        if self.disco_session:
            params = self.params.debug_get_from_remote(0)

            get_used_memory_func = self.disco_session.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            # For Disco, we explicitly query the device 0.
            peak_memory = get_used_memory_func(
                tvm.device("cuda", 0)
            ).debug_get_from_remote(0)

        else:
            params = self.params

            get_used_memory_func = tvm.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            peak_memory = get_used_memory_func(self.dev)

        param_bytes = 0

        for param in params:
            param_bytes += param.numpy().nbytes

        return peak_memory + param_bytes

    def profile_memory_usage(self, seq_lens):
        input_ids = [0] * sum(seq_lens)
        positions = []

        for s in seq_lens:
            positions += range(s)

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)

        self.mod["evaluate"](input_ids, positions, seq_lens, self.params)

        return self.get_used_memory()

    def generate(
        self,
        requests: List[Union[PrefillRequest, DecodeRequest]],
        cache: KVCache,
        is_prefill: bool,
    ) -> List[TextGenerationResult]:
        block_tables = []
        seq_lens = []
        input_ids = []
        slot_mapping = []
        positions = []
        max_num_blocks_per_seq = 0
        block_size = cache.block_size
        sampling_params = []
        sequence_ids = []
        indices_within_window = []

        start_idx = 0

        for request in requests:
            if isinstance(request, PrefillRequest):
                assert request.num_sequence == 1, "Multiple sequences not supported yet"
                sequence_id = SequenceId(request.request_id, 0)
                sequence_ids.append(sequence_id)
            else:
                sequence_id = request.sequence_id
                sequence_ids.append(request.sequence_id)

            block_table = cache.block_tables[sequence_id.request_id]
            sampling_params.append(request.sampling_params)

            if is_prefill:
                input_ids += request.token_ids
                prompt_len = len(request.token_ids)
                seq_lens.append(prompt_len)
                positions += range(prompt_len)

                if self.sliding_window:
                    indices_within_window += range(
                        start_idx + max(0, prompt_len - self.sliding_window),
                        start_idx + prompt_len,
                    )
                    start_idx += prompt_len

                for i in range(len(request.token_ids)):
                    if self.sliding_window:
                        block_number = block_table[
                            (i // block_size) % self.block_sliding_window
                        ]
                    else:
                        block_number = block_table[i // block_size]

                    block_offset = i % block_size
                    slot = block_number * block_size + block_offset
                    slot_mapping.append(slot)
            else:
                input_ids.append(request.token_ids[-1])
                pos = len(request.token_ids) - 1
                positions.append(pos)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
                block_tables.append(block_table)

                if self.sliding_window:
                    seq_lens.append(min(len(request.token_ids), self.sliding_window))
                    block_number = block_table[
                        (pos // block_size) % self.block_sliding_window
                    ]
                else:
                    block_number = block_table[pos // block_size]
                    seq_lens.append(len(request.token_ids))

                block_offset = pos % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)

        input_ids_np = np.array(input_ids, dtype="int32")
        input_ids = tvm.nd.array(input_ids_np, self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)
        slot_mapping = tvm.nd.array(np.array(slot_mapping, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        kv_cache = cache.cache

        if is_prefill:
            torch.cuda.nvtx.range_push(f"forward prefill {input_ids_np.shape}")

            if self.sliding_window:
                indices_within_window = tvm.nd.array(
                    np.array(indices_within_window, dtype="int32"), self.dev
                )

                if self.disco_session:
                    indices_within_window = copy_to_worker_0(
                        self.disco_session, indices_within_window
                    )

                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    kv_cache,
                    slot_mapping,
                    indices_within_window,
                    self.params,
                )
            else:
                out = self.mod["prefill"](
                    input_ids, positions, seq_lens, kv_cache, slot_mapping, self.params
                )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[
                    0
                ]  # Ignore returned KV cache since it is updated in-place anyway.
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_ids_np.shape}")

            def _pad_to_max(x: List[int], max_len: int) -> List[int]:
                return x + [0] * (max_len - len(x))

            padded_block_tables = [
                _pad_to_max(block_table, max_num_blocks_per_seq)
                for block_table in block_tables
            ]

            block_tables_np = np.vstack(padded_block_tables).astype("int32")
            block_tables = tvm.nd.array(
                np.array(block_tables_np, dtype="int32"), self.dev
            )

            if self.disco_session:
                block_tables = copy_to_worker_0(self.disco_session, block_tables)

            out = self.mod["decode"](
                input_ids,
                positions,
                seq_lens,
                kv_cache,
                slot_mapping,
                block_tables,
                self.params,
            )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[0]

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        next_tokens = sample(logits, sampling_params, self.vocab_size)

        return [
            TextGenerationResult(
                sequence_id=sequence_id,
                generated_tokens=[new_token],
                error=None,
            )
            for sequence_id, new_token in zip(sequence_ids, next_tokens)
        ]


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_num_cache_blocks(
    model,
    seq_lens,
    num_layers,
    num_kv_heads,
    head_size,
    gpu_memory_utilization=0.9,  # the default used by vllm
):
    used_memory_bytes = model.profile_memory_usage(seq_lens)
    cache_block_size = CacheManager.get_cache_block_size(
        num_layers, num_kv_heads, head_size
    )
    total_vram = get_gpu_memory()
    return int(
        (total_vram * gpu_memory_utilization - used_memory_bytes) // cache_block_size
    )


class PagedCacheModelTextGenerator:
    def __init__(self, model: Model):
        self.model = model

    def generate(
        self, requests: list[Union[PrefillRequest, DecodeRequest]], kv_cache
    ) -> list[TextGenerationResult]:
        prefill_requests = [r for r in requests if isinstance(r, PrefillRequest)]
        decode_requests = [r for r in requests if isinstance(r, DecodeRequest)]

        out = []
        if prefill_requests:
            out.extend(self.model.generate(prefill_requests, kv_cache, is_prefill=True))
        if decode_requests:
            out.extend(self.model.generate(decode_requests, kv_cache, is_prefill=False))

        return out


class Tokenizer:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.eos_token_id = self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class ConversationTemplate:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def apply(self, messages: list[ChatMessage]) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


class PagedCacheModelModule:
    def __init__(
        self,
        model_name: str,
        artifact_path: str,
        quantization: str,
        num_shards: int,
        max_num_batched_tokens: int = 0,
        max_input_len: int = 0,
    ):
        model_path = os.path.join(artifact_path, "models", model_name)

        dev = tvm.device("cuda", 0)

        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
            config = LlamaConfig(**json.load(i_f))

        model = Model(
            artifact_path,
            model_name,
            quantization,
            config.vocab_size,
            num_shards,
            dev,
            config.sliding_window,
        )

        if num_shards > 1:
            model.disco_session.sync_worker_0()

        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )

        num_kv_heads = config.get_num_key_value_heads() // num_shards
        head_size = config.hidden_size // config.num_attention_heads

        if max_num_batched_tokens > 0:
            assert max_input_len > 0
            assert max_num_batched_tokens % max_input_len == 0  # for simplicity

            num_seqs = max_num_batched_tokens // max_input_len
            num_blocks = get_num_cache_blocks(
                model,
                [max_input_len] * num_seqs,
                config.num_hidden_layers,
                num_kv_heads,
                head_size,
            )
        else:
            num_blocks = 500

        logger.info(f"Using {num_blocks} cache blocks.")

        cache_manager = CacheManager(
            num_blocks,
            config.num_hidden_layers,
            num_kv_heads,
            head_size,
            model.disco_session,
        )

        self.text_generator = PagedCacheModelTextGenerator(model)
        self.cache_manager = cache_manager
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.conversation_template = ConversationTemplate(hf_tokenizer)
