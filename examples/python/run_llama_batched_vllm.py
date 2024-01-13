import argparse
import math
import os
import json
from collections import defaultdict
from typing import List
from dataclasses import dataclass

import numpy as np

import tvm
from tvm import relax
from tvm.runtime import disco as di

import torch
from transformers import AutoTokenizer

from mlc_llm.relax_model.llama import LlamaConfig
from mlc_llm import utils


class KVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_size, disco_session):
        if disco_session:
            init_cache_func = disco_session.get_global_func("tvm.contrib.vllm.allocate_kv_cache")
        else:
            init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

        self.cache = init_cache_func(head_size, num_layers, num_heads, block_size, num_blocks)

        self.block_tables = defaultdict(list)
        self.slot_mappings = defaultdict(list)
        self.block_size = block_size


class CacheManager:
    block_size: int = 16

    def __init__(
        self, num_blocks, num_layers, num_heads, head_size, disco_session=None, sliding_window=None
    ):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(
            num_blocks, self.block_size, num_layers, num_heads, head_size, disco_session
        )

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
                del self.kv_cache.slot_mappings[id]

            elif id in self.kv_cache.block_tables:
                # Decoding
                if len(self.kv_cache.block_tables[id]) < num_needed_block:
                    # Need to allocate a new block for this request
                    assert len(self.kv_cache.block_tables[id]) + 1 == num_needed_block
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

                pos = size - 1
                block_number = self.kv_cache.block_tables[id][-1]

                if self.block_sliding_window:
                    block_number = self.kv_cache.block_tables[id][
                        (pos // self.block_size) % self.block_sliding_window
                    ]
                else:
                    block_number = self.kv_cache.block_tables[id][-1]

                block_offset = pos % self.block_size
                slot = block_number * self.block_size + block_offset
                self.kv_cache.slot_mappings[id].append(slot)

            elif id not in self.kv_cache.block_tables:
                assert len(self.free_blocks) >= num_needed_block, "Not enough free blocks."

                for _ in range(num_needed_block):
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

                for i in range(size):
                    block_idx = i // self.block_size

                    if self.block_sliding_window:
                        block_idx %= self.block_sliding_window

                    block_number = self.kv_cache.block_tables[id][block_idx]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    self.kv_cache.slot_mappings[id].append(slot)

    def get(self):
        return self.kv_cache


@dataclass
class SequenceGenerationRequest:
    request_id: int
    token_ids: List[int]


@dataclass
class SequenceGenerationResponse:
    request_id: int
    token_id: int


@dataclass
class EvalQueryRequest:
    request_id: int
    num_past_tokens: int
    query_token_ids: List[int]


def sample(logits):
    logits = torch.from_dlpack(logits)
    return torch.argmax(logits, -1).cpu().numpy()


def load_params_disco(artifact_path, lib_path, num_shards):
    sess = di.ProcessSession(num_workers=num_shards)
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
    lib_path = os.path.join(artifact_path, f"{model}-{quantization}-cuda.so")

    if num_shards == 1:
        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)
        params = utils.load_params(artifact_path, dev)
        return vm.module, params, None

    return load_params_disco(artifact_path, lib_path, num_shards)


def _prepare_inputs(
    requests,
    all_slot_mappings,
    all_block_tables,
    sliding_window,
    dev,
    is_prefill,
):
    block_tables = []
    seq_lens = []
    input_ids = []
    slot_mapping = []
    positions = []
    max_num_blocks_per_seq = 0
    indices_within_window = []
    start_idx = 0

    for request in requests:
        request_id = request.request_id
        token_ids = request.token_ids

        if is_prefill:
            input_ids += token_ids
            prompt_len = len(token_ids)
            seq_lens.append(prompt_len)
            positions += range(prompt_len)
            slot_mapping += all_slot_mappings[request_id]

            if sliding_window:
                indices_within_window += range(
                    start_idx + max(0, prompt_len - sliding_window),
                    start_idx + prompt_len,
                )
                start_idx += prompt_len

        else:
            input_ids.append(token_ids[-1])
            pos = len(token_ids) - 1
            positions.append(pos)
            block_table = all_block_tables[request_id]
            max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
            block_tables.append(block_table)
            slot_mapping.append(all_slot_mappings[request_id][-1])

            if sliding_window:
                seq_lens.append(min(len(token_ids), sliding_window))
            else:
                seq_lens.append(len(token_ids))

    input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), dev)
    positions = tvm.nd.array(np.array(positions, dtype="int32"), dev)
    seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), dev)
    slot_mapping = tvm.nd.array(np.array(slot_mapping, dtype="int32"), dev)

    if is_prefill and sliding_window:
        indices_within_window = tvm.nd.array(np.array(indices_within_window, dtype="int32"), dev)
    else:
        indices_within_window = None

    if not is_prefill:

        def _pad_to_max(x: List[int], max_len: int) -> List[int]:
            return x + [0] * (max_len - len(x))

        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq) for block_table in block_tables
        ]

        block_tables_np = np.vstack(padded_block_tables).astype("int32")
        block_tables = tvm.nd.array(np.array(block_tables_np, dtype="int32"), dev)
    else:
        block_tables = None

    return (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        indices_within_window,
        block_tables,
    )


def _prepare_eval_queries(
    requests: List[EvalQueryRequest],
    all_slot_mappings,
    sliding_window,
    dev,
):
    seq_lens = []
    query_lens = []
    input_ids = []
    slot_mapping = []
    past_slot_mapping = []
    positions = []
    permute_map = []

    query_offset = sum([request.num_past_tokens for request in requests])
    past_offset = 0

    for request in requests:
        num_past_tokens = request.num_past_tokens
        num_queries = len(request.query_token_ids)
        query_lens.append(num_queries)
        request_id = request.request_id
        input_ids += request.query_token_ids

        positions += [num_past_tokens + i for i in range(num_queries)]

        if sliding_window and num_past_tokens + num_queries >= sliding_window:
            seq_lens.append(sliding_window)
            past_slot_mapping += all_slot_mappings[request_id][
                num_past_tokens - (sliding_window - num_queries) : num_past_tokens
            ]
        else:
            seq_lens.append(num_past_tokens + num_queries)
            past_slot_mapping += all_slot_mappings[request_id][:num_past_tokens]

        slot_mapping += all_slot_mappings[request_id][
            num_past_tokens : num_past_tokens + num_queries
        ]

        permute_map += list(range(past_offset, past_offset + num_past_tokens)) + list(
            range(query_offset, query_offset + num_queries)
        )

        query_offset += num_queries
        past_offset += num_past_tokens

    input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), dev)
    positions = tvm.nd.array(np.array(positions, dtype="int32"), dev)
    seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), dev)
    slot_mapping = tvm.nd.array(np.array(slot_mapping, dtype="int32"), dev)

    query_lens = tvm.nd.array(np.array(query_lens, dtype="int32"), dev)
    past_slot_mapping = tvm.nd.array(np.array(past_slot_mapping, dtype="int32"), dev)
    permute_map = tvm.nd.array(np.array(permute_map, dtype="int32"), dev)

    return (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        query_lens,
        past_slot_mapping,
        permute_map,
    )


class Model:
    def __init__(
        self, artifact_path, model_name, quant, vocab_size, num_shards, dev, sliding_window
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

    def generate(
        self, requests: List[SequenceGenerationRequest], cache: KVCache, is_prefill: bool
    ) -> List[SequenceGenerationResponse]:
        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            indices_within_window,
            block_tables,
        ) = _prepare_inputs(
            requests,
            cache.slot_mappings,
            cache.block_tables,
            self.sliding_window,
            self.dev,
            is_prefill,
        )

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        kv_cache = cache.cache

        if is_prefill:
            if self.sliding_window:
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
                logits = out[0]  # Ignore returned KV cache since it is updated in-place anyway.
        else:
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

        next_tokens = sample(logits)

        return [
            SequenceGenerationResponse(request.request_id, new_token)
            for request, new_token in zip(requests, next_tokens)
        ]


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft --use-cache=0 --max-seq-len 768 --enable-batching --use-vllm-attention
    # python examples/python/run_llama_batched_vllm.py --local-id vicuna-v1-7b-q4f16_ft
    #
    # For Disco:
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768 --enable-batching --use-vllm-attention --build-model-only --num-shards 2
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768 --enable-batching --use-vllm-attention --convert-weight-only
    # CUDA_VISIBLE_DEVICES=0,1 python examples/python/run_llama_batched_vllm.py --local-id vicuna-v1-7b-q0f16 --num-shards 2

    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--num-shards", type=int, default=1)
    args.add_argument("--num-decode-steps", type=int, default=20)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    return parsed


def run(args):
    quantization = args.quantization.name
    artifact_path = args.artifact_path
    model_name = args.model
    model_path = f"dist/models/{model_name}"

    dev = tvm.device("cuda", 0)

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
        config = LlamaConfig(**json.load(i_f))

    model = Model(
        artifact_path,
        model_name,
        quantization,
        config.vocab_size,
        args.num_shards,
        dev,
        config.sliding_window,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    num_kv_heads = config.get_num_key_value_heads() // args.num_shards
    head_size = config.hidden_size // config.num_attention_heads
    num_blocks = 500

    cache_manager = CacheManager(
        num_blocks,
        config.num_hidden_layers,
        num_kv_heads,
        head_size,
        model.disco_session,
        sliding_window=config.sliding_window,
    )
    cache = cache_manager.get()

    model.block_sliding_window = cache_manager.block_sliding_window

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    batched_token_ids = [tokenizer.encode(p) for p in prompts]
    prompts_len = [len(ids) for ids in batched_token_ids]
    request_ids = list(range(len(prompts)))
    target_sizes = []
    requests = []

    for token_ids, request_id in zip(batched_token_ids, request_ids):
        request_ids.append(request_id)
        target_sizes.append(len(token_ids))
        requests.append(SequenceGenerationRequest(request_id, token_ids))

    cache_manager.set_size(request_ids, target_sizes)

    out = model.generate(requests, cache, True)

    for _ in range(args.num_decode_steps):
        for i, response in enumerate(out):
            new_token_id = response.token_id
            requests[i].token_ids.append(new_token_id)
            target_sizes[i] += 1

        cache_manager.set_size(request_ids, target_sizes)

        out = model.generate(requests, cache, False)

    output_tokens = [
        tokenizer.convert_ids_to_tokens(
            requests[i].token_ids[prompts_len[i] :], skip_special_tokens=True
        )
        for i in range(len(requests))
    ]

    generated = [tokenizer.convert_tokens_to_string(tokens) for tokens in output_tokens]

    for p, g in zip(prompts, generated):
        print("Prompt = '{}', generated text = '{}'".format(p, g))

    query_token_lens = [4, 3, 5, 2]

    eval_query_requests = []

    for request_id, query_token_len in zip(request_ids, query_token_lens):
        queries_to_eval = requests[request_id].token_ids[-query_token_len:]
        num_past = len(requests[request_id].token_ids) - query_token_len
        eval_query_requests.append(EvalQueryRequest(request_id, num_past, queries_to_eval))

    (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        query_lens,
        past_slot_mapping,
        permute_map,
    ) = _prepare_eval_queries(
        eval_query_requests,
        cache.slot_mappings,
        None,
        model.dev,
    )

    logits = model.mod["evaluate_multi_query"](
        input_ids,
        positions,
        seq_lens,
        cache.cache,
        slot_mapping,
        query_lens,
        past_slot_mapping,
        permute_map,
        model.params,
    )[0].numpy()

    assert logits.shape[0] == sum(query_token_lens)

    logits_offset = 0

    for request_id, query_token_len in zip(request_ids, query_token_lens):
        for i in range(query_token_len - 1):
            # requests[request_id].token_ids[-query_token_len:] are the "ground truth" tokens.
            # Doing argmax over multi-timestep logits computed in parallel should yield the same
            # tokens at the corresponding positions.
            past_tokens = requests[request_id].token_ids[:-query_token_len]
            assert (
                np.argmax(logits[logits_offset + i])
                == requests[request_id].token_ids[len(past_tokens) + i + 1]
            )

        logits_offset += query_token_len


if __name__ == "__main__":
    run(parse_args())
