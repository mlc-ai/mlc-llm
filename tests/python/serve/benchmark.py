# pylint: disable=import-error,line-too-long,missing-docstring,no-member,too-many-locals
# type: ignore
import argparse
import json
import os
import random
import time
from typing import Any, Callable, List, Tuple

import numpy as np
from transformers import AutoTokenizer

from mlc_chat.serve import Engine, GenerationConfig, KVCacheConfig
from mlc_chat.serve.engine import ModelInfo


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-lib-path", type=str, required=True)
    # Download dataset from
    # https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--num-prompts", type=int, default=500)
    args.add_argument("--batch-size", type=int, default=80)
    args.add_argument("--page-size", type=int, default=16)
    args.add_argument("--max-total-seq-length", type=int)
    args.add_argument("--seed", type=int, default=0)

    parsed = args.parse_args()
    parsed.model = os.path.dirname(parsed.model_lib_path)
    assert parsed.batch_size % 16 == 0
    assert parsed.page_size == 16
    return parsed


def sample_requests(
    dataset_path: str, num_requests: int, model_path: str
) -> Tuple[List[str], List[GenerationConfig]]:
    """Sample requests from dataset.
    Acknowledgement to the benchmark scripts in the vLLM project.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)

    # Construct generation config.
    prompts = [prompt for prompt, _, _ in sampled_requests]
    generation_config_list = [
        GenerationConfig(temperature=1.0, top_p=1.0, max_tokens=output_len)
        for _, _, output_len in sampled_requests
    ]
    return prompts, generation_config_list


def time_evaluator(func: Callable, args: List[Any], num_runs: int = 3):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def benchmark(args: argparse.Namespace):
    random.seed(args.seed)

    # Initialize model loading info and KV cache config
    model = ModelInfo(args.model, args.model_lib_path, args.device)
    kv_cache_config = KVCacheConfig(
        page_size=args.page_size,
        max_num_sequence=args.batch_size,
        max_total_sequence_length=args.max_total_seq_length,
    )

    # Create engine
    engine = Engine(model, kv_cache_config)
    # Sample prompts from dataset
    prompts, generation_config = sample_requests(args.dataset, args.num_prompts, args.model)
    # Engine statistics
    num_runs = 1
    single_token_prefill_latency = []
    single_token_decode_latency = []
    engine_total_prefill_time = []
    engine_total_decode_time = []
    total_prefill_tokens = []
    total_decode_tokens = []

    def engine_generate():
        engine.reset()
        engine.generate(prompts, generation_config)
        engine_stats = engine.stats()
        single_token_prefill_latency.append(engine_stats["single_token_prefill_latency"])
        single_token_decode_latency.append(engine_stats["single_token_decode_latency"])
        engine_total_prefill_time.append(engine_stats["engine_total_prefill_time"])
        engine_total_decode_time.append(engine_stats["engine_total_decode_time"])
        total_prefill_tokens.append(engine_stats["total_prefill_tokens"])
        total_decode_tokens.append(engine_stats["total_decode_tokens"])

    e2e_latency = time_evaluator(engine_generate, args=[], num_runs=num_runs)
    single_token_prefill_latency = np.array(single_token_prefill_latency)
    single_token_decode_latency = np.array(single_token_decode_latency)
    engine_total_prefill_time = np.array(engine_total_prefill_time)
    engine_total_decode_time = np.array(engine_total_decode_time)
    total_prefill_tokens = np.array(total_prefill_tokens)
    total_decode_tokens = np.array(total_decode_tokens)
    prefill_throughput = total_prefill_tokens / engine_total_prefill_time
    decode_throughput = total_decode_tokens / engine_total_decode_time
    overall_throughput = (total_prefill_tokens + total_decode_tokens) / e2e_latency

    print(args)
    print(f"Average end-to-end latency: {e2e_latency.mean():.4f} seconds for the entire batch")
    print(f"Single token prefill latency: {single_token_prefill_latency.mean() * 1e3:.4f} ms/tok")
    print(f"Single token decode latency: {single_token_decode_latency.mean() * 1e3:.4f} ms/tok")
    print(f"Engine prefill time: {engine_total_prefill_time.mean():.4f} s")
    print(f"Engine decode time: {engine_total_decode_time.mean():.4f} s")
    print(f"Request throughput: {args.num_prompts / e2e_latency.mean():.4f} req/s")
    print(f"Prefill token throughput: {prefill_throughput.mean():.4f} tok/s")
    print(f"Decode token throughput: {decode_throughput.mean():.4f} tok/s")
    print(f"Overall token throughput: {overall_throughput.mean():.4f} tok/s")


if __name__ == "__main__":
    ARGS = _parse_args()
    benchmark(ARGS)
