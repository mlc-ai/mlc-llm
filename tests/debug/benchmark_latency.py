import argparse
import time
from typing import Any, Callable, List

from mlc_chat.serve import GenerationConfig, KVCacheConfig
from mlc_chat.serve.engine import Engine, ModelInfo


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-id", type=str, default="Llama-2-7b-chat-hf-q4f16_1")
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--input-length", type=int, default=32)
    args.add_argument("--output-length", type=int, default=256)
    args.add_argument("--batch-size", type=int, default=16)
    parsed = args.parse_args()
    return parsed


def time_evaluator(func: Callable, args: List[Any] = [], num_runs: int = 3, num_warmups: int = 1):
    # warmup run
    for _ in range(num_warmups):
        func(*args)

    total_time = 0.0
    for run in range(num_runs):
        print(f"start round {run}")
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        total_time += end - start
        print(f"finish round {run}")

    return total_time / num_runs


def benchmark(args: argparse.Namespace):
    # Initialize model loading info and KV cache config
    model = ModelInfo(args.model_id, args.device)
    kv_cache_config = KVCacheConfig(
        page_size=16,
        max_num_sequence=args.batch_size,
        max_total_sequence_length=args.output_length * args.batch_size * 2,
    )
    generation_config = GenerationConfig(
        temperature=1.0, top_p=1.0, max_new_tokens=args.output_length
    )
    prompts = [[0] * args.input_length] * args.batch_size
    # Create engine
    engine = Engine(model, kv_cache_config)

    def engine_generate():
        engine.generate(prompts, generation_config)

    avg_latency = time_evaluator(engine_generate, num_runs=3)
    print(args)
    print(f"Average latency: {avg_latency} seconds for the entire batch")


if __name__ == "__main__":
    ARGS = _parse_args()
    benchmark(ARGS)
