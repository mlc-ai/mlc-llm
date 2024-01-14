# pylint: disable=line-too-long,missing-docstring
import argparse
import os
import random
from typing import List, Tuple

from mlc_chat.serve import Engine, GenerationConfig, KVCacheConfig
from mlc_chat.serve.engine import ModelInfo


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-lib-path", type=str)
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--batch-size", type=int, default=80)
    args.add_argument("--page-size", type=int, default=16)
    args.add_argument("--max-total-seq-length", type=int)
    args.add_argument("--seed", type=int, default=0)

    parsed = args.parse_args()
    parsed.model = os.path.dirname(parsed.model_lib_path)
    assert parsed.batch_size % 16 == 0
    assert parsed.page_size == 16
    assert parsed.max_total_seq_length >= 2048
    return parsed


def generate_requests(
    num_requests: int, input_length: int, output_length: int
) -> Tuple[List[List[int]], List[GenerationConfig]]:
    prompt_ids = []
    for _ in range(num_requests):
        token_ids = []
        for _ in range(input_length):
            token_ids.append(random.randint(0, 30000))
        prompt_ids.append(token_ids)
    generation_config_list = [
        GenerationConfig(temperature=1.0, top_p=1.0, max_tokens=output_length)
    ] * num_requests
    return prompt_ids, generation_config_list


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

    print(args)
    for num_requests in [1, 2, 4, 8, 16, 32, 64]:
        if num_requests > args.batch_size:
            continue
        for input_length in [64, 128, 256, 512, 1024]:
            if num_requests * input_length >= 16384:
                continue
            for output_length in [4]:
                print(f"nreq={num_requests}\t" f"in={input_length}\t" f"out={output_length}")
                prompt_ids, generation_config = generate_requests(
                    num_requests, input_length, output_length
                )
                engine.reset()
                engine.generate(prompt_ids, generation_config)
                print()


if __name__ == "__main__":
    ARGS = _parse_args()
    benchmark(ARGS)
