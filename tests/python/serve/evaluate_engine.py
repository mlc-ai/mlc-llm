# pylint: disable=line-too-long,missing-docstring
import argparse
import os
import random
from typing import List, Tuple

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-lib", type=str)
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--batch-size", type=int, default=80)
    args.add_argument("--max-total-seq-length", type=int)
    args.add_argument("--seed", type=int, default=0)

    parsed = args.parse_args()
    parsed.model = os.path.dirname(parsed.model_lib)
    assert parsed.batch_size % 16 == 0
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

    # Create engine
    engine = SyncMLCEngine(
        model=args.model,
        device=args.device,
        model_lib=args.model_lib,
        mode="server",
        engine_config=EngineConfig(
            max_num_sequence=args.batch_size,
            max_total_sequence_length=args.max_total_seq_length,
        ),
    )

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
