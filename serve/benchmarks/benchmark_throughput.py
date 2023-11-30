"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time, os
from typing import List, Tuple

import pandas as pd
from mlc_serve.engine import (
    ChatMessage,
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
    get_engine_config
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.logging_utils import configure_logging
from pathlib import Path


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
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
    return sampled_requests


def run_mlc(
    requests: List[Tuple[str, int, int]],
    engine,
) -> float:
    for i, (prompt, _, output_len) in enumerate(requests):
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
        )

        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=sampling_params,
                    stopping_criteria=StoppingCriteria(max_tokens=output_len, stop_sequences=None),
                    debug_options=DebugOptions(ignore_eos=True, prompt=prompt),
                )
            ]
        )

    start = time.time()

    while engine.has_pending_requests():
        engine.step()

    end = time.time()
    return end - start


def create_engine_and_tokenizer_module(
    args: argparse.Namespace,
):
    engine_config = get_engine_config({
        "use_staging_engine": args.use_staging_engine,
        "max_num_sequences": args.max_num_sequences,
        "max_input_len": args.max_input_len,
        "min_decode_steps": args.min_decode_steps,
        "max_decode_steps": args.max_decode_steps,
        "prompt_allocate_ratio": args.prompt_allocate_ratio
    })

    if args.use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(args.model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
        tokenizer = engine.tokenizer
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path = args.model_artifact_path,
                engine_config = engine_config,
        ))
        tokenizer = engine.tokenizer

    return engine, tokenizer


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    engine, tokenizer = create_engine_and_tokenizer_module(args)

    # Sample the requests.
    requests = sample_requests(
        args.dataset, args.num_prompts, tokenizer._tokenizer
    )

    elapsed_time = run_mlc(
        requests,
        engine,
    )

    if args.use_staging_engine:
        engine.stop()

    total_num_tokens = sum(
        prompt_len + output_len for _, prompt_len, output_len in requests
    )
    req_per_sec = len(requests) / elapsed_time
    tok_per_sec = total_num_tokens / elapsed_time

    print(
        f"Throughput: {req_per_sec:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )
    if args.report_path is not None:
        data = {
            "# prompts": [args.num_prompts],
            "req/s": [req_per_sec],
            "tok/s": [tok_per_sec],
            "elapsed time (s)": [elapsed_time],
        }
        df = pd.DataFrame(data)
        df.to_csv(args.report_path, mode="a", index=False, header=False)
        print("Data appended successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--use-staging-engine", action="store_true")
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--min-decode-steps", type=int, default=32)
    parser.add_argument("--max-decode-steps", type=int, default=56)
    parser.add_argument("--prompt-allocate-ratio", type=float, default=2.0)
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Append the current result to the given path if provided.",
    )
    parser.add_argument("--debug-logging", action="store_true")
    args = parser.parse_args()

    log_level = "DEBUG" if args.debug_logging else "INFO"
    configure_logging(enable_json_logs=False, log_level=log_level)

    args.model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))
    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    main(args)
