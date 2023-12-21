"""Benchmark offline inference throughput."""
import json
import random
import time
from typing import List, Tuple
import argparse
import pandas as pd
from mlc_serve.engine import (
    ChatMessage,
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
    get_engine_config,
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args


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
    num_sequences,
) -> float:
    for i, (prompt, _, output_len) in enumerate(requests):
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=-1,
            logit_bias={1: -1, 3: 1, 2: 2},
        )

        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=sampling_params,
                    stopping_criteria=StoppingCriteria(
                        max_tokens=output_len, stop_sequences=None
                    ),
                    debug_options=DebugOptions(ignore_eos=True, prompt=prompt),
                    num_sequences=num_sequences,
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
    engine_config = get_engine_config(
        {
            "use_staging_engine": args.use_staging_engine,
            "max_num_sequences": args.max_num_sequences,
            "max_input_len": args.max_input_len,
            "min_decode_steps": args.min_decode_steps,
            "max_decode_steps": args.max_decode_steps,
        }
    )

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
                model_artifact_path=args.model_artifact_path,
                engine_config=engine_config,
            )
        )
        tokenizer = engine.tokenizer

    return engine, tokenizer


def main(args: argparse.Namespace):
    print(args)

    engine, tokenizer = create_engine_and_tokenizer_module(args)

    # Sample the requests.
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer._tokenizer)

    elapsed_time = run_mlc(
        requests,
        engine,
        args.num_sequences_to_sample,
    )

    if args.use_staging_engine:
        engine.stop()

    total_num_tokens = sum(
        prompt_len + output_len * args.num_sequences_to_sample
        for _, prompt_len, output_len in requests
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
    parser = get_default_mlc_serve_argparser(description="Benchmark the throughput.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Append the current result to the given path if provided.",
    )
    args = parser.parse_args()
    args = postproc_mlc_serve_args(args)

    main(args)
