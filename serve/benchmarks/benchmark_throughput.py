"""Benchmark offline inference throughput."""
import json
import random
import time
from typing import List, Tuple
import argparse
import pandas as pd
from mlc_serve.engine import (
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.utils import (
    get_default_mlc_serve_argparser,
    postproc_mlc_serve_args,
    create_mlc_engine,
)
from utils import add_sampling_flags, postproc_sampling_args


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


def run_mii(requests: List[Tuple[str, int, int]], args) -> float:
    from mii import pipeline

    engine = pipeline(args.model, tensor_parallel=args.num_shards)
    prompts = [prompt for prompt, _, _ in requests]
    temp = 0.000001 if random.random() <= args.greedy_sampling_ratio else 0.7
    start = time.perf_counter()
    engine(
        prompts,
        max_new_tokens=args.num_output_tokens,
        ignore_eos=args.sampling_setting["ignore_eos"],
        # mii does not support temperature of zero.
        temperature=temp,
        top_p=1 if temp == 0.0 else args.sampling_setting["top_p"],
        top_k=-1 if temp == 0.0 else args.sampling_setting["top_k"],
        # mii currently does not support `logit_bias` and any of penalties
    )
    end = time.perf_counter()
    return end - start


def run_vllm(requests: List[Tuple[str, int, int]], args) -> float:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        quantization=args.quantization,
        tensor_parallel_size=args.num_shards,
        trust_remote_code=True,
        max_model_len=None,  # derive from the model
    )

    # Add the requests to the engine.
    for prompt, _, _ in requests:
        temp = 0.0 if random.random() <= args.greedy_sampling_ratio else 0.7
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=SamplingParams(
                n=args.num_sequences_to_sample,
                use_beam_search=False,
                temperature=temp,
                top_p=1 if temp == 0.0 else args.sampling_setting["top_p"],
                top_k=-1 if temp == 0.0 else args.sampling_setting["top_k"],
                repetition_penalty=args.sampling_setting["repetition_penalty"],
                frequency_penalty=args.sampling_setting["frequency_penalty"],
                presence_penalty=args.sampling_setting["presence_penalty"],
                # vllm does not support `logit bias`
                ignore_eos=args.sampling_setting["ignore_eos"],
                max_tokens=args.num_output_tokens,
            ),
        )

    start = time.perf_counter()
    llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    return end - start


def run_mlc(engine, requests, args) -> float:
    for i, (prompt, _, _) in enumerate(requests):
        temp = 0.0 if random.random() <= args.greedy_sampling_ratio else 0.7
        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=None,  # Provide prompt as `DebugOption` to bypass the conv template
                    sampling_params=SamplingParams(
                        temperature=temp,
                        top_p=1 if temp == 0.0 else args.sampling_setting["top_p"],
                        top_k=-1 if temp == 0.0 else args.sampling_setting["top_k"],
                        repetition_penalty=args.sampling_setting["repetition_penalty"],
                        frequency_penalty=args.sampling_setting["frequency_penalty"],
                        presence_penalty=args.sampling_setting["presence_penalty"],
                        logit_bias=args.sampling_setting["logit_bias"],
                    ),
                    stopping_criteria=StoppingCriteria(
                        max_tokens=args.num_output_tokens, stop_sequences=None
                    ),
                    num_sequences=args.num_sequences_to_sample,
                    debug_options=DebugOptions(
                        ignore_eos=args.sampling_setting["ignore_eos"], prompt=prompt
                    ),
                )
            ]
        )

    start = time.perf_counter()

    while engine.has_pending_requests():
        engine.step()

    end = time.perf_counter()

    if args.use_staging_engine:
        engine.stop()

    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    if args.backend == "mlc-serve":
        # Create mlc engine
        engine = create_mlc_engine(args)
        # Sample the requests.
        requests = sample_requests(
            args.dataset, args.num_prompts, engine.tokenizer._tokenizer
        )
        elapsed_time = run_mlc(engine, requests, args)
    else:
        from transformers import AutoTokenizer

        assert (
            args.model is not None
        ), "Please provide model path for vllm and deepspeed mii."
        assert (
            args.num_shards is not None
        ), "Please provide number of gpus for vllm and deepspeed mii."

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
        if args.backend == "mii":
            elapsed_time = run_mii(requests, args)
        elif args.backend == "vllm":
            elapsed_time = run_vllm(requests, args)

    total_num_tokens = sum(
        prompt_len + args.num_output_tokens * args.num_sequences_to_sample
        for _, prompt_len, _ in requests
    )
    req_per_sec = len(requests) / elapsed_time
    tok_per_sec = total_num_tokens / elapsed_time

    print(
        f"Engine Throughput: {req_per_sec:.2f} requests/s, "
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
    parser = get_default_mlc_serve_argparser(
        description="Benchmark the throughput.", allow_override=True
    )
    parser.add_argument(
        "--backend", type=str, default="mlc-serve", choices=["mlc-serve", "vllm", "mii"]
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--greedy-sampling-ratio",
        type=float,
        default=0.5,
        help="Ratio of greedy sampling in the requests.",
    )
    parser.add_argument(
        "--num-output-tokens",
        type=int,
        default=128,
        help="Maximum number of generation tokens.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Append the current result to the given path if provided.",
    )
    # flags for vllm and deepspeed mii
    # override local-id to make it non-required as it is only for mlc-serve
    parser.add_argument("--local-id", type=str, required=False)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path. This is for vLLM and Deepspeed MII.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of GPUs. This is for vLLM and Deepspeed MII.",
    )
    # flags for vllm
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["awq", "gptq", "squeezellm", None],
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="data type for model weights and activations. "
        'The "auto" option will use FP16 precision '
        "for FP32 and FP16 models, and BF16 precision "
        "for BF16 models.",
    )
    add_sampling_flags(parser)
    args = parser.parse_args()
    if args.backend == "mlc-serve":
        postproc_mlc_serve_args(args)

    assert args.greedy_sampling_ratio >= 0.0 and args.greedy_sampling_ratio <= 1.0
    postproc_sampling_args(args)

    main(args)
