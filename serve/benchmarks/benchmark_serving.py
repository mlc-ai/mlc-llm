import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, float]] = []
RESPONSES = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
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
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    request_start_time = time.time()

    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "model": "test",
        "messages": prompt,
        "max_tokens": output_len,
        "stream": False,
        "temperature": 1.0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "ignore_eos": True,
    }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            try:
                output = json.loads(output)
            except:
                print(f"Cannot convert response to json. Returned response: '{output}', original prompt: {prompt}")
                return

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time

    REQUEST_LATENCY.append((prompt_len, request_latency))
    RESPONSES.append((prompt, output["choices"][0]["message"]["content"]))


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    request_rate: float,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(api_url, prompt, prompt_len, output_len, best_of, tokenizer)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://127.0.0.1:8000/v1/chat/completions"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    benchmark_start_time = time.time()
    asyncio.run(
        benchmark(api_url, input_requests, args.best_of, args.request_rate, tokenizer)
    )
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    throughput = args.num_prompts / benchmark_time
    print(f"Throughput: {throughput:.2f} requests/s")

    if args.dump_responses:
        for prompt, response in RESPONSES:
            print(f"Prompt: {prompt}")
            print(f"Response: {response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Append the current result to the given path if provided.",
    )
    parser.add_argument(
        "--dump-responses",
        action="store_true",
        help="Dump prompt / response pairs to stdout.",
    )
    args = parser.parse_args()
    main(args)
