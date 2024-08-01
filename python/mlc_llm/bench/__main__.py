"""MLC LLM benchmark main entrance"""

import asyncio
import random
import requests
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer  # pylint: disable=import-error

import mlc_llm
from mlc_llm.bench.api_endpoint import SUPPORTED_BACKENDS, create_api_endpoint
from mlc_llm.bench.dataset import SUPPORTED_DATASET, Dataset, create_dataset
from mlc_llm.bench.executor import Executor, create_executors
from mlc_llm.bench.request_processor import (
    AttachStreamFlag,
    MetricAnalyzer,
    SampleRequests,
    SequentialProcessor,
)
from mlc_llm.bench.request_record import convert_reports_to_df, generate_metrics_summary
from mlc_llm.cli.serve import EngineConfigOverride
from mlc_llm.serve import EngineConfig
from mlc_llm.support import argparse, logging

logging.enable_logging()
logger = logging.getLogger(__name__)


def _parse_num_concurrent_requests(num_str: Optional[str]) -> Optional[List[int]]:
    if num_str is None:
        return None
    numbers = num_str.split(",")
    if any(not number.isdigit() for number in numbers):
        raise ValueError(f"Unrecognized num_concurrent_requests list: {numbers}")
    return list(int(number) for number in numbers)


def _parse_mlc_engine_config(config_str: Optional[str]) -> EngineConfig:
    if config_str is None:
        return None
    engine_config_override = EngineConfigOverride.from_str(config_str)
    return EngineConfig(
        tensor_parallel_shards=engine_config_override.tensor_parallel_shards,
        max_num_sequence=engine_config_override.max_num_sequence,
        max_total_sequence_length=engine_config_override.max_total_seq_length,
        prefill_chunk_size=engine_config_override.prefill_chunk_size,
        sliding_window_size=engine_config_override.sliding_window_size,
        attention_sink_size=engine_config_override.attention_sink_size,
        max_history_size=engine_config_override.max_history_size,
        gpu_memory_utilization=engine_config_override.gpu_memory_utilization,
        spec_draft_length=engine_config_override.spec_draft_length,
    )


def _launch_mlc_server(args: argparse.argparse.Namespace):
    return mlc_llm.serve.PopenServer(
        model=args.tokenizer,
        mode="server",
        model_lib=args.mlc_model_lib,
        enable_tracing=False,
        host=args.host,
        port=args.port,
        engine_config=args.mlc_engine_config,
    )


def run_executor(
    executor: Executor,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.argparse.Namespace,
) -> Dict[str, Any]:
    """Run the executor with the given dataset and args. Return the benchmark report dict."""
    # Pre-process
    num_warmup_requests = executor.get_num_warmup_requests()
    pre_processor = SequentialProcessor(
        SampleRequests(args.num_requests + num_warmup_requests),
        AttachStreamFlag(args.stream),
    )
    request_records = dataset.generate_request_records(
        args.input_len,
        args.output_len,
        args.input_len_std,
        args.output_len_std,
    )
    request_records = pre_processor(request_records)
    assert len(request_records) == args.num_requests + num_warmup_requests
    warmup_requests = request_records[:num_warmup_requests]
    request_records = request_records[num_warmup_requests:]

    # Warmup and run
    logger.info(
        "Executor %s created for %s dataset at %s",
        type(executor).__name__,
        args.dataset,
        args.dataset_path,
    )
    logger.info("Warmup with %d request(s)...", len(warmup_requests))
    asyncio.run(executor.warmup(warmup_requests))
    logger.info("Warmup finished. Start benchmarking...")

    if args.cuda_profile:
        cuda_profiler_start_url = f"http://{args.host}:{args.port}/debug/cuda_profiler_start"
        cuda_profiler_start_response = requests.post(cuda_profiler_start_url, timeout=60)
        assert cuda_profiler_start_response.status_code == 200

    request_records, duration = asyncio.run(executor.run_benchmark(request_records))

    if args.cuda_profile:
        cuda_profiler_stop_url = f"http://{args.host}:{args.port}/debug/cuda_profiler_stop"
        cuda_profiler_stop_response = requests.post(cuda_profiler_start_url, timeout=60)
        assert cuda_profiler_stop_response.status_code == 200

    # Post-process
    request_records = MetricAnalyzer(tokenizer)(request_records)
    report = generate_metrics_summary(request_records, duration, args.num_requests, args.num_gpus)
    report = {**report, **executor.get_executor_feature_dict()}
    return report


def main(args: argparse.argparse.Namespace):
    """Main benchmark entrance."""
    random.seed(args.seed)
    np.random.seed(args.seed)

    mlc_server = None
    if args.mlc_model_lib or args.cuda_profile:
        if not args.mlc_model_lib:
            raise ValueError("The model-lib argument is required.")
        mlc_server = _launch_mlc_server(args)

    def _main():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = create_dataset(args, tokenizer)
        api_endpoint = create_api_endpoint(args)
        executors = create_executors(args, api_endpoint)
        reports = []
        for executor in executors:
            reports.append(run_executor(executor, dataset, tokenizer, args))

        # Construct data frame
        df = convert_reports_to_df(reports)
        print(df)
        df.to_csv(args.output)
        logger.info("Benchmark results dumped to file %s", args.output)

    if mlc_server is not None:
        with mlc_server:
            _main()
    else:
        _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The dataset file path.",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        choices=SUPPORTED_BACKENDS,
        default="openai",
        help="The API endpoint API for benchmarking.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="The path of the tokenizer directory.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="The number of GPUs used by the server. "
        "We need this to better analyze the throughput per GPU.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="The number of requests for benchmark.",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=_parse_num_concurrent_requests,
        help="The number(s) of concurrent requests to benchmark. "
        'It can be either one integer or a list of integer separated by commas(","). '
        "When specified, for each integer, the benchmark keeps these many consistent "
        "number of concurrently running requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=int,
        help="The request rate, denoting the number of new requests each second. "
        "When specified, the benchmark sends these many new requests each second.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        help="The benchmark request average input length. Default to None, "
        "which means the request input length depends on the dataset being used.",
    )
    parser.add_argument(
        "--input-len-std",
        type=float,
        default=0,
        help="The benchmark request input length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        help="The benchmark request average output length. Default to None, "
        "which means the request output length depends on the dataset being used.",
    )
    parser.add_argument(
        "--output-len-std",
        type=float,
        default=0,
        help="The benchmark request output length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--stream",
        type=bool,
        default=True,
        help="Whether to benchmark stream responses. "
        "When not enabled, metrics such as time-to-first-token (TTFT) will not be available. "
        "Default to True.",
    )
    parser.add_argument(
        # NOTE: The current implementation of server metrics still has some issues that need fixes,
        # which makes it not work to include server metrics.
        "--include-server-metrics",
        action="store_true",
        help="Whether to also benchmark the server side request metrics. "
        "This option is only available when benchmarking MLC server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="The host address of the backend API.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="The port of the backend API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="The timeout limit of each request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random number seed. Default to 0.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Whether to disable showing progress bar with tqdm during benchmarking.",
    )
    parser.add_argument(
        "--mlc-model-lib",
        type=str,
        help="The model lib path when benchmarking MLC serve. "
        "When specified, the server is automatic launched and no external server launch is needed.",
    )
    parser.add_argument(
        "--mlc-engine-config",
        type=_parse_mlc_engine_config,
        help="The engine config used when launch MLC server.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mlc_benchmark.csv",
        help="The path of the output file where to dump the benchmark results.",
    )
    parser.add_argument(
        "--cuda-profile",
        type=bool,
        default=False,
        help="Whether to enable cuda profile on server. "
        "The --mlc-model-lib path should be provided when enabling this option.",
    )

    main(parser.parse_args())
