"""MLC LLM benchmark main entrance"""

import functools
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from transformers import AutoTokenizer  # pylint: disable=import-error

import mlc_llm
from mlc_llm.bench.api_endpoint import SUPPORTED_BACKENDS, create_api_endpoint
from mlc_llm.bench.dataset import SUPPORTED_DATASET, Dataset, create_dataset
from mlc_llm.bench.request_processor import (
    MetricAnalyzer,
    RequestProcessor,
    create_pipelines,
)
from mlc_llm.bench.request_record import (
    RequestRecord,
    convert_reports_to_df,
    generate_metrics_summary,
    pretty_print_report,
)
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


def _parse_request_rate(request_rate_str: Optional[str]) -> Optional[List[np.float32]]:
    if request_rate_str is None:
        return None
    request_rates = request_rate_str.split(",")
    results = []
    for rate_str in request_rates:
        request_rate = float(rate_str)
        if request_rate <= 0:
            raise ValueError(f"Invalid request rate {request_rate}")
        results.append(np.float32(request_rate))
    return results


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
        prefill_mode=engine_config_override.prefill_mode,
        prefix_cache_max_num_recycling_seqs=engine_config_override.prefix_cache_max_num_recycling_seqs,  # pylint: disable=line-too-long
        prefix_cache_mode=engine_config_override.prefix_cache_mode,
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


def run_pipeline(
    pipeline: RequestProcessor,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.argparse.Namespace,
) -> Tuple[Dict[str, Any], List[RequestRecord]]:
    """Run the pipeline with the given dataset and args. Return the benchmark report dict."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    request_records = dataset.generate_request_records(
        args.input_len,
        args.output_len,
        args.input_len_std,
        args.output_len_std,
    )
    request_records = pipeline(request_records)
    num_total_requests = (
        args.num_requests if not args.per_gpu_workload else args.num_requests * args.num_gpus
    )
    assert len(request_records) == num_total_requests
    sorted_requests: List[RequestRecord] = [None] * num_total_requests
    for request_record in request_records:
        assert request_record.request_id is not None
        assert sorted_requests[request_record.request_id] is None
        sorted_requests[request_record.request_id] = request_record

    request_records = MetricAnalyzer(tokenizer)(request_records)
    report = generate_metrics_summary(request_records, num_total_requests, args.num_gpus)
    return report, sorted_requests


def query_mlc_server_metrics(host: str, port: int):
    """Try to get the MLC server metrics whenever it exists."""
    try:
        r = requests.post(f"http://{host}:{port}/debug/dump_engine_metrics", json={}, timeout=10)
        if r.status_code == 200:
            print(f"MLC server metrics: {r.json()}")
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def main(args: argparse.argparse.Namespace):
    """Main benchmark entrance."""
    mlc_server = None
    if args.mlc_model_lib:
        mlc_server = _launch_mlc_server(args)
    if args.num_requests <= 0:
        raise ValueError("Number of requests to benchmark must be positive.")

    def _main():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = create_dataset(args, tokenizer)
        f_create_api_endpoint = functools.partial(create_api_endpoint, args)
        pipelines = create_pipelines(args, f_create_api_endpoint, dataset)
        reports = []
        alltime_records = {}
        for i, pipeline in enumerate(pipelines):
            report, request_records = run_pipeline(pipeline, dataset, tokenizer, args)
            exec_feature = (
                json.dumps(report["exec_feature"])
                if report["exec_feature"] is not None
                else f"pipeline{i}"
            )
            alltime_records[exec_feature] = [
                request_record.model_dump() for request_record in request_records
            ]
            reports.append(report)
            pretty_print_report(report)
        query_mlc_server_metrics(args.host, args.port)

        # Construct data frame
        df = convert_reports_to_df(reports)
        print(df)
        df.to_csv(args.output, index=False)
        logger.info("Benchmark results dumped to file %s", args.output)
        if args.debug_dump:
            debug_dump_filepath = (
                args.output[:-4] if args.output.endswith(".csv") else args.output
            ) + "_debug_dump.log"
            with open(debug_dump_filepath, "w", encoding="utf-8") as file:
                json.dump(alltime_records, file, indent=4)
            logger.info("Debug log dumped to file %s", debug_dump_filepath)

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
        "--num-warmup-requests",
        type=int,
        help="The number of requests for warmup. "
        "It is optional when fixing the number of concurrent requests, and is required otherwise.",
    )
    parser.add_argument(
        "--per-gpu-workload",
        default=False,
        action="store_true",
        help='When set to True, the specified "num_concurrent_requests"/"request_rate" '
        "denote the workload **per GPU**, which means that the real values of "
        '"num_concurrent_requests"/"request_rate" used in benchmark'
        'will be multiplied by "num_gpus".',
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
        type=_parse_request_rate,
        help="The request rate(s) denoting the number of new requests each second. "
        'It can be either one float number (or "inf") or a list of numbers separated '
        'by commas(","). '
        "When specified, the benchmark sends these many new requests each second. "
        'If it is "inf", all requests will be sent together at once.',
    )
    parser.add_argument(
        "--replay-timestamp-scale",
        type=float,
        help="The timestamp scale when replaying the timestamps in a dataset. "
        'The dataset replay mode is enabled when neither "--num-concurrent-requests" and '
        '"--request-rate" is specified. '
        "The scale is 1 by default in the replay mode.",
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
        default=3 * 60 * 60,
        help="The timeout limit of each request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random number seed. Default to 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature value for logit adjustment. Default to 1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The top-p value for sampling. Default to 1.",
    )
    parser.add_argument(
        "--ignore-eos",
        default=False,
        action="store_true",
        help='Whether to set the "ignore_eos" field.',
    )
    parser.add_argument(
        "--apply-chat-template",
        default=False,
        action="store_true",
        help="Whether to apply chat template to the request input text. "
        'It is not supported when "--input-len" is specified.',
    )
    parser.add_argument(
        "--num-process-workers",
        type=int,
        help="The number of parallel process workers to send the requests.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Whether to disable showing progress bar with tqdm during benchmarking.",
    )
    parser.add_argument(
        "--max-schedule-gap",
        type=float,
        default=0.5,
        help="The maximum allowed delay between the scheduled time in seconds.",
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
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Whether to enable cuda profile on server. "
        "The --mlc-model-lib path should be provided when enabling this option.",
    )
    parser.add_argument(
        "--debug-dump",
        default=False,
        action="store_true",
        help="Whether to dump all request record raw data to file.",
    )
    parser.add_argument(
        "--multi-round",
        default=False,
        action="store_true",
        help="Whether to chat like multi round conversion with history log each request. "
        "Only enabled when benchmarked with fixed concurrent request mode."
        "The --num-concurrent-requests should be provided when enabling this option.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mlc_benchmark.csv",
        help="The path of the output file where to dump the benchmark results.",
    )

    main(parser.parse_args())
