"""MLC LLM Bench Request"""

import argparse
import asyncio
import concurrent.futures
import copy
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer  # pylint: disable=import-error

from mlc_llm.bench.api_endpoint import APIEndPoint
from mlc_llm.bench.dataset import Dataset
from mlc_llm.bench.request_record import GroupedRequestRecord, RequestRecord
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    DebugConfig,
)
from mlc_llm.support import logging

logger = logging.getLogger(__name__)


class RequestProcessor:  # pylint: disable=too-few-public-methods
    """The request processor base class.
    Each processor can take a list of RequestRecord, applying the process,
    and returning the processed RequestRecord in the end.
    """

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        raise NotImplementedError()


class LogMessage(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that prints the logger message."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        logger.info(self.message)
        return request_records


class SampleRequests(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that samples requests out from the given request list."""

    def __init__(self, num_requests: int, take_first_x_requests: bool = False) -> None:
        self.num_requests = num_requests
        # If `take_first_x_requests` is True, the first `num_requests` requests
        # are returned and sampling will not happen.
        self.take_first_x_requests = take_first_x_requests

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        assert len(request_records) > 0, "Empty input request record."

        # We expect the input request records to be all grouped or all plain.
        if isinstance(request_records[0], GroupedRequestRecord):
            assert all(isinstance(record, GroupedRequestRecord) for record in request_records)
            return self._sample_from_grouped_request_records(request_records)

        assert all(not isinstance(record, GroupedRequestRecord) for record in request_records)
        return self._sample_from_plain_request_records(request_records)

    def _sample_from_plain_request_records(
        self, request_records: List[RequestRecord]
    ) -> List[RequestRecord]:
        samples: List[RequestRecord] = []
        if self.take_first_x_requests:
            if len(request_records) < self.num_requests:
                raise ValueError(
                    f"Insufficient requests. Requiring {self.num_requests} requests "
                    f"but only {len(request_records)} are available."
                )
            samples = copy.deepcopy(list(request_records[: self.num_requests]))
        else:
            while len(samples) < self.num_requests:
                # Create a new list so that the in-place shuffle does not mutate the input list.
                records = list(request_records)
                random.shuffle(records)
                samples += copy.deepcopy(records)
            samples = samples[: self.num_requests]
        for i, record in enumerate(samples):
            record.request_id = i
        return samples

    def _sample_from_grouped_request_records(
        self, grouped_request_records: List[GroupedRequestRecord]
    ) -> List[RequestRecord]:
        num_total_available_requests = sum(
            len(record.records) for record in grouped_request_records
        )
        if self.num_requests > num_total_available_requests:
            raise ValueError(
                "Due to the existence of shared common prefixes, we do not allow "
                "benchmarking with requests more than the available requests in the dataset. "
                f"The required number of requests {self.num_requests} exceeds the "
                f"number of total available requests {num_total_available_requests}."
            )

        # Create a new list so that the in-place shuffle does not mutate the input list.
        records = list(grouped_request_records)
        if not self.take_first_x_requests:
            random.shuffle(records)
        remaining = self.num_requests
        samples: List[RequestRecord] = []
        for grouped_request_record in grouped_request_records:
            num_used_requests = min(len(grouped_request_record.records), remaining)
            samples += grouped_request_record.records[:num_used_requests]
            remaining -= num_used_requests
            if remaining == 0:
                break
        for i, record in enumerate(samples):
            record.request_id = i
        return samples


class AttachModelName(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that attaches model name to requests."""

    def __init__(self, model: str) -> None:
        self.model = model

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for request_record in request_records:
            request_record.chat_cmpl.model = self.model
        return request_records


class AttachRequestRateTimestamp(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that applies timestamps to the requests."""

    def __init__(self, request_rate: np.float32) -> None:
        self.request_rate = request_rate

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        timestamp = 0.0
        for request_record in request_records:
            assert request_record.timestamp is None, "The request record already has a timestamp"
            request_record.timestamp = timestamp
            timestamp += float(np.random.exponential(1.0 / self.request_rate))
        return request_records


class AttachExecutionFeature(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that attaches execution features to all requests"""

    def __init__(self, exec_feature: Dict[str, Any]) -> None:
        self.exec_feature = exec_feature

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for request_record in request_records:
            assert request_record.metrics is not None
            request_record.metrics.exec_feature = self.exec_feature
        return request_records


class AttachStreamFlag(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that attaches the stream flag to the requests."""

    def __init__(self, stream: Optional[bool]) -> None:
        self.stream = stream

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        if self.stream is None:
            return request_records
        for request_record in request_records:
            request_record.chat_cmpl.stream = self.stream
        return request_records


class AttachSamplingOptions(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that attaches the stream flag to the requests."""

    def __init__(self, temperature: float, top_p: float, ignore_eos: bool) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.ignore_eos = ignore_eos

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for request_record in request_records:
            request_record.chat_cmpl.temperature = self.temperature
            request_record.chat_cmpl.top_p = self.top_p
            request_record.chat_cmpl.frequency_penalty = 0.0
            request_record.chat_cmpl.presence_penalty = 0.0
            request_record.chat_cmpl.tool_choice = "none"
            if self.ignore_eos:
                request_record.chat_cmpl.debug_config = DebugConfig(ignore_eos=True)
        return request_records


class ScaleTimestamp(RequestProcessor):  # pylint: disable=too-few-public-methods
    """Scale the timestamp of requests by the given scale factor."""

    def __init__(self, timestamp_scale: float):
        self.timestamp_scale = timestamp_scale

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for request_record in request_records:
            if request_record.timestamp is None:
                raise ValueError(
                    f"The timestamp of request {request_record} has not been initialized."
                )
            request_record.timestamp *= self.timestamp_scale
        return request_records


class MetricAnalyzer(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that analyzes the raw benchmark results and computes more detailed metrics."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        updated_records = []
        for request_record in request_records:
            metrics = request_record.metrics
            if not metrics.success:
                assert request_record.error_msg is not None
                continue

            metrics.output_tokens = len(
                self.tokenizer.encode(request_record.output_str, add_special_tokens=False)
            )
            first_chunk_output_tokens = len(
                self.tokenizer.encode(
                    request_record.first_chunk_output_str, add_special_tokens=False
                )
            )
            if metrics.output_tokens <= first_chunk_output_tokens:
                metrics.success = False
                request_record.error_msg = (
                    f"Total output token num ({metrics.output_tokens}) equals "
                    f'the first chunk output token. Output text "{request_record.output_str}", '
                    f'first chunk output text "{request_record.first_chunk_output_str}"'
                )
                continue
            assert metrics.input_tokens > 0, "Invalid prompt tokens"
            metrics.inter_token_latency_s = metrics.end_to_end_latency_s / metrics.output_tokens
            if metrics.time_to_first_token_s is None:
                metrics.time_to_first_token_s = 0
            metrics.time_per_output_token_s = (
                metrics.end_to_end_latency_s - metrics.time_to_first_token_s
            ) / (metrics.output_tokens - first_chunk_output_tokens)
            updated_records.append(request_record)
        return updated_records


class WarmupAndRun(RequestProcessor):  # pylint: disable=too-few-public-methods,line-too-long
    """The processor that runs warmup first and then runs the benchmark with the given pipeline."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_warmup_requests: int,
        num_benchmark_requests: int,
        pipeline: RequestProcessor,
        cuda_profile_url: Optional[str],
        fake_warmup: bool = False,
    ) -> None:
        self.num_warmup_requests = num_warmup_requests
        self.num_benchmark_requests = num_benchmark_requests
        self.pipeline = pipeline
        self.cuda_profile_url = cuda_profile_url
        self.fake_warmup = fake_warmup

    def generate_fake_warmup_requests(  # pylint: disable=missing-function-docstring
        self, num_warmup_requests: int, example_request: RequestRecord
    ) -> List[RequestRecord]:
        records = []
        for _ in range(num_warmup_requests):
            record = copy.deepcopy(example_request)
            record.chat_cmpl = ChatCompletionRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Please output arbitrary coherent sentences. Do not output eos token.",  # pylint: disable=line-too-long
                    }
                ],
                model="",
                max_tokens=128,
            )
            records.append(record)
        return records

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        # Warmup
        if self.fake_warmup:
            assert len(request_records) == self.num_benchmark_requests
            benchmark_requests = request_records
            example_request = benchmark_requests[0]
            warmup_requests = self.generate_fake_warmup_requests(
                self.num_warmup_requests, example_request=example_request
            )
        else:
            assert len(request_records) == self.num_warmup_requests + self.num_benchmark_requests
            benchmark_requests = request_records[: -self.num_warmup_requests]
            warmup_requests = request_records[-self.num_warmup_requests :]
        for request_record in warmup_requests:
            request_record.timestamp = 0 if request_record.timestamp is not None else None
        warmup_requests = self._process_warmup_requests(warmup_requests)
        logger.info("Warmup with %d request(s)...", self.num_warmup_requests)
        self.pipeline(warmup_requests)

        # Then run benchmark
        if self.cuda_profile_url is not None:
            cuda_profiler_start_url = self.cuda_profile_url + "/debug/cuda_profiler_start"
            cuda_profiler_start_response = requests.post(cuda_profiler_start_url, timeout=60)
            assert cuda_profiler_start_response.status_code == 200
        logger.info("Warmup finished. Start benchmarking...")
        updated_request_records = self.pipeline(benchmark_requests)
        if self.cuda_profile_url is not None:
            cuda_profiler_stop_url = self.cuda_profile_url + "/debug/cuda_profiler_stop"
            cuda_profiler_stop_response = requests.post(cuda_profiler_stop_url, timeout=60)
            assert cuda_profiler_stop_response.status_code == 200

        return updated_request_records

    def _process_warmup_requests(self, warmup_requests: List[RequestRecord]) -> List[RequestRecord]:
        if len(warmup_requests) == 0:
            return warmup_requests
        # NOTE: to warm up the server for as more different batch sizes as possible,
        # we usese 128 output tokens for the first request and use two more tokens
        # for every followup request.
        # Setting a high temperature and top-p to avoid early stop as much as possible.
        warmup_requests[0].chat_cmpl.max_tokens = 128
        for i in range(1, len(warmup_requests)):
            warmup_requests[i].chat_cmpl.max_tokens = (
                warmup_requests[i - 1].chat_cmpl.max_tokens + 1
            )
            warmup_requests[i].chat_cmpl.temperature = 2.0
            warmup_requests[i].chat_cmpl.top_p = 1.0
        return warmup_requests


class SequentialProcessor(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that sequentially applies a list of processors in order."""

    processors: List[RequestProcessor]

    def __init__(self, *processors: RequestProcessor) -> None:
        self.processors = list(processors)

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for processor in self.processors:
            request_records = processor(request_records)
        return request_records


class Executor(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The executor base class, denoting the kind of benchmark mode."""

    def __init__(
        self,
        f_create_api_endpoint: Callable[[], APIEndPoint],
        num_processes: int,
        disable_tqdm: bool,
    ) -> None:
        self.f_create_api_endpoint = f_create_api_endpoint
        self.disable_tqdm = disable_tqdm
        self.num_processes = num_processes

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        raise NotImplementedError()


class FixedConcurrentRequestExecutor(Executor):  # pylint: disable=too-few-public-methods
    """The benchmark executor of fixing the number of concurrent requests."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        f_create_api_endpoint: Callable[[], APIEndPoint],
        num_processes: Optional[int],
        disable_tqdm: bool,
        num_concurrent_requests: int,
        multi_round: bool,
    ) -> None:
        if num_processes is None:
            # We assign each process at most 32 concurrent requests to send
            # so that the asyncio pressure will not be too much.
            num_processes = min((num_concurrent_requests + 31) // 32, 10)
        super().__init__(f_create_api_endpoint, num_processes, disable_tqdm)
        self.num_concurrent_requests = num_concurrent_requests
        self.multi_round = multi_round

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        partitions: List[List[RequestRecord]] = [
            request_records[slice(i, len(request_records), self.num_processes)]
            for i in range(self.num_processes)
        ]
        # Package "tokenizers" reports warnings with multiprocessing.
        # We disable "TOKENIZERS_PARALLELISM" to depress the warnings.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        pbar = None if self.disable_tqdm else tqdm(total=len(request_records))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as pool:
            futures = [
                pool.submit(
                    FixedConcurrentRequestExecutor._process_task,
                    self.f_create_api_endpoint,
                    partition,
                    self.num_concurrent_requests // self.num_processes
                    + int(i < self.num_concurrent_requests % self.num_processes),
                    self.multi_round,
                )
                for i, partition in enumerate(partitions)
            ]
            results: List[RequestRecord] = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                results.extend(future.result())
                if pbar is not None:
                    pbar.update(len(partitions[i]))

        return results

    @staticmethod
    def _process_task(
        f_create_api_endpoint: Callable[[], APIEndPoint],
        request_records: List[RequestRecord],
        num_concurrent_requests: int,
        multi_round: bool,
    ) -> List[RequestRecord]:
        if len(request_records) == 0:
            return []
        chat_history: List[List[ChatCompletionMessage]] = [
            [] for _ in range(num_concurrent_requests)
        ]

        async def process_task_impl(
            f_create_api_endpoint: Callable[[], APIEndPoint],
            request_records: List[RequestRecord],
            num_concurrent_requests: int,
            multi_round: bool,
        ) -> List[RequestRecord]:
            api_endpoint = f_create_api_endpoint()
            updated_request_records: List[RequestRecord] = [None for _ in request_records]
            async with api_endpoint:
                num_sent_request = 0

                async def _task(i: int) -> None:
                    nonlocal num_sent_request
                    while True:
                        if num_sent_request == len(request_records):
                            break
                        idx = num_sent_request
                        num_sent_request += 1
                        request = request_records[idx]

                        if multi_round:
                            request.chat_cmpl.messages = (
                                chat_history[i] + request.chat_cmpl.messages
                            )

                        updated_request_records[idx] = await api_endpoint(request)

                        if multi_round:
                            chat_history[i] = updated_request_records[idx].chat_cmpl.messages + [
                                ChatCompletionMessage(
                                    content=updated_request_records[idx].output_str,
                                    role="assistant",
                                )
                            ]

                tasks = [asyncio.create_task(_task(i)) for i in range(num_concurrent_requests)]
                await asyncio.gather(*tasks)

            return updated_request_records

        return asyncio.run(
            process_task_impl(
                f_create_api_endpoint,
                request_records,
                num_concurrent_requests,
                multi_round,
            )
        )


class FixTimestampExecutor(Executor):  # pylint: disable=too-few-public-methods
    """The benchmark executor of fixing the timestamps of sending requests."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        f_create_api_endpoint: Callable[[], APIEndPoint],
        num_processes: Optional[int],
        disable_tqdm: bool,
        max_schedule_gap: float,
        num_requests: int,
    ) -> None:
        if num_processes is None:
            # We assign each process at most 32 requests to send
            # so that the asyncio pressure will not be too much.
            num_processes = min((num_requests + 31) // 32, 10)
        super().__init__(f_create_api_endpoint, num_processes, disable_tqdm)
        self.max_schedule_gap = max_schedule_gap
        self.num_requests = num_requests

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        assert len(request_records) > 0
        assert all(request_record.timestamp is not None for request_record in request_records)
        # Sort the request records in timestamp ascending order before partitioning.
        request_records.sort(key=lambda request_record: request_record.timestamp)
        base_timestamp = request_records[0].timestamp
        partitions: List[List[RequestRecord]] = [
            request_records[slice(i, len(request_records), self.num_processes)]
            for i in range(self.num_processes)
        ]
        base_sys_time = time.time()
        # Package "tokenizers" reports warnings with multiprocessing.
        # We disable "TOKENIZERS_PARALLELISM" to depress the warnings.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        pbar = None if self.disable_tqdm else tqdm(total=len(request_records))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as pool:
            futures = [
                pool.submit(
                    FixTimestampExecutor._process_task,
                    self.f_create_api_endpoint,
                    partition,
                    base_timestamp,
                    base_sys_time,
                    self.max_schedule_gap,
                )
                for partition in partitions
            ]
            results: List[RequestRecord] = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                results.extend(future.result())
                if pbar is not None:
                    pbar.update(len(partitions[i]))

        return results

    @staticmethod
    def _process_task(
        f_create_api_endpoint: Callable[[], APIEndPoint],
        request_records: List[RequestRecord],
        base_timestamp: float,
        base_sys_time: float,
        max_schedule_gap: float,
    ) -> List[RequestRecord]:
        if len(request_records) == 0:
            return []

        async def process_task_impl(
            f_create_api_endpoint: Callable[[], APIEndPoint],
            request_records: List[RequestRecord],
            base_timestamp: float,
            base_sys_time: float,
            max_schedule_gap: float,
        ) -> List[RequestRecord]:
            api_endpoint = f_create_api_endpoint()
            loop = asyncio.get_running_loop()
            # Get the delta time to convert system time to the loop time.
            # We must use the system time `time.time()` which is consistent across processes.
            loop_sys_delta_time = loop.time() - time.time()
            updated_request_records: List[RequestRecord] = []
            async with api_endpoint:

                async def _task(request_record: RequestRecord) -> None:
                    updated_request_records.append(await api_endpoint(request_record))

                tasks = []
                for request_record in request_records:
                    launch_time = (
                        (request_record.timestamp - base_timestamp)
                        + (base_sys_time + max_schedule_gap)
                        + loop_sys_delta_time
                    )
                    loop.call_at(
                        launch_time,
                        lambda record: tasks.append(asyncio.create_task(_task(record))),
                        request_record,
                    )
                    # Sleep to allow runs of other scheduled tasks if any.
                    await asyncio.sleep(max(launch_time - loop.time() - max_schedule_gap, 0))

                # Sleep until all the tasks are launched.
                await asyncio.sleep(launch_time - loop.time() + max_schedule_gap)
                # Wait for all tasks to be scheduled
                assert len(tasks) == len(request_records)
                await asyncio.gather(*tasks)

            assert len(updated_request_records) == len(request_records)
            return updated_request_records

        return asyncio.run(
            process_task_impl(
                f_create_api_endpoint,
                request_records,
                base_timestamp,
                base_sys_time,
                max_schedule_gap,
            )
        )


def create_pipelines(  # pylint: disable=too-many-branches
    args: argparse.Namespace, f_create_api_endpoint: Callable[[], APIEndPoint], dataset: Dataset
) -> List[RequestProcessor]:
    """Creating request processing pipelines with regard to the specified args."""
    cuda_profile_url = f"http://{args.host}:{args.port}" if args.cuda_profile else None
    pipelines: List[RequestProcessor] = []
    if args.num_concurrent_requests is not None:
        if args.request_rate is not None:
            raise ValueError(
                'Both "num_concurrent_requests" and "request_rate" are specified. '
                "Please specify only one of them."
            )
        if args.replay_timestamp_scale is not None:
            raise ValueError(
                "Dataset replay is unsupported when fixing number of concurrent requests."
            )
        for num_concurrent_requests in args.num_concurrent_requests:
            num_warmup_requests = (
                args.num_warmup_requests
                if args.num_warmup_requests is not None
                else num_concurrent_requests
            )
            pipelines.append(
                SequentialProcessor(
                    LogMessage(f"Fixing number of concurrent requests: {num_concurrent_requests}"),
                    SampleRequests(args.num_requests + num_warmup_requests),
                    AttachModelName(args.tokenizer),
                    AttachStreamFlag(args.stream),
                    AttachSamplingOptions(args.temperature, args.top_p, args.ignore_eos),
                    AttachExecutionFeature({"num_concurrent_requests": num_concurrent_requests}),
                    WarmupAndRun(
                        num_warmup_requests=num_warmup_requests,
                        num_benchmark_requests=args.num_requests,
                        pipeline=FixedConcurrentRequestExecutor(
                            f_create_api_endpoint,
                            args.num_process_workers,
                            args.disable_tqdm,
                            num_concurrent_requests,
                            args.multi_round,
                        ),
                        cuda_profile_url=cuda_profile_url,
                        fake_warmup=dataset.require_fake_warmup,
                    ),
                )
            )
        return pipelines
    if args.request_rate is not None:
        if args.num_warmup_requests is None:
            raise ValueError(
                "Please specify the number of warmup requests via "
                '"--num-warmup-requests" when fixing request rate.'
            )
        if args.replay_timestamp_scale is not None:
            raise ValueError("Dataset replay is unsupported when fixing request rates.")
        num_total_requests = int(
            args.num_requests if not args.per_gpu_workload else args.num_requests * args.num_gpus
        )
        if dataset.require_fake_warmup:
            num_samples = num_total_requests
        else:
            num_samples = num_total_requests + args.num_warmup_requests
        return [
            SequentialProcessor(
                LogMessage(f"Fixing request rate: {request_rate}"),
                SampleRequests(num_samples),
                AttachModelName(args.tokenizer),
                AttachRequestRateTimestamp(
                    request_rate if not args.per_gpu_workload else request_rate * args.num_gpus
                ),
                AttachStreamFlag(args.stream),
                AttachSamplingOptions(args.temperature, args.top_p, args.ignore_eos),
                AttachExecutionFeature({"request_rate": float(request_rate)}),
                WarmupAndRun(
                    num_warmup_requests=args.num_warmup_requests,
                    num_benchmark_requests=num_total_requests,
                    pipeline=FixTimestampExecutor(
                        f_create_api_endpoint,
                        args.num_process_workers,
                        args.disable_tqdm,
                        args.max_schedule_gap,
                        args.num_requests,
                    ),
                    cuda_profile_url=cuda_profile_url,
                    fake_warmup=dataset.require_fake_warmup,
                ),
            )
            for request_rate in args.request_rate
        ]

    # Default: dataset replay mode
    # The dataset must come with timestamps.
    if not dataset.timestamp_available:
        raise ValueError(
            "The dataset does not have timestamps, so dataset replay is unsupported. "
            'Please specify one of "num_concurrent_requests" '
            'and "request_rate".'
        )
    if args.per_gpu_workload:
        raise ValueError("Fixing per-GPU workload is not compatible with dataset replay.")
    if args.num_warmup_requests is None:
        raise ValueError(
            "Please specify the number of warmup requests via "
            '"--num-warmup-requests" for dataset replay.'
        )
    timestamp_scale = args.replay_timestamp_scale or 1.0
    if dataset.require_fake_warmup:
        num_samples = args.num_requests
    else:
        num_samples = args.num_requests + args.num_warmup_requests
    return [
        SequentialProcessor(
            LogMessage(f"Dataset replay with time scaling of {timestamp_scale}"),
            SampleRequests(num_samples, take_first_x_requests=True),
            AttachModelName(args.tokenizer),
            ScaleTimestamp(timestamp_scale),
            AttachStreamFlag(args.stream),
            AttachSamplingOptions(args.temperature, args.top_p, args.ignore_eos),
            AttachExecutionFeature({"timestamp_scale": timestamp_scale}),
            WarmupAndRun(
                num_warmup_requests=args.num_warmup_requests,
                num_benchmark_requests=args.num_requests,
                pipeline=FixTimestampExecutor(
                    f_create_api_endpoint,
                    args.num_process_workers,
                    args.disable_tqdm,
                    args.max_schedule_gap,
                    args.num_requests,
                ),
                cuda_profile_url=cuda_profile_url,
                fake_warmup=dataset.require_fake_warmup,
            ),
        )
    ]
