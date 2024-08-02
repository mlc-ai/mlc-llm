"""MLC LLM benchmark executor classes"""

import argparse
import asyncio
import concurrent.futures
import os
from typing import Any, Callable, Dict, List, Optional

from tqdm.asyncio import tqdm

from mlc_llm.bench.api_endpoint import APIEndPoint
from mlc_llm.bench.request_record import RequestRecord


class Executor:
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

    async def run_benchmark(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        """Run benchmark with the given requests."""
        raise NotImplementedError()

    async def warmup(self, warmup_requests: List[RequestRecord]) -> None:
        """Run warmup with the given requests."""
        raise NotImplementedError()

    def get_num_warmup_requests(self) -> int:
        """Return the number of warmup requests needed by the executor."""
        raise NotImplementedError()

    def get_executor_feature_dict(self) -> Dict[str, Any]:
        """Return the features of the executor."""
        raise NotImplementedError()


class FixedConcurrentRequestExecutor(Executor):
    """The benchmark executor of fixing the number of concurrent requests."""

    num_concurrent_requests: int

    def __init__(
        self,
        f_create_api_endpoint: Callable[[], APIEndPoint],
        num_processes: Optional[int],
        disable_tqdm: bool,
        num_concurrent_requests: int,
    ) -> None:
        if num_processes is None:
            # We assign each process at most 32 concurrent requests to send
            # so that the asyncio pressure will not be too much.
            num_processes = min((num_concurrent_requests + 31) // 32, 16)
        super().__init__(f_create_api_endpoint, num_processes, disable_tqdm)
        self.num_concurrent_requests = num_concurrent_requests

    async def run_benchmark(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        partitions: List[List[RequestRecord]] = [
            request_records[slice(i, len(request_records), self.num_processes)]
            for i in range(self.num_processes)
        ]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        loop = asyncio.get_running_loop()
        pbar = None if self.disable_tqdm else tqdm(total=len(request_records))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as pool:
            futures = [
                loop.run_in_executor(
                    pool,
                    FixedConcurrentRequestExecutor._process_task,
                    self.f_create_api_endpoint,
                    partition,
                    self.num_concurrent_requests // self.num_processes
                    + int(i < self.num_concurrent_requests % self.num_processes),
                )
                for i, partition in enumerate(partitions)
            ]
            results: List[RequestRecord] = []
            for i, future in enumerate(asyncio.as_completed(futures)):
                results.extend(await future)
                if pbar is not None:
                    pbar.update(len(partitions[i]))

        return results

    @staticmethod
    def _process_task(
        f_create_api_endpoint: Callable[[], APIEndPoint],
        request_records: List[RequestRecord],
        num_concurrent_requests: int,
    ) -> List[RequestRecord]:
        if len(request_records) == 0:
            return []

        async def process_task_impl(
            f_create_api_endpoint: Callable[[], APIEndPoint],
            request_records: List[RequestRecord],
            num_concurrent_requests: int,
        ) -> List[RequestRecord]:
            api_endpoint = f_create_api_endpoint()
            updated_request_records: List[RequestRecord] = [None for _ in request_records]
            async with api_endpoint:
                num_sent_request = 0

                async def _task() -> None:
                    nonlocal num_sent_request
                    while True:
                        if num_sent_request == len(request_records):
                            break
                        idx = num_sent_request
                        num_sent_request += 1
                        request = request_records[idx]

                        updated_request_records[idx] = await api_endpoint(request)

                tasks = [asyncio.create_task(_task()) for _ in range(num_concurrent_requests)]
                await asyncio.gather(*tasks)

            return updated_request_records

        return asyncio.run(
            process_task_impl(
                f_create_api_endpoint,
                request_records,
                num_concurrent_requests,
            )
        )

    async def warmup(self, warmup_requests: List[RequestRecord]) -> None:
        # Disable tqdm for warmup
        disable_tqdm = self.disable_tqdm
        self.disable_tqdm = True
        await self.run_benchmark(warmup_requests)
        self.disable_tqdm = disable_tqdm

    def get_num_warmup_requests(self) -> int:
        return self.num_concurrent_requests

    def get_executor_feature_dict(self) -> Dict[str, Any]:
        return {"num_concurrent_requests": self.num_concurrent_requests}


# Todo: Timestamp executor for fixed request rate or log replay  # pylint: disable=fixme
# class FixTimestampExecutor(Executor):
#     pass


def create_executors(
    args: argparse.Namespace,
    f_create_api_endpoint: Callable[[], APIEndPoint],
) -> List[Executor]:
    """Create executor instances with regard to the specified args and endpoint."""
    if args.num_concurrent_requests is not None:
        if args.request_rate is not None:
            raise ValueError(
                'Both "num_concurrent_requests" and "request_rate" are specified. '
                "Please specify only one of them."
            )
        return [
            FixedConcurrentRequestExecutor(
                f_create_api_endpoint,
                args.num_process_workers,
                args.disable_tqdm,
                num_concurrent_requests,
            )
            for num_concurrent_requests in args.num_concurrent_requests
        ]
    if args.request_rate is not None:
        raise NotImplementedError('"FixTimestampExecutor" is yet to be implemented.')
    raise ValueError(
        'Unable to create executor. Please specify one of "num_concurrent_requests" '
        'and "request_rate".'
    )
