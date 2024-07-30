"""MLC LLM benchmark executor classes"""

import argparse
import asyncio
import time
from typing import Any, Dict, List, Tuple

from tqdm.asyncio import tqdm

from mlc_llm.bench.api_endpoint import APIEndPoint
from mlc_llm.bench.request_record import RequestRecord


class Executor:
    """The executor base class, denoting the kind of benchmark mode."""

    api_endpoint: APIEndPoint

    def __init__(self, api_endpoint: APIEndPoint, disable_tqdm: bool) -> None:
        self.api_endpoint = api_endpoint
        self.disable_tqdm = disable_tqdm
        self.pbar = None

    async def run_benchmark(
        self, request_records: List[RequestRecord]
    ) -> Tuple[List[RequestRecord], float]:
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

    def _init_progress_bar(self, num_requests: int) -> None:
        """Run warmup with the given requests."""
        self.pbar = tqdm(total=num_requests) if not self.disable_tqdm else None

    def _update_progress_bar(self) -> None:
        if self.pbar is not None:
            self.pbar.update(1)

    def _terminate_progress_bar(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class FixedConcurrentRequestExecutor(Executor):
    """The benchmark executor of fixing the number of concurrent requests."""

    num_concurrent_requests: int

    def __init__(
        self,
        api_endpoint: APIEndPoint,
        disable_tqdm: bool,
        num_concurrent_requests: int,
    ) -> None:
        super().__init__(api_endpoint, disable_tqdm)
        self.num_concurrent_requests = num_concurrent_requests

    async def run_benchmark(
        self, request_records: List[RequestRecord]
    ) -> Tuple[List[RequestRecord], float]:
        updated_request_records: List[RequestRecord] = [None for _ in request_records]
        async with self.api_endpoint:
            num_sent_request = 0

            async def _task() -> None:
                nonlocal num_sent_request
                while True:
                    if num_sent_request == len(request_records):
                        break
                    idx = num_sent_request
                    num_sent_request += 1
                    request = request_records[idx]

                    updated_request_records[idx] = await self.api_endpoint(request)
                    self._update_progress_bar()

            tasks = [asyncio.create_task(_task()) for _ in range(self.num_concurrent_requests)]
            self._init_progress_bar(len(request_records))

            start_time = time.monotonic()
            await asyncio.gather(*tasks)
            end_time = time.monotonic()

            self._terminate_progress_bar()
        return updated_request_records, end_time - start_time

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
    api_endpoint: APIEndPoint,
) -> List[Executor]:
    """Create executor instances with regard to the specified args and endpoint."""
    if args.num_concurrent_requests is not None:
        if args.request_rate is not None:
            raise ValueError(
                'Both "num_concurrent_requests" and "request_rate" are specified. '
                "Please specify only one of them."
            )
        return [
            FixedConcurrentRequestExecutor(api_endpoint, args.disable_tqdm, num_concurrent_requests)
            for num_concurrent_requests in args.num_concurrent_requests
        ]
    if args.request_rate is not None:
        raise NotImplementedError('"FixTimestampExecutor" is yet to be implemented.')
    raise ValueError(
        'Unable to create executor. Please specify one of "num_concurrent_requests" '
        'and "request_rate".'
    )
