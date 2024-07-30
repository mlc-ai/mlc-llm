"""MLC LLM Bench Request"""

import random
from typing import List, Optional

from transformers import AutoTokenizer  # pylint: disable=import-error

from mlc_llm.bench.request_record import RequestRecord


class RequestProcessor:  # pylint: disable=too-few-public-methods
    """The request processor base class.
    Each processor can take a list of RequestRecord, applying the process,
    and returning the processed RequestRecord in the end.
    """

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        raise NotImplementedError()


class SampleRequests(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that samples requests out from the given request list."""

    def __init__(self, num_requests: int) -> None:
        self.num_requests = num_requests

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        return random.sample(request_records, self.num_requests)


class AttachTimestamp(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that applies timestamps to the requests."""

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        raise NotImplementedError()


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


class MetricAnalyzer(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that analyzes the raw benchmark results and computes more detailed metrics."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        updated_records = []
        for request_record in request_records:
            metrics = request_record.metrics
            if not metrics.success:
                continue

            metrics.output_tokens = len(self.tokenizer.encode(request_record.output_str))
            assert metrics.input_tokens > 0 and metrics.output_tokens > 0, "Invalid prompt tokens"
            metrics.inter_token_latency_s = metrics.end_to_end_latency_s / metrics.output_tokens
            if metrics.time_to_first_token_s is None:
                metrics.time_to_first_token_s = 0
            metrics.time_per_output_token_s = (
                metrics.end_to_end_latency_s - metrics.time_to_first_token_s
            ) / (metrics.output_tokens - 1)
            updated_records.append(request_record)
        return updated_records


class SequentialProcessor(RequestProcessor):  # pylint: disable=too-few-public-methods
    """The processor that sequentially applies a list of processors in order."""

    processors: List[RequestProcessor]

    def __init__(self, *processors: RequestProcessor) -> None:
        self.processors = list(processors)

    def __call__(self, request_records: List[RequestRecord]) -> List[RequestRecord]:
        for processor in self.processors:
            request_records = processor(request_records)
        return request_records
