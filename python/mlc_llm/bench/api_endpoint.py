"""MLC LLM bench backends"""

import argparse
import json
import os
import time
import traceback
from typing import Optional

from typing_extensions import Self

from mlc_llm.bench.request_record import Metrics, RequestRecord, ServerMetrics
from mlc_llm.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)


class APIEndPoint:
    """Manages the sending of requests to a specified API endpoint and gathers
    inference statistics.
    """

    def __init__(self, include_server_metrics: bool = False) -> None:
        self.include_server_metrics = include_server_metrics

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        pass

    async def __call__(self, request: RequestRecord) -> RequestRecord:
        raise NotImplementedError()


class OpenAIEndPoint(APIEndPoint):
    """The backend of sending HTTP requests in OpenAI API."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        include_server_metrics: bool = False,
    ) -> None:
        super().__init__(include_server_metrics=include_server_metrics)

        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.timeout = timeout
        self.client: aiohttp.ClientSession = None
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("MLC_LLM_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('MLC_LLM_API_KEY')}"

    async def __aenter__(self) -> Self:
        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.client = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-branches
        self, request_record: RequestRecord
    ) -> RequestRecord:
        kwargs = request_record.chat_cmpl.model_dump()
        if self.timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        if self.include_server_metrics:
            if "stream_options" not in kwargs or kwargs["stream_options"] is None:
                kwargs["stream_options"] = {"include_usage": True}
            else:
                kwargs["stream_options"]["include_usage"] = True
        kwargs["ignore_eos"] = True

        generated_text = ""
        time_to_first_token_s = None
        start_time = time.monotonic()
        server_metrics = None

        try:
            async with self.client.post(self.url, json=kwargs, headers=self.headers) as response:
                if kwargs["stream"]:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk or chunk == b"\n":
                            continue
                        # Get rid of the prefix "data: " and suffix "\n"
                        raw_data = chunk[6:].strip()
                        if raw_data == b"[DONE]":
                            continue
                        data = json.loads(raw_data)
                        if not data["choices"]:
                            continue
                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None) and not time_to_first_token_s:
                            time_to_first_token_s = time.monotonic() - start_time
                        if self.include_server_metrics and data["usage"] is not None:
                            # fmt: off
                            # pylint: disable=line-too-long
                            server_metrics = ServerMetrics(
                                input_tokens=data["usage"]["extra"]["prompt_tokens"],
                                prefill_tokens=data["usage"]["extra"]["prefill_tokens"],
                                output_tokens=data["usage"]["extra"]["completion_tokens"],
                                end_to_end_latency_s=data["usage"]["extra"]["end_to_end_latency_s"],
                                prefill_tokens_per_s=data["usage"]["extra"]["prefill_tokens_per_s"],
                                inter_token_latency_s=data["usage"]["extra"]["inter_token_latency_s"],
                                time_per_output_token_s=1 / data["usage"]["extra"]["decode_tokens_per_s"],
                                time_to_first_token_s=data["usage"]["extra"]["ttft_s"],
                            )
                            # pylint: enable=line-too-long
                            # fmt: on

                        generated_text += delta["content"]
                else:
                    data = await response.json()
                    generated_text = data["choices"][0]["message"]["content"]
                    if self.include_server_metrics and data["usage"] is not None:
                        # fmt: off
                        # pylint: disable=line-too-long
                        server_metrics = ServerMetrics(
                            input_tokens=data["usage"]["extra"]["prompt_tokens"],
                            prefill_tokens=data["usage"]["extra"]["prefill_tokens"],
                            output_tokens=data["usage"]["extra"]["completion_tokens"],
                            end_to_end_latency_s=data["usage"]["extra"]["end_to_end_latency_s"],
                            prefill_tokens_per_s=data["usage"]["extra"]["prefill_tokens_per_s"],
                            inter_token_latency_s=data["usage"]["extra"]["inter_token_latency_s"],
                            time_per_output_token_s=1 / data["usage"]["extra"]["decode_tokens_per_s"],
                            time_to_first_token_s=data["usage"]["extra"]["ttft_s"],
                        )
                        # pylint: enable=line-too-long
                        # fmt: on
        except Exception:  # pylint: disable=broad-except
            logger.info("Error sending request: %s", traceback.format_exc())
            finish_time = time.monotonic()
            request_record.output_str = generated_text
            request_record.metrics = Metrics(
                success=False,
                start_time=start_time,
                finish_time=finish_time,
                end_to_end_latency_s=finish_time - start_time,
                input_tokens=request_record.metrics.input_tokens,
                time_to_first_token_s=time_to_first_token_s,
                server_metrics=server_metrics,
                exec_feature=request_record.metrics.exec_feature,
            )
            return request_record

        finish_time = time.monotonic()
        request_record.output_str = generated_text
        request_record.metrics = Metrics(
            success=len(generated_text) > 0,
            start_time=start_time,
            finish_time=finish_time,
            end_to_end_latency_s=finish_time - start_time,
            input_tokens=request_record.metrics.input_tokens,
            time_to_first_token_s=time_to_first_token_s,
            server_metrics=server_metrics,
            exec_feature=request_record.metrics.exec_feature,
        )
        return request_record


# Todo: APIEndPoint with AsyncOpenAI Python interface  # pylint: disable=fixme
# class OpenAIPythonEndPoint(APIEndPoint):
#     pass

SUPPORTED_BACKENDS = [
    "openai",
]


def create_api_endpoint(args: argparse.Namespace) -> APIEndPoint:
    """Create an API endpoint instance with regard to the specified endpoint kind."""
    if args.api_endpoint == "openai":
        return OpenAIEndPoint(args.host, args.port, args.timeout, args.include_server_metrics)
    raise ValueError(f'Unrecognized endpoint "{args.api_endpoint}"')
