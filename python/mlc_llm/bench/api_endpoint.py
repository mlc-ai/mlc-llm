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


class OpenAIChatEndPoint(APIEndPoint):
    """The backend of sending HTTP requests in OpenAI API through "v1/chat/completions"."""

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

        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        self, request_record: RequestRecord
    ) -> RequestRecord:
        payload = request_record.chat_cmpl.model_dump()
        if self.timeout is not None and "timeout" not in payload:
            payload["timeout"] = self.timeout
        if self.include_server_metrics:
            if "stream_options" not in payload or payload["stream_options"] is None:
                payload["stream_options"] = {"include_usage": True}
            else:
                payload["stream_options"]["include_usage"] = True
        if (
            request_record.chat_cmpl.debug_config is not None
            and request_record.chat_cmpl.debug_config.ignore_eos
        ):
            payload["ignore_eos"] = True

        generated_text = ""
        first_chunk_output_str = ""
        time_to_first_token_s = None
        start_time = time.monotonic()
        server_metrics = None

        try:
            async with self.client.post(self.url, json=payload, headers=self.headers) as response:
                assert response.status == 200, await response.text()
                if payload["stream"]:
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
                        content = delta.get("content", None)
                        if content is not None and not time_to_first_token_s:
                            time_to_first_token_s = time.monotonic() - start_time
                            first_chunk_output_str = content
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

                        if content is not None:
                            generated_text += content
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
            error_msg = "API endpoint errored when sending request: " + traceback.format_exc()
            logger.info(error_msg)
            finish_time = time.monotonic()
            request_record.output_str = generated_text
            request_record.first_chunk_output_str = first_chunk_output_str
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
            request_record.error_msg = error_msg
            return request_record

        finish_time = time.monotonic()
        request_record.output_str = generated_text
        request_record.first_chunk_output_str = first_chunk_output_str
        success = True
        error_msg = None
        if len(generated_text) == 0:
            success = False
            error_msg = "Empty generated text."
        request_record.metrics = Metrics(
            success=success,
            start_time=start_time,
            finish_time=finish_time,
            end_to_end_latency_s=finish_time - start_time,
            input_tokens=request_record.metrics.input_tokens,
            time_to_first_token_s=time_to_first_token_s,
            server_metrics=server_metrics,
            exec_feature=request_record.metrics.exec_feature,
        )
        request_record.error_msg = error_msg
        return request_record


class OpenAIEndPoint(APIEndPoint):
    """The backend of sending HTTP requests in OpenAI API through "v1/completions"."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        include_server_metrics: bool = False,
        no_debug_config: bool = False,
    ) -> None:
        super().__init__(include_server_metrics=include_server_metrics)

        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.timeout = timeout
        self.client: aiohttp.ClientSession = None
        self.url = f"http://{host}:{port}/v1/completions"
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("MLC_LLM_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('MLC_LLM_API_KEY')}"
        assert (
            not include_server_metrics
        ), '"include_server_metrics" only works for "openai-chat" endpoint for now'
        self.no_debug_config = no_debug_config

    async def __aenter__(self) -> Self:
        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.client = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-branches,too-many-statements
        self, request_record: RequestRecord
    ) -> RequestRecord:
        assert (
            len(request_record.chat_cmpl.messages) == 1
        ), 'Endpoint "openai" does not support system prompt and multi-round conversation.'
        assert isinstance(request_record.chat_cmpl.messages[0].content, str)
        payload = {
            "model": request_record.chat_cmpl.model,
            "prompt": request_record.chat_cmpl.messages[0].content,
            "temperature": request_record.chat_cmpl.temperature,
            "top_p": request_record.chat_cmpl.top_p,
            "max_tokens": request_record.chat_cmpl.max_tokens,
            "stream": True,
        }
        if self.timeout is not None and "timeout" not in payload:
            payload["timeout"] = self.timeout
        if (
            request_record.chat_cmpl.debug_config is not None
            and request_record.chat_cmpl.debug_config.ignore_eos
        ):
            payload["ignore_eos"] = True
            if not self.no_debug_config:
                payload["debug_config"] = {"ignore_eos": True}

        generated_text = ""
        first_chunk_output_str = ""
        time_to_first_token_s = None
        start_time = time.monotonic()

        try:
            async with self.client.post(
                self.url, json=payload, headers=self.headers, timeout=3600
            ) as response:
                assert response.status == 200, await response.text()
                if payload["stream"]:
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
                        content = data["choices"][0]["text"]
                        if content is not None and not time_to_first_token_s:
                            time_to_first_token_s = time.monotonic() - start_time
                            first_chunk_output_str = content
                        if content is not None:
                            generated_text += content
                else:
                    data = await response.json()
                    generated_text = data["choices"][0]["message"]["content"]
        except Exception:  # pylint: disable=broad-except
            error_msg = "API endpoint errored when sending request: " + traceback.format_exc()
            logger.info(error_msg)
            finish_time = time.monotonic()
            request_record.output_str = generated_text
            request_record.first_chunk_output_str = first_chunk_output_str
            request_record.metrics = Metrics(
                success=False,
                start_time=start_time,
                finish_time=finish_time,
                end_to_end_latency_s=finish_time - start_time,
                input_tokens=request_record.metrics.input_tokens,
                time_to_first_token_s=time_to_first_token_s,
                server_metrics=None,
                exec_feature=request_record.metrics.exec_feature,
            )
            request_record.error_msg = error_msg
            return request_record

        finish_time = time.monotonic()
        request_record.output_str = generated_text
        request_record.first_chunk_output_str = first_chunk_output_str
        success = True
        error_msg = None
        if len(generated_text) == 0:
            success = False
            error_msg = "Empty generated text."
        request_record.metrics = Metrics(
            success=success,
            start_time=start_time,
            finish_time=finish_time,
            end_to_end_latency_s=finish_time - start_time,
            input_tokens=request_record.metrics.input_tokens,
            time_to_first_token_s=time_to_first_token_s,
            server_metrics=None,
            exec_feature=request_record.metrics.exec_feature,
        )
        request_record.error_msg = error_msg
        return request_record


class TensorRTLLMEndPoint(APIEndPoint):
    """The backend of sending HTTP requests in TensorRT-LLM API."""

    def __init__(  # pylint: disable=too-many-arguments
        self, host: str, port: int, timeout: Optional[float] = None
    ) -> None:
        super().__init__(include_server_metrics=False)

        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.timeout = timeout
        self.client: aiohttp.ClientSession = None
        self.url_stream = f"http://{host}:{port}/v2/models/ensemble/generate_stream"
        self.url_no_stream = f"http://{host}:{port}/v2/models/ensemble/generate"

    async def __aenter__(self) -> Self:
        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.client = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        self, request_record: RequestRecord
    ) -> RequestRecord:
        assert len(request_record.chat_cmpl.messages) == 1
        assert isinstance(request_record.chat_cmpl.messages[0].content, str)
        payload = {
            "accumulate_tokens": True,
            "text_input": request_record.chat_cmpl.messages[0].content,
            "temperature": (
                max(request_record.chat_cmpl.temperature, 1e-5)
                if request_record.chat_cmpl.temperature
                else 1
            ),
            "top_p": request_record.chat_cmpl.top_p if request_record.chat_cmpl.top_p else 1,
            "max_tokens": request_record.chat_cmpl.max_tokens,
            "stream": request_record.chat_cmpl.stream,
        }
        if (
            request_record.chat_cmpl.debug_config is not None
            and request_record.chat_cmpl.debug_config.ignore_eos
        ):
            payload["min_length"] = payload["max_tokens"]
        if self.timeout is not None and "timeout" not in payload:
            payload["timeout"] = self.timeout

        generated_text = ""
        first_chunk_output_str = ""
        url = self.url_stream if request_record.chat_cmpl.stream else self.url_no_stream
        time_to_first_token_s = None
        start_time = time.monotonic()

        try:
            async with self.client.post(url, json=payload) as response:
                assert response.status == 200, await response.text()
                if payload["stream"]:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk or chunk == b"\n":
                            continue
                        # Get rid of the prefix "data:" and suffix "\n"
                        raw_data = chunk[5:].strip()
                        data = json.loads(raw_data)
                        delta = data["text_output"]
                        if delta is None:
                            continue

                        if not time_to_first_token_s:
                            time_to_first_token_s = time.monotonic() - start_time
                            first_chunk_output_str = delta
                        generated_text += delta
                else:
                    data = await response.json()
                    generated_text = data["text_output"]
        except Exception:  # pylint: disable=broad-except
            error_msg = "API endpoint errored when sending request: " + traceback.format_exc()
            logger.info(error_msg)
            finish_time = time.monotonic()
            request_record.output_str = generated_text
            request_record.first_chunk_output_str = first_chunk_output_str
            request_record.metrics = Metrics(
                success=False,
                start_time=start_time,
                finish_time=finish_time,
                end_to_end_latency_s=finish_time - start_time,
                input_tokens=request_record.metrics.input_tokens,
                time_to_first_token_s=time_to_first_token_s,
                exec_feature=request_record.metrics.exec_feature,
            )
            request_record.error_msg = error_msg
            return request_record

        finish_time = time.monotonic()
        request_record.output_str = generated_text
        request_record.first_chunk_output_str = first_chunk_output_str
        success = True
        error_msg = None
        if len(generated_text) == 0:
            success = False
            error_msg = "Empty generated text."
        request_record.metrics = Metrics(
            success=success,
            start_time=start_time,
            finish_time=finish_time,
            end_to_end_latency_s=finish_time - start_time,
            input_tokens=request_record.metrics.input_tokens,
            time_to_first_token_s=time_to_first_token_s,
            exec_feature=request_record.metrics.exec_feature,
        )
        request_record.error_msg = error_msg
        return request_record


# Todo: APIEndPoint with AsyncOpenAI Python interface  # pylint: disable=fixme
# class OpenAIPythonEndPoint(APIEndPoint):
#     pass

SUPPORTED_BACKENDS = [
    "openai",
    "openai-chat",
    "mlc",
    "sglang",
    "tensorrt-llm",
    "vllm",
]


def create_api_endpoint(args: argparse.Namespace) -> APIEndPoint:
    """Create an API endpoint instance with regard to the specified endpoint kind."""
    if args.api_endpoint in ["openai", "mlc", "sglang"]:
        return OpenAIEndPoint(args.host, args.port, args.timeout, args.include_server_metrics)
    if args.api_endpoint == "vllm":
        return OpenAIEndPoint(
            args.host, args.port, args.timeout, include_server_metrics=False, no_debug_config=True
        )
    if args.api_endpoint == "openai-chat":
        return OpenAIChatEndPoint(args.host, args.port, args.timeout, args.include_server_metrics)
    if args.api_endpoint == "tensorrt-llm":
        return TensorRTLLMEndPoint(args.host, args.port, args.timeout)
    raise ValueError(f'Unrecognized endpoint "{args.api_endpoint}"')
