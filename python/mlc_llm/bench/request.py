"""MLC LLM Bench Request"""
import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import Self

from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.support import logging

from .prompts import PromptsGenerator

logging.enable_logging()
logger = logging.getLogger(__name__)


class RequestRecords(BaseModel):
    """The request records collected from LLM inference requests."""

    input: str
    output: str
    end_to_end_latency_s: float
    ttft: Optional[float] = None
    server_metrics: Optional[Dict] = None


class OpenAIRequestSender:  # pylint: disable=too-many-instance-attributes
    """
    Manages the sending of requests to a specified API endpoint and gathers inference statistics.

    Parameters
    ----------
    host : Optional[str]
        The host address for the API, defaulting to "127.0.0.1".
    port : Optional[int]
        The port number for the API, defaulting to 8008.
    stream : Optional[bool]
        Specifies if streaming should be enabled, default is True.
    timeout : Optional[float]
        The maximum duration in seconds for each request, default is 180.
    client : Optional[Any]
        The client to use for sending requests.
    include_server_metrics : Optional[bool]
        Specifies if server metrics should be included, default is False.

    Attributes
    ----------
    stats : dict
        Statistics about the performance.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 8008,
        stream: Optional[bool] = None,
        timeout: Optional[float] = None,
        client: Optional[Any] = None,
        include_server_metrics: Optional[bool] = False,
    ) -> None:
        import aiohttp  # pylint: disable=import-outside-toplevel,import-error
        from transformers import (  # pylint: disable=import-outside-toplevel,import-error
            LlamaTokenizerFast,
        )

        self.stream = stream
        self.timeout = timeout
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.prompt_generator = PromptsGenerator()
        self.request_records: List[RequestRecords] = []
        self.client = client if client else aiohttp.ClientSession()
        self.include_server_metrics = include_server_metrics
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("MLC_LLM_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('MLC_LLM_API_KEY')}"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, params: Dict[str, Any] = None
    ) -> None:
        if "messages" not in params:
            prompt_tokens = 128
            if "prompt_tokens" in params:
                prompt_tokens = params["prompt_tokens"]
            else:
                logger.warning("A random prompt with %d tokens will be generated.", prompt_tokens)
            prompt = self.prompt_generator.generate_prompt(prompt_tokens)
            params["messages"] = [{"role": "system", "content": prompt}]
        else:
            prompt = params["messages"][-1]["content"]
        chat_params = self._get_chat_completion_params(params)
        if "stream" not in chat_params:
            chat_params["stream"] = self.stream
        if "timeout" not in chat_params:
            chat_params["timeout"] = self.timeout
        if self.include_server_metrics:
            if "stream_options" not in chat_params:
                chat_params["stream_options"] = {"include_usage": True}
            else:
                chat_params["stream_options"]["include_usage"] = True

        total_request_time = 0
        generated_text = ""
        ttft = None
        start_time = time.monotonic()
        server_metrics = None

        # AsyncOpenAI chat completion
        if isinstance(self.client, AsyncOpenAI):
            response = await self.client.chat.completions.create(**chat_params)
            if chat_params["stream"]:
                async for chunk in response:
                    if chunk.usage:
                        server_metrics = chunk.usage.extra
                    elif chunk.choices[0].delta.content is not None:
                        if not ttft:
                            ttft = time.monotonic() - start_time  # type: ignore
                        generated_text += chunk.choices[0].delta.content
            else:
                generated_text = response.choices[0].message.content
        else:
            try:
                async with self.client.post(
                    self.url, json=chat_params, headers=self.headers
                ) as response:
                    if chat_params["stream"]:
                        async for chunk in response.content:
                            chunk = chunk.strip()
                            if not chunk or chunk == b"\n":
                                continue
                            # Get rid of the prefix "data: " and suffix "\n"
                            raw_data = chunk[6:].strip()
                            if raw_data == b"[DONE]":
                                continue
                            data = json.loads(raw_data)
                            if data["usage"] is not None:
                                server_metrics = data["usage"]["extra"]
                            if not data["choices"]:
                                continue
                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                if not ttft:
                                    ttft = time.monotonic() - start_time

                            generated_text += delta["content"]
                    else:
                        data = await response.json()
                        generated_text = data["choices"][0]["message"]["content"]
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error sending request: %s", str(e))
                raise e

        total_request_time = time.monotonic() - start_time  # type: ignore

        req_rec = RequestRecords(
            input=prompt,
            output=generated_text,
            end_to_end_latency_s=total_request_time,
            ttft=ttft,
            server_metrics=server_metrics,
        )
        self.request_records.append(req_rec)

    def _get_chat_completion_params(self, params: Dict) -> Dict:
        """
        Extract chat completion parameters from the provided request parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the request.

        Returns
        -------
        result : Dict
            The chat completion parameters.
        """
        chat_completion_params = {}
        for k, _ in ChatCompletionRequest.model_fields.items():
            if k in params:
                chat_completion_params[k] = params[k]
        return chat_completion_params

    def get_request_records(self) -> List[RequestRecords]:
        """
        Retrieve the collected reqeust records.

        Returns
        -------
        request_records : List[RequestRecords]
            The list of collected request records.
        """
        return self.request_records
