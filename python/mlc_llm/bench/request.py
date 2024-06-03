"""MLC LLM Bench Request"""
import time
from typing import Any, Dict, List, Optional

import httpx
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
    end_to_end_latency: float
    ttft: Optional[float] = 0


class OpenAIRequestSender:
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

    Attributes
    ----------
    stats : dict
        Statistics about the performance.
    """

    def __init__(
        self,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 8008,
        stream: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        from transformers import (  # pylint: disable=import-outside-toplevel,import-error
            LlamaTokenizerFast,
        )

        self.stream = stream
        self.timeout = timeout
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.prompt_generator = PromptsGenerator()
        self.metrics: List[RequestRecords] = []
        self.client = AsyncOpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="None",
            http_client=httpx.AsyncClient(http2=True),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.client.close()

    async def __call__(self, params: Dict[str, Any] = None) -> None:
        """
        Send a request to the deployed serving endpoint and collect metrics.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the request.

        Returns
        -------
        response : Union[Dict, None]
            The JSON response from the server or None if an error occurs.
        """
        if "messages" not in params:
            prompt_tokens = 128
            if "prompt_tokens" in params:
                prompt_tokens = params["prompt_tokens"]
            else:
                logger.warning("A random prompt with %d tokens will be generated.", prompt_tokens)

            prompt = self.prompt_generator.generate_prompt(prompt_tokens)
            params["messages"] = [{"role": "system", "content": prompt}]
        else:
            prompt = params["messages"][0]["content"]
        chat_params = self._get_chat_completion_params(params)
        if "stream" not in chat_params:
            chat_params["stream"] = self.stream
        if "timeout" not in chat_params:
            chat_params["timeout"] = self.timeout

        total_request_time = 0
        generated_text = ""
        ttft = 0
        start_time = time.monotonic()
        response = await self.client.chat.completions.create(**chat_params)

        if chat_params["stream"]:
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    if not ttft:
                        ttft = time.monotonic() - start_time  # type: ignore
                    generated_text += chunk.choices[0].delta.content
        else:
            generated_text = response.choices[0].message.content

        total_request_time = time.monotonic() - start_time  # type: ignore
        raw_metric = RequestRecords(
            input=prompt,
            output=generated_text,
            end_to_end_latency=total_request_time,
            ttft=ttft,
        )
        self.metrics.append(raw_metric)

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

    def get_metrics(self) -> List[RequestRecords]:
        """
        Retrieve the collected metrics.

        Returns
        -------
        metrics : List[RequestRecords]
            The list of collected metrics.
        """
        return self.metrics
