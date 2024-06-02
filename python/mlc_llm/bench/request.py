"""MLC LLM bench request"""
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


class RawMetrics(BaseModel):
    """The raw metrics collected from the request."""

    input: str
    output: str
    end_to_end_latency: float
    ttft: Optional[float] = 0


class OpenAIRequestSender:
    """
    Collect inference statistics.

    Parameters
    ----------
    host : Optional[str]
        The host address for the API, by default "127.0.0.1".

    port : Optional[int]
        The port number for the API, by default 8008.

    stream : Optional[bool]
        Indicates whether streaming should be enabled. Default is True.

    timeout : Optional[float]
        The timeout in seconds for each request, by default 180.

    Attributes
    ----------
    stats : dict
        A dictionary to store statistics about requests and responses.
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
        self.metrics: List[RawMetrics] = []
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
        Send request to the deployed serving endpoint.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the request.

        Returns
        -------
        response : Union[Dict, None]
            The JSON response from the server or None if an error occurs.
        """
        # Generate prompts if not provided
        if "messages" not in params:
            num_tokens = 128
            if "prompt_tokens" in params:
                num_tokens = params["prompt_tokens"]
            else:
                logger.warning("Neither messages nor prompt tokens provided.")
            prompt = self.prompt_generator.generate_prompt(num_tokens)
            params["messages"] = [{"role": "system", "content": prompt}]
        else:
            prompt = params["messages"][0]["content"]
        chat_params = self._get_chat_completion_params(params)
        # Use the default parameters if not provided
        if "stream" not in chat_params:
            chat_params["stream"] = self.stream
        if "timeout" not in chat_params:
            chat_params["timeout"] = self.timeout

        total_request_time = 0
        generated_text = ""
        ttft = 0
        start_time = time.monotonic()
        # TODO(yongwww): handle Completion request
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
        raw_metric = RawMetrics(
            input=prompt,
            output=generated_text,
            end_to_end_latency=total_request_time,
            ttft=ttft,
        )
        self.metrics.append(raw_metric)

    def _get_chat_completion_params(self, params: Dict) -> Dict:
        """
        Get the chat completion parameters from the request parameters.

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

    def get_metrics(self) -> List[RawMetrics]:
        """
        Get the metrics collected.

        Returns
        -------
        metrics : List[RawMetrics]
            The metrics collected.
        """
        return self.metrics
