"""MLC LLM bench request"""
import time
from typing import Dict, Optional

import httpx
from openai import AsyncOpenAI
from typing_extensions import Self

from mlc_llm.support import logging

from .metrics import MetricsCollector, get_token_length

logging.enable_logging()
logger = logging.getLogger(__name__)

OPENAI_COMPLETION_PARAMS = [
    "messages",
    "model",
    "stream",
    "loglikelihood",
    "seed",
    "max_tokens",
    "presence_penalty",
    "echo",
    "n",
    "top_p",
    "frequency_penalty",
    "best_of",
    "stop",
    "stream",
    "temperature",
    "model",
    "force_asset_download",
]


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
        self.stream = stream
        self.timeout = timeout
        self.client = AsyncOpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="None",
            http_client=httpx.AsyncClient(http2=True),
        )
        self.metrics_collector = MetricsCollector()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.client.close()

    def get_stats(self, start_time: float, end_time: float) -> Dict:
        """
        Returns the statistics summary.

        Parameters
        ----------
        start_time : float
            The start time for the statistics collection.

        end_time : float
            The end time for the statistics collection.

        Returns
        -------
        stats : Dict
            The performance statistics.
        """
        return self.metrics_collector.get_metrics_summary(start_time, end_time)

    async def __call__(self, params: Dict = None) -> None:
        """
        Send request to the deployed serving endpoint.

        Parameters
        ----------
        params : Dict
            The parameters for the request.

        Returns
        -------
        response : Union[Dict, None]
            The JSON response from the server or None if an error occurs.
        """

        def _get_chat_completion_params(params):
            chat_params = {}
            for key in OPENAI_COMPLETION_PARAMS:
                if key in params:
                    chat_params[key] = params[key]
            return chat_params

        chat_params = _get_chat_completion_params(params)
        # Use the stream specified in constructor
        if self.stream:
            chat_params["stream"] = self.stream
        if self.timeout:
            chat_params["timeout"] = self.timeout

        metrics = {}
        time_to_next_token = []
        total_request_time = 0
        tokens_received = 0
        generated_text = ""
        ttft = None
        prompt = params["messages"][0]["content"]
        metrics["num_input_tokens"] = get_token_length(prompt)
        most_recent_received_token_time = start_time = time.monotonic()

        response = await self.client.chat.completions.create(**chat_params)

        # Don't collect stats for non-streaming requests
        if chat_params["stream"] is False:
            return None

        async for chunk in response:
            tokens_received += 1
            if chunk.choices[0].delta.content is not None:
                if not ttft:
                    ttft = time.monotonic() - start_time
                    time_to_next_token.append(ttft)
                else:
                    time_to_next_token.append(time.monotonic() - most_recent_received_token_time)
                most_recent_received_token_time = time.monotonic()
                generated_text += chunk.choices[0].delta.content

        total_request_time = time.monotonic() - start_time  # type: ignore
        # assert num_output_tokens == get_token_length(generated_text)
        metrics["num_output_tokens"] = tokens_received
        metrics["ttft"] = ttft
        metrics["end_to_end_latency"] = total_request_time
        metrics["inter_token_latency"] = sum(time_to_next_token)
        metrics["decode_token_latency"] = metrics["inter_token_latency"] - ttft
        self.metrics_collector.add_metrics(metrics)
