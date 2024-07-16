""" MLC LLM bench Metrics"""

import json
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from mlc_llm.support import logging

from .request import RequestRecords

logging.enable_logging()
logger = logging.getLogger(__name__)


class ServerMetrics(BaseModel):
    """The metrics from the server side."""

    prompt_tokens: int
    prefill_tokens: int
    completion_tokens: int
    decode_tokens_per_s: float
    prefill_tokens_per_s: float
    end_to_end_latency_s: float
    inter_token_latency_s: float
    ttft_s: Optional[float] = None


class Metrics(BaseModel):
    """The list of metric keys"""

    prompt_tokens: int
    completion_tokens: int
    end_to_end_latency_s: float
    inter_token_latency_s: float
    decode_tokens_per_s: float
    ttft: Optional[float] = None
    server_metrics: Optional[ServerMetrics] = None


class MetricsProcessor:
    """The metrics processor class

    Parameters
    ----------
    tokenizer : Optional[Tokenizer]
        The tokenizer.

    request_records : List[RequestRecords]
        The list of request records.
    """

    def __init__(self, request_records: List[RequestRecords], tokenizer=None) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            from transformers import (  # pylint: disable=import-outside-toplevel,import-error
                LlamaTokenizerFast,
            )

            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                "hf-internal-testing/llama-tokenizer"
            )
            logger.warning("No tokenizer provided. Using default tokenizer.")
        self.all_metrics: List[Metrics] = self.extract_metrics_from_request_records(request_records)

    def count_tokens(self, prompt: str) -> int:
        """Count the number of tokens in the text

        Parameters
        ----------
        prompt : str
            The text to count the tokens.

        Returns
        -------
        prompt_tokens : int
            The number of tokens in the prompt.
        """
        return len(self.tokenizer.encode(prompt))

    def extract_metrics_from_request_records(
        self, request_records: List[RequestRecords]
    ) -> List[Metrics]:
        """
        Extract the metrics from request records.

        Parameters
        ----------
        request_records : List[RequestRecords]
            The list of raw request records collected.

        Returns
        -------
        metrics : List[Metrics]
            The list of extracted metrics with additional items.
        """

        result = []
        for metric in request_records:
            prompt_tokens = self.count_tokens(metric.input)
            completion_tokens = self.count_tokens(metric.output)
            assert prompt_tokens > 0 and completion_tokens >= 0, "Invalid prompt tokens"
            end_to_end_latency_s = metric.end_to_end_latency_s
            ttft = metric.ttft if metric.ttft is not None else 0
            server_metric = None
            if metric.server_metrics is not None:
                server_metric = ServerMetrics(
                    prompt_tokens=metric.server_metrics["prompt_tokens"],
                    prefill_tokens=metric.server_metrics["prefill_tokens"],
                    completion_tokens=metric.server_metrics["completion_tokens"],
                    decode_tokens_per_s=metric.server_metrics["decode_tokens_per_s"],
                    prefill_tokens_per_s=metric.server_metrics["prefill_tokens_per_s"],
                    end_to_end_latency_s=metric.server_metrics["end_to_end_latency_s"],
                    inter_token_latency_s=metric.server_metrics["inter_token_latency_s"],
                    ttft_s=metric.server_metrics["ttft_s"],
                )
            refined_metric = Metrics(
                inter_token_latency_s=end_to_end_latency_s / completion_tokens,
                decode_tokens_per_s=(completion_tokens - 1) / (end_to_end_latency_s - ttft),
                ttft=metric.ttft,
                end_to_end_latency_s=end_to_end_latency_s,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                server_metrics=server_metric,
            )
            result.append(refined_metric)
        return result

    def get_metrics(self) -> List[Metrics]:
        """
        Get the metrics collected.

        Returns
        -------
        all_metrics : List[Metrics]
            The list of metrics collected.
        """
        return self.all_metrics

    def reset_metrics(self, metrics: List[Metrics]) -> None:
        """Reset the metrics collected.

        Parameters
        ----------
        metrics : List[Metrics]
            The list of metrics to reset.
        """
        self.all_metrics = metrics

    def filter_metrics(self, criteria: Optional[Callable[[Metrics], bool]] = None) -> List[Metrics]:
        """
        Filters the metrics based on the provided criteria. If no criteria are provided,
        it filters out metrics with any fields set to None or 0.

        Parameters
        ----------
        criteria : Optional[Callable[[Metrics], bool]]
            A function that takes a metric as input,
            returns True if the metric should be included.

        Returns
        -------
        filtered_metrics : List[Metrics]
            The list of metrics that meet the specified criteria.
        """
        if criteria is None:
            # Default criteria to filter out metrics with None or 0 in certain fields
            def criteria(metric: Metrics) -> bool:
                for field, _ in Metrics.model_fields.items():
                    val = getattr(metric, field)
                    if val is None or val == 0:
                        return False
                return True

        filered_metrics = [metric for metric in self.all_metrics if criteria(metric)]
        self.reset_metrics(filered_metrics)
        return filered_metrics

    def generate_metrics_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Computes summary statistics across all metrics collected.

        Parameters
        ----------
        all_metrics : List[RequestRecords]
            All the metrics data collected in the monitoring period.

        start_time : float
            The start time of the monitoring period.

        end_time : float
            The end time of the monitoring period.

        Returns
        -------
        report : Dict
            A dictionary containing the summary statistics of the collected metrics.
        """
        if not self.all_metrics:
            return {}

        # Generate the client metrics statistics
        report = self._compute_metrics_statistics(self.all_metrics)
        report["num_completed_requests"] = len(self.all_metrics)
        total_tokens = sum(metric.completion_tokens for metric in self.all_metrics)
        report["overall_output_throughput"] = total_tokens / (end_time - start_time)

        # Generate the server metrics statistics
        server_metrics = [
            metric.server_metrics for metric in self.all_metrics if metric.server_metrics
        ]
        server_report = self._compute_metrics_statistics(server_metrics)
        report["server_metrics"] = server_report

        logger.info("Metrics Summary:\n%s", json.dumps(report, indent=4, default=str))
        return report

    def _compute_metrics_statistics(self, metrics: List[Union[Metrics, ServerMetrics]]) -> Dict:
        """
        Compute the statistics of the metrics.

        Parameters
        ----------
        metrics : List[Union[Metrics, ServerMetrics]]
            The list of metrics to get the statistics.

        Returns
        -------
        report : Dict
            The statistics of the metrics.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel,import-error

        report: Dict = {}
        if not metrics:
            return report

        df = pd.DataFrame([metric.model_dump() for metric in metrics])
        for key, _ in metrics[0].model_fields.items():
            if key == "server_metrics":
                continue
            if key in df.columns:
                series = df[key].dropna()
                report[key] = {
                    "quantiles": {
                        f"p{int(q * 100)}": v
                        for q, v in series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).items()
                    },
                    "mean": series.mean(),
                    "min": series.min(),
                    "max": series.max(),
                    "stddev": series.std(),
                }
        return report
