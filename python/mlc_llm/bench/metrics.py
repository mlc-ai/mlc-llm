""" MLC LLM bench Metrics"""
import json
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from mlc_llm.support import logging

from .request import RequestRecords

logging.enable_logging()
logger = logging.getLogger(__name__)


class Metrics(BaseModel):
    """The list of metric keys"""

    prompt_tokens: int
    completion_tokens: int
    end_to_end_latency: float
    inter_token_latency: float
    decode_token_latency: float
    ttft: Optional[float] = None


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
            end_to_end_latency = metric.end_to_end_latency
            if metric.ttft is None:
                ttft = 0
            refined_metric = Metrics(
                inter_token_latency=end_to_end_latency / completion_tokens,
                decode_token_latency=(end_to_end_latency - ttft) / completion_tokens,
                ttft=metric.ttft,
                end_to_end_latency=end_to_end_latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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

    def generate_metrics_summary(
        self, start_time: float, end_time: float
    ) -> Dict[str, Union[int, float]]:
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
        import pandas as pd  # pylint: disable=import-outside-toplevel,import-error

        if not self.all_metrics:
            return {}

        metrics = self.all_metrics
        df = pd.DataFrame([metric.model_dump() for metric in metrics])

        report: Dict = {}
        for key, _ in Metrics.model_fields.items():
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

        report["num_completed_requests"] = len(metrics)
        report["overall_output_throughput"] = df["completion_tokens"].sum() / (
            end_time - start_time
        )

        logger.info("Metrics Summary:\n%s", json.dumps(report, indent=4, default=str))
        return report
