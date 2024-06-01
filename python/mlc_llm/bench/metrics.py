""" MLC LLM bench Metrics Collector"""
import json
from typing import Any, Dict, List, Union

import pandas as pd
from transformers import LlamaTokenizerFast

from mlc_llm.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)

# The list of metric keys, overall_output_throughput and num_completed_requests
# are able to be computed based on the other metrics
METRIC_NAMES = [
    "inter_token_latency",
    "decode_token_latency",
    "ttft",
    "end_to_end_latency",
    "num_input_tokens",
    "num_output_tokens",
]


def get_token_length(text):
    """Get the number of tokens."""
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    return len(tokenizer.encode(text))


class MetricsCollector:
    """
    A class to manage various performance metrics.

    Attributes
    ----------
    metric_keys : List[str]
        A list of all the metric keys managed by the collector.
    all_metrics : List[Dict[str, Any]]
        A list containing all metrics data recorded, each as a dictionary.
    """

    def __init__(self) -> None:
        """
        Initializes the metrics collector.
        """
        self.metric_keys: List = list(METRIC_NAMES)
        self.all_metrics: List[Dict] = []

    def add_metrics(self, metrics: Dict[str, Union[int, float]]) -> None:
        """
        Adds a new set of metric data to the collection.

        Parameters
        ----------
        metrics : Dict[str, Union[int, float]]
            A dictionary containing values for the metrics defined in metric_keys.
        """
        if not all(key in self.metric_keys for key in metrics.keys()):
            logger.error("Metric dictionary keys do not match the predefined metric keys.")
        self.all_metrics.append(metrics)

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Returns all the metric data collected.

        Returns
        -------
        List[Dict[str, Any]]
            A list of all metric data dictionaries.
        """
        return self.all_metrics

    def get_metrics_summary(self, start_time: int, end_time: int) -> Dict[str, Any]:
        """
        Computes summary statistics across all metrics collected.

        Returns
        -------
        ret : Dict[str, Any]
            A dictionary containing the summary statistics of the collected metrics.

        start_time : int
            The start time of the metrics collection.

        end_time : int
            The end time of the metrics collection.
        """
        if not self.all_metrics:
            return None

        ret: Dict[str, Any] = {}
        metrics = self.all_metrics

        df = pd.DataFrame(metrics)

        for key in self.metric_keys:
            ret[key] = {}
            series = df[key].dropna()
            quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f"p{int(quantile * 100)}"
                quantiles_reformatted_keys[reformatted_key] = value
            ret[key]["quantiles"] = quantiles_reformatted_keys
            mean = series.mean()
            ret[key]["mean"] = mean
            ret[key]["min"] = series.min()
            ret[key]["max"] = series.max()
            ret[key]["stddev"] = series.std()

        ret["num_completed_requests"] = len(metrics)
        ret["overall_output_throughput"] = df["num_output_tokens"].sum() / (end_time - start_time)
        logger.info("Metrics Summary:\n%s", json.dumps(ret, indent=4, default=str))
        return ret
