""" MLC LLM bench Metrics"""
import json
from typing import Dict, List, Union

from pydantic import BaseModel

from mlc_llm.support import logging

from .request import RawMetrics

logging.enable_logging()
logger = logging.getLogger(__name__)


class RefinedMetrics(BaseModel):
    """The list of metric keys"""

    ttft: float
    end_to_end_latency: float
    inter_token_latency: float
    decode_token_latency: float
    num_input_tokens: int
    num_output_tokens: int


def refine_metrics(metrics: List[RawMetrics]) -> List[RefinedMetrics]:
    """
    Augments raw metrics.

    Parameters
    ----------
    metrics : List[RefinedMetrics]
        The list of raw metrics data collected.

    Returns
    -------
    metrics : List[RefinedMetrics]
        The list of augmented metrics with additional items.
    """
    from transformers import (  # pylint: disable=import-outside-toplevel,import-error
        LlamaTokenizerFast,
    )

    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

    def _count_tokens(text: str, tokenizer) -> int:
        return len(tokenizer.encode(text))

    result = []
    for metric in metrics:
        num_input_tokens = _count_tokens(metric.input, tokenizer)
        num_output_tokens = _count_tokens(metric.output, tokenizer)
        assert num_input_tokens > 0 and num_output_tokens >= 0, "Invalid number of tokens"
        end_to_end_latency = metric.end_to_end_latency
        # TODO(yongwww): handle the non-streaming case where ttft is 0
        refined_metric = RefinedMetrics(
            inter_token_latency=end_to_end_latency / num_output_tokens,
            decode_token_latency=(end_to_end_latency - metric.ttft) / num_output_tokens,
            ttft=metric.ttft,
            end_to_end_latency=end_to_end_latency,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
        )
        result.append(refined_metric)
    return result


def get_metrics_summary(
    all_metrics: List[RawMetrics], start_time: float, end_time: float
) -> Dict[str, Union[int, float]]:
    """
    Computes summary statistics across all metrics collected.

    Parameters
    ----------
    all_metrics : List[RawMetrics]
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

    if not all_metrics:
        logger.warning("No metrics collected.")
        return {}

    metrics = refine_metrics(all_metrics)
    df = pd.DataFrame([metric.model_dump() for metric in metrics])

    report: Dict = {}
    for key, _ in RefinedMetrics.model_fields.items():
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
    report["overall_output_throughput"] = df["num_output_tokens"].sum() / (end_time - start_time)

    logger.info("Metrics Summary:\n%s", json.dumps(report, indent=4, default=str))
    return report
