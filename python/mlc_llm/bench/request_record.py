"""MLC LLM Bench Request"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd  # pylint: disable=import-error
from pydantic import BaseModel

from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.support import logging

logger = logging.getLogger(__name__)


class ServerMetrics(BaseModel):
    """The metrics from the server side."""

    input_tokens: int
    prefill_tokens: int
    output_tokens: int
    end_to_end_latency_s: float
    prefill_tokens_per_s: float
    inter_token_latency_s: float
    time_per_output_token_s: float
    time_to_first_token_s: Optional[float] = None


class Metrics(BaseModel):
    """The list of metric keys"""

    success: bool
    start_time: float
    finish_time: float
    end_to_end_latency_s: float

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    inter_token_latency_s: Optional[float] = None
    time_per_output_token_s: Optional[float] = None
    time_to_first_token_s: Optional[float] = None
    server_metrics: Optional[ServerMetrics] = None

    exec_feature: Optional[Dict[str, Any]] = None


class RequestRecord(BaseModel):
    """The request records collected from LLM inference requests."""

    request_id: Optional[int] = None
    chat_cmpl: ChatCompletionRequest
    output_str: Optional[str] = None
    first_chunk_output_str: str = ""
    timestamp: Optional[float] = None
    metrics: Optional[Metrics] = None
    error_msg: Optional[str] = None


class GroupedRequestRecord(RequestRecord):
    """The data structure for request record groups.
    For datasets that have common prefix sharing, the request records
    that share a same common prefix will be wrapped in a GroupedRequestRecord
    at the beginning.
    """

    records: List[RequestRecord]


def generate_metrics_summary(
    request_records: List[RequestRecord],
    num_total_requests: int,
    num_gpus: int,
) -> Dict[str, Any]:
    """Computes summary statistics across all metrics collected.
    Return a dictionary as the report.
    """
    num_completed_requests = len(request_records)
    assert num_completed_requests <= num_total_requests
    request_metrics = [record.metrics for record in request_records]
    duration = (
        max(metrics.finish_time for metrics in request_metrics)
        - min(metrics.start_time for metrics in request_metrics)
        if num_completed_requests > 0
        else 1e-5
    )

    report = _compute_metrics_statistics(request_metrics)
    report["num_gpus"] = num_gpus
    report["duration"] = duration
    report["num_total_requests"] = num_total_requests
    report["num_completed_requests"] = num_completed_requests
    report["request_throughput"] = num_completed_requests / duration

    total_input_tokens = sum(metric.input_tokens for metric in request_metrics)
    total_output_tokens = sum(metric.output_tokens for metric in request_metrics)
    report["total_input_tokens"] = total_input_tokens
    report["total_output_tokens"] = total_output_tokens
    report["input_token_throughput"] = total_input_tokens / duration
    report["input_token_throughput_per_gpu"] = report["input_token_throughput"] / num_gpus
    report["output_token_throughput"] = total_output_tokens / duration
    report["output_token_throughput_per_gpu"] = report["output_token_throughput"] / num_gpus

    # Generate the server metrics statistics
    server_metrics = [metric.server_metrics for metric in request_metrics if metric.server_metrics]
    server_report = _compute_metrics_statistics(server_metrics)
    if server_report is not None and len(server_report) > 0:
        report["server_metrics"] = server_report

    report = {
        "exec_feature": (
            request_records[0].metrics.exec_feature if num_completed_requests > 0 else None
        ),
        **report,
    }
    return report


def _compute_metrics_statistics(metrics: List[Union[Metrics, ServerMetrics]]) -> Dict[str, Any]:
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
    if not metrics:
        return {}

    report: Dict = {}
    df = pd.DataFrame([metric.model_dump() for metric in metrics])
    for key, _ in metrics[0].model_fields.items():
        if key in ["success", "start_time", "finish_time", "server_metrics", "exec_feature"]:
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


def convert_reports_to_df(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert benchmark reports to pandas DataFrame."""

    def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        items: List[Tuple[str, Any]] = []
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key != "" else key
            if isinstance(value, dict):
                items.extend(_flatten_dict(value, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)

    return pd.DataFrame([_flatten_dict(report) for report in reports])


def pretty_print_report(report: Dict[str, Any]) -> None:  # pylint: disable=too-many-statements
    """Pretty print the metrics report."""

    def _print(report: Dict[str, Any], server_metrics: bool):  # pylint: disable=too-many-statements
        # pylint: disable=line-too-long
        # fmt: off
        title = "Benchmark Result"
        if server_metrics:
            title += " (server side)"
        print(f" {title} ".center(50, "="))
        if not server_metrics:
            print(f"{'Total requests:':<40} {report['num_total_requests']:<10}")
            print(f"{'Completed requests:':<40} {report['num_completed_requests']:<10}")
            print(f"{'Duration (s):':<40} {report['duration']:<10.2f}")
            print(f"{'Num GPUs:':<40} {report['num_gpus']:<10}")
            print(f"{'Total input tokens:':<40} {report['total_input_tokens']:<10}")
            print(f"{'Total output tokens:':<40} {report['total_output_tokens']:<10}")
            print(f"{'Request throughput (req/s):':<40} {report['request_throughput']:<10.2f}")
            print(f"{'Input token throughput (tok/s):':<40} {report['input_token_throughput']:<10.2f}")
            print(f"{'Input token throughput per GPU (tok/s):':<40} {report['input_token_throughput_per_gpu']:<10.2f}")
            print(f"{'Output token throughput (tok/s):':<40} {report['output_token_throughput']:<10.2f}")
            print(f"{'Output token throughput per GPU (tok/s):':<40} {report['output_token_throughput_per_gpu']:<10.2f}")

        if report["num_completed_requests"] == 0:
            return
        ttft = report["time_to_first_token_s"]
        print(" Time to First Token (TTFT, ms) ".center(50, "-"))
        print(f"{'Mean:':<40} {ttft['mean'] * 1000:<10.2f}")
        print(f"{'Stddev:':<40} {ttft['stddev'] * 1000:<10.2f}")
        print(f"{'P25:':<40} {ttft['quantiles']['p25'] * 1000:<10.2f}")
        print(f"{'P50:':<40} {ttft['quantiles']['p50'] * 1000:<10.2f}")
        print(f"{'P75:':<40} {ttft['quantiles']['p75'] * 1000:<10.2f}")
        print(f"{'P90:':<40} {ttft['quantiles']['p90'] * 1000:<10.2f}")
        print(f"{'P95:':<40} {ttft['quantiles']['p95'] * 1000:<10.2f}")
        print(f"{'P99:':<40} {ttft['quantiles']['p99'] * 1000:<10.2f}")
        print(f"{'Min:':<40} {ttft['min'] * 1000:<10.2f}")
        print(f"{'Max:':<40} {ttft['max'] * 1000:<10.2f}")

        tpot = report["time_per_output_token_s"]
        print(" Time per Output Token (TPOT, ms) ".center(50, "-"))
        print(f"{'Mean:':<40} {tpot['mean'] * 1000:<10.2f}")
        print(f"{'Stddev:':<40} {tpot['stddev'] * 1000:<10.2f}")
        print(f"{'P25:':<40} {tpot['quantiles']['p25'] * 1000:<10.2f}")
        print(f"{'P50:':<40} {tpot['quantiles']['p50'] * 1000:<10.2f}")
        print(f"{'P75:':<40} {tpot['quantiles']['p75'] * 1000:<10.2f}")
        print(f"{'P90:':<40} {tpot['quantiles']['p90'] * 1000:<10.2f}")
        print(f"{'P95:':<40} {tpot['quantiles']['p95'] * 1000:<10.2f}")
        print(f"{'P99:':<40} {tpot['quantiles']['p99'] * 1000:<10.2f}")
        print(f"{'Min:':<40} {tpot['min'] * 1000:<10.2f}")
        print(f"{'Max:':<40} {tpot['max'] * 1000:<10.2f}")

        itl = report["inter_token_latency_s"]
        print(" Inter-Token Latency (ms) ".center(50, "-"))
        print(f"{'Mean:':<40} {itl['mean'] * 1000:<10.2f}")
        print(f"{'Stddev:':<40} {itl['stddev'] * 1000:<10.2f}")
        print(f"{'P25:':<40} {itl['quantiles']['p25'] * 1000:<10.2f}")
        print(f"{'P50:':<40} {itl['quantiles']['p50'] * 1000:<10.2f}")
        print(f"{'P75:':<40} {itl['quantiles']['p75'] * 1000:<10.2f}")
        print(f"{'P90:':<40} {itl['quantiles']['p90'] * 1000:<10.2f}")
        print(f"{'P95:':<40} {itl['quantiles']['p95'] * 1000:<10.2f}")
        print(f"{'P99:':<40} {itl['quantiles']['p99'] * 1000:<10.2f}")
        print(f"{'Min:':<40} {itl['min'] * 1000:<10.2f}")
        print(f"{'Max:':<40} {itl['max'] * 1000:<10.2f}")

        e2e_latency = report["end_to_end_latency_s"]
        print(" End-to-End Latency (ms) ".center(50, "-"))
        print(f"{'Mean:':<40} {e2e_latency['mean'] * 1000:<10.2f}")
        print(f"{'Stddev:':<40} {e2e_latency['stddev'] * 1000:<10.2f}")
        print(f"{'P25:':<40} {e2e_latency['quantiles']['p25'] * 1000:<10.2f}")
        print(f"{'P50:':<40} {e2e_latency['quantiles']['p50'] * 1000:<10.2f}")
        print(f"{'P75:':<40} {e2e_latency['quantiles']['p75'] * 1000:<10.2f}")
        print(f"{'P90:':<40} {e2e_latency['quantiles']['p90'] * 1000:<10.2f}")
        print(f"{'P95:':<40} {e2e_latency['quantiles']['p95'] * 1000:<10.2f}")
        print(f"{'P99:':<40} {e2e_latency['quantiles']['p99'] * 1000:<10.2f}")
        print(f"{'Min:':<40} {e2e_latency['min'] * 1000:<10.2f}")
        print(f"{'Max:':<40} {e2e_latency['max'] * 1000:<10.2f}")

        input_tokens = report["input_tokens"]
        print(" Input Tokens ".center(50, "-"))
        print(f"{'Mean:':<40} {input_tokens['mean']:<1}")
        print(f"{'Stddev:':<40} {input_tokens['stddev']:<1}")
        print(f"{'P25:':<40} {input_tokens['quantiles']['p25']:<1}")
        print(f"{'P50:':<40} {input_tokens['quantiles']['p50']:<1}")
        print(f"{'P95:':<40} {input_tokens['quantiles']['p95']:<1}")
        print(f"{'Min:':<40} {input_tokens['min']:<1}")
        print(f"{'Max:':<40} {input_tokens['max']:<1}")

        output_tokens = report["output_tokens"]
        print(" Output Tokens ".center(50, "-"))
        print(f"{'Mean:':<40} {output_tokens['mean']:<1}")
        print(f"{'Stddev:':<40} {output_tokens['stddev']:<1}")
        print(f"{'P25:':<40} {output_tokens['quantiles']['p25']:<1}")
        print(f"{'P50:':<40} {output_tokens['quantiles']['p50']:<1}")
        print(f"{'P95:':<40} {output_tokens['quantiles']['p95']:<1}")
        print(f"{'Min:':<40} {output_tokens['min']:<1}")
        print(f"{'Max:':<40} {output_tokens['max']:<1}")

        print("=" * 50)

    # fmt: on
    # pylint: enable=line-too-long
    _print(report, server_metrics=False)
    if "server_metrics" in report:
        _print(report["server_metrics"], server_metrics=True)
