"""Tests for engine metrics text serialization without model inference."""

import asyncio
import copy
from unittest.mock import AsyncMock, patch

import pytest

from mlc_llm.serve.engine_base import EngineMetrics
from mlc_llm.serve.entrypoints import metrics_entrypoints


@pytest.mark.parametrize(
    "scope, key, value, expected",
    [
        (
            "decode_time_by_batch_size",
            "mean{batch_size=1}",
            0.01,
            'decode_time_by_batch_size_mean{batch_size="1"}',
        ),
        (
            "decode_time_by_batch_size",
            "count{batch_size=0}",
            3,
            'decode_time_by_batch_size_count{batch_size="0"}',
        ),
        (
            "spec_decode_accept_prob",
            "accept_prob{step=0}",
            0.5,
            'spec_decode_accept_prob_accept_prob{step="0"}',
        ),
        (
            "spec_decode_accept_len",
            "accept_len{step=12}",
            1.5,
            'spec_decode_accept_len_accept_len{step="12"}',
        ),
        (
            "spec_decode_accept_rate",
            "accept_rate{step=1}",
            0.25,
            'spec_decode_accept_rate_accept_rate{step="1"}',
        ),
        (
            "decode_time_by_batch_size",
            'mean{batch_size="1"}',
            0.01,
            'decode_time_by_batch_size_mean{batch_size="1"}',
        ),
    ],
)
def test_prometheus_labels_quoted_without_changing_json(scope, key, value, expected):
    raw = {"prefill_tokens": 10, scope: {key: value}}
    original = copy.deepcopy(raw)
    metrics = EngineMetrics(raw)
    text = metrics.prometheus_text()
    assert f"{expected}\t{value}\n" in text
    assert "prefill_tokens\t10\n" in text
    assert metrics.metrics == original
    assert metrics[scope] == original[scope]


def test_metrics_endpoint_is_parseable():
    parser = pytest.importorskip("prometheus_client.parser")
    engine = AsyncMock()
    engine.metrics.return_value = EngineMetrics(
        {
            "decode_time_by_batch_size": {"mean{batch_size=1}": 0.01, "count{batch_size=1}": 2},
            "spec_decode": {"accept_prob": {"accept_prob{step=0}": 0.5}},
        }
    )
    with patch("mlc_llm.serve.entrypoints.metrics_entrypoints.ServerContext.current") as current:
        current.return_value.get_model_list.return_value = ["model"]
        current.return_value.get_engine.return_value = engine
        text = asyncio.run(metrics_entrypoints.metrics(None))
    samples = {
        s.name: s for family in parser.text_string_to_metric_families(text) for s in family.samples
    }
    assert samples["decode_time_by_batch_size_mean"].labels == {"batch_size": "1"}
    assert samples["decode_time_by_batch_size_mean"].value == 0.01
    assert samples["spec_decode_accept_prob_accept_prob"].labels == {"step": "0"}
    assert samples["spec_decode_accept_prob_accept_prob"].value == 0.5
    assert text.endswith("\n")


def test_empty_metrics_has_final_line_feed():
    assert EngineMetrics({}).prometheus_text().endswith("\n")
