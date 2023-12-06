from .metrics_labels import *
from prometheus_client import Counter, Histogram, Gauge


class PrometheusMetrics:
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}

        for label in [NUM_CACHE_EVICTONS]:
            self.counters[label] = Counter(label, label)

        buckets_e2e_lat = (0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5)
        buckets_ttft = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0)
        buckets_batched_prefill_tokens = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000)
        buckets_batched_decode_tokens = (1, 10, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300)

        for label, buckets in [
            (E2E_LATENCY, buckets_e2e_lat),
            (FIRST_TOKEN_LATENCY, buckets_ttft),
            (BATCHED_PREFILL_TOKENS, buckets_batched_prefill_tokens),
            (BATCHED_DECODE_TOKENS, buckets_batched_decode_tokens),
        ]:
            self.histograms[label] = Histogram(label, label, buckets=buckets)

        for label in [KV_CACHE_UTILIZATION]:
            self.gauges[label] = Gauge(label, label)

    def _lookup(self, metrics_dict, label):
        if label in metrics_dict:
            return metrics_dict[label]

        return RuntimeError(f"No metric {label} found.")

    def counter(self, label: str):
        return self._lookup(self.counters, label)

    def histogram(self, label: str):
        return self._lookup(self.histograms, label)

    def gauge(self, label: str):
        return self._lookup(self.gauges, label)
