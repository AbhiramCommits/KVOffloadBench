"""Prometheus exporter for replay runs.

Histograms: TTFT and inter-token latency. Counters: KV bytes per tier and
operation. Gauge: in-flight requests and final goodput ratio.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

log = logging.getLogger(__name__)


class ReplayMetrics:
    def __init__(self, port: int = 0, run_id: str = "") -> None:
        self.registry = CollectorRegistry()
        self.run_id = run_id
        self.ttft = Histogram(
            "kvob_ttft_seconds",
            "Time to first token",
            buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 60),
            registry=self.registry,
        )
        self.itl = Histogram(
            "kvob_itl_seconds",
            "Inter-token latency",
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.e2e = Histogram(
            "kvob_e2e_seconds",
            "End-to-end request latency",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 60, 120),
            registry=self.registry,
        )
        self.kv_bytes = Counter(
            "kvob_kv_bytes_total",
            "KV bytes read or written per tier",
            ["tier", "op"],
            registry=self.registry,
        )
        self.inflight = Gauge(
            "kvob_inflight_requests",
            "Requests currently being served",
            registry=self.registry,
        )
        self.goodput_ratio = Gauge(
            "kvob_goodput_ratio",
            "Fraction of attempted requests meeting the SLO",
            registry=self.registry,
        )
        self.requests_total = Counter(
            "kvob_requests_total",
            "Completed requests by status",
            ["status"],
            registry=self.registry,
        )
        self._prev_tier: dict[str, tuple[int, int]] = {}
        self._port = port
        self._server_started = False

    def start(self) -> None:
        if self._port and not self._server_started:
            start_http_server(self._port, registry=self.registry)
            self._server_started = True
            log.info("Prometheus metrics on :%d", self._port)

    def request_finished(
        self, ttft_ms: float | None, e2e_ms: float | None, itl_ms: list[float], status: str
    ) -> None:
        self.requests_total.labels(status=status).inc()
        if ttft_ms is not None:
            self.ttft.observe(ttft_ms / 1000.0)
        if e2e_ms is not None:
            self.e2e.observe(e2e_ms / 1000.0)
        for value in itl_ms:
            self.itl.observe(value / 1000.0)

    def update_store(self, store_stats: dict[str, Any] | None) -> None:
        if not store_stats:
            return
        for tier in store_stats.get("tiers", []):
            name = str(tier.get("tier", "?"))
            read = int(tier.get("bytes_read", 0))
            written = int(tier.get("bytes_written", 0))
            prev = self._prev_tier.get(name, (0, 0))
            if read > prev[0]:
                self.kv_bytes.labels(tier=name, op="read").inc(read - prev[0])
            if written > prev[1]:
                self.kv_bytes.labels(tier=name, op="write").inc(written - prev[1])
            self._prev_tier[name] = (read, written)


_metrics: ReplayMetrics | None = None


def get_metrics(port: int = 0, run_id: str = "") -> ReplayMetrics:
    global _metrics
    if _metrics is None:
        _metrics = ReplayMetrics(port=port, run_id=run_id)
    return _metrics
