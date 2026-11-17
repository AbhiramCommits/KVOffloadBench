"""Latency sampling and summary statistics shared by tiers, engines, and analysis."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100].

    Returns ``0.0`` for an empty input so downstream tables never contain NaN.
    """
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[int(pos)])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


@dataclass
class LatencyReservoir:
    """Bounded reservoir sample of latencies in nanoseconds.

    The reservoir keeps the first ``capacity`` samples and then replaces random
    members, so ``p99`` stays computable without unbounded memory. A fixed seed
    keeps summaries deterministic across runs of the same workload.
    """

    capacity: int = 200_000
    seed: int = 12345
    _values: list[int] = field(default_factory=list)
    _seen: int = 0
    _rng_state: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._values = []
        self._seen = 0
        self._rng_state = self.seed

    def _rand(self) -> int:
        # xorshift64*, deterministic and dependency-free
        x = self._rng_state or 1
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self._rng_state = x
        return (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF

    def add_ns(self, value_ns: int) -> None:
        if value_ns < 0:
            raise ValueError("latency samples must be non-negative")
        self._seen += 1
        if len(self._values) < self.capacity:
            self._values.append(int(value_ns))
        else:
            idx = self._rand() % self._seen
            if idx < self.capacity:
                self._values[idx] = int(value_ns)

    def extend_ns(self, values: Iterable[int]) -> None:
        for v in values:
            self.add_ns(v)

    @property
    def count(self) -> int:
        return self._seen

    def values_ns(self) -> list[int]:
        return list(self._values)

    def summary_us(self) -> dict[str, float]:
        vals = [v / 1000.0 for v in self._values]
        if not vals:
            return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "count": float(len(vals)),
            "mean": sum(vals) / len(vals),
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "max": max(vals),
        }

    def to_dict(self) -> dict[str, float]:
        return self.summary_us()


def summarize_ns(values_ns: Sequence[int]) -> dict[str, float]:
    us = [v / 1000.0 for v in values_ns]
    return {
        "count": float(len(us)),
        "mean": (sum(us) / len(us)) if us else 0.0,
        "p50": percentile(us, 50),
        "p95": percentile(us, 95),
        "p99": percentile(us, 99),
        "max": max(us) if us else 0.0,
    }


def human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def human_rate(bytes_per_s: float) -> str:
    return f"{human_bytes(bytes_per_s)}/s"


# ---------------------------------------------------------------------------
# Request-level serving metrics
# ---------------------------------------------------------------------------


def request_timings(
    arrival_ns: int, token_timestamps_ns: Sequence[int]
) -> dict[str, float]:
    """TTFT, TPOT, ITL and end-to-end latency for one request.

    TPOT uses the standard ``(e2e - ttft) / (n_out - 1)`` definition; with a
    single output token there is no inter-token gap and TPOT is 0.
    """
    if not token_timestamps_ns:
        raise ValueError("cannot compute timings for a request with no tokens")
    first = token_timestamps_ns[0]
    last = token_timestamps_ns[-1]
    ttft_ms = (first - arrival_ns) / 1e6
    e2e_ms = (last - arrival_ns) / 1e6
    n_out = len(token_timestamps_ns)
    tpot_ms = ((last - first) / 1e6 / (n_out - 1)) if n_out > 1 else 0.0
    itl_ms = [
        (b - a) / 1e6 for a, b in itertools.pairwise(token_timestamps_ns)
    ]
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "itl_ms": itl_ms,
        "output_tokens": float(n_out),
    }


def slo_met(ttft_ms: float, tpot_ms: float, slo_ttft_ms: float, slo_tpot_ms: float) -> bool:
    """Inclusive SLO boundary: exactly at the budget counts as goodput."""
    return ttft_ms <= slo_ttft_ms and tpot_ms <= slo_tpot_ms


def _dist(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def aggregate_requests(
    records: Sequence[dict[str, Any]],
    *,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    window_s: float,
    attempted: int,
    engine_stats: dict[str, Any] | None = None,
    store_stats: dict[str, Any] | None = None,
    warmup_requests: int = 0,
) -> dict[str, Any]:
    """Aggregate per-request rows into the headline serving metrics."""
    measured = [r for r in records if not r.get("warmup", False)]
    completed = [r for r in measured if r.get("status") == "completed"]
    failed = [r for r in measured if r.get("status") != "completed"]
    good = [r for r in completed if r.get("slo_met", r.get("goodput", False))]
    ttft = [float(r["ttft_ms"]) for r in completed]
    tpot = [float(r["tpot_ms"]) for r in completed]
    e2e = [float(r["e2e_ms"]) for r in completed]
    itl: list[float] = []
    for r in completed:
        itl.extend(float(v) for v in r.get("itl_ms", []))
    out_tokens = int(sum(int(r["output_tokens"]) for r in completed))
    goodput_rate = len(good) / window_s if window_s > 0 else 0.0
    counters = (store_stats or {}).get("counters", {}) if store_stats else {}
    restore_bytes = int(counters.get("restore_bytes", 0)) + int(
        counters.get("lookup_restore_bytes", 0)
    )
    service_s = float(counters.get("restore_seconds", 0.0))
    if service_s <= 0 and engine_stats:
        service_s = float(engine_stats.get("storage_io_s", 0.0) or 0.0)
    if service_s <= 0:
        service_s = float(counters.get("lookup_restore_seconds", 0.0))
    offload_bytes = int(counters.get("offload_bytes", 0))
    return {
        "requests_attempted": attempted,
        "requests_completed": len(completed),
        "requests_failed": len(failed),
        "requests_measured": len(measured),
        "warmup_requests": warmup_requests,
        "window_s": window_s,
        "ttft_ms": _dist(ttft),
        "tpot_ms": _dist(tpot),
        "e2e_ms": _dist(e2e),
        "itl_ms": _dist(itl),
        "output_tokens_total": out_tokens,
        "output_tokens_per_s": out_tokens / window_s if window_s > 0 else 0.0,
        "goodput_requests": len(good),
        "goodput_req_s": goodput_rate,
        "goodput_fraction_of_attempted": len(good) / attempted if attempted else 0.0,
        "slo": {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms},
        "prefix_cache_hit_rate": float(
            (engine_stats or {}).get("prefix_cache_hit_rate", 0.0)
        ),
        "kv_bytes_offloaded": offload_bytes,
        "kv_bytes_restored": restore_bytes,
        "offload_bandwidth_gib_s": offload_bytes / window_s / (1024**3)
        if window_s > 0
        else 0.0,
        "restore_bandwidth_gib_s": restore_bytes / window_s / (1024**3)
        if window_s > 0
        else 0.0,
        "restore_service_gib_s": restore_bytes / service_s / (1024**3)
        if service_s > 0
        else 0.0,
        "per_tier_hits": (store_stats or {}).get("hit_distribution", []),
    }
