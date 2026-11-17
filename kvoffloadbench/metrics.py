"""Latency sampling and summary statistics shared by tiers, engines, and analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


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
