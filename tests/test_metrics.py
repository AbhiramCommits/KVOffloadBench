"""Exact TTFT/TPOT/ITL/goodput arithmetic, including SLO boundaries."""

from __future__ import annotations

import pytest
from kvoffloadbench.metrics import (
    aggregate_requests,
    percentile,
    request_timings,
    slo_met,
)

MS = 1_000_000


def test_request_timings_exact_values() -> None:
    timings = request_timings(
        0, [100 * MS, 150 * MS, 200 * MS, 300 * MS]
    )
    assert timings["ttft_ms"] == pytest.approx(100.0)
    assert timings["e2e_ms"] == pytest.approx(300.0)
    assert timings["tpot_ms"] == pytest.approx(200.0 / 3.0)
    assert timings["itl_ms"] == [50.0, 50.0, 100.0]
    assert timings["output_tokens"] == 4


def test_single_token_tpot_is_zero() -> None:
    timings = request_timings(10 * MS, [42 * MS])
    assert timings["ttft_ms"] == pytest.approx(32.0)
    assert timings["tpot_ms"] == 0.0
    assert timings["itl_ms"] == []


def test_percentile_linear_interpolation() -> None:
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    assert percentile([50, 50, 100], 99) == pytest.approx(99.0)
    assert percentile([], 99) == 0.0
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_slo_boundary_is_inclusive() -> None:
    assert slo_met(2000.0, 100.0, 2000.0, 100.0) is True
    assert slo_met(2000.0001, 100.0, 2000.0, 100.0) is False
    assert slo_met(2000.0, 100.0001, 2000.0, 100.0) is False


def _record(
    ttft: float,
    tpot: float,
    *,
    slo_ttft: float = 100.0,
    slo_tpot: float = 50.0,
    warmup: bool = False,
    status: str = "completed",
) -> dict:
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": ttft + tpot * 2,
        "itl_ms": [tpot, tpot],
        "output_tokens": 3,
        "status": status,
        "warmup": warmup,
        "slo_met": ttft <= slo_ttft and tpot <= slo_tpot,
    }


def test_aggregate_goodput_and_fraction() -> None:
    records = [
        _record(100.0, 50.0),  # exactly at both boundaries -> good
        _record(99.0, 49.0),  # good
        _record(101.0, 10.0),  # TTFT over
        _record(10.0, 51.0),  # TPOT over
        _record(100.0, 50.0, warmup=True),  # excluded
        _record(50.0, 5.0, status="failed"),
    ]
    agg = aggregate_requests(
        records,
        slo_ttft_ms=100.0,
        slo_tpot_ms=50.0,
        window_s=10.0,
        attempted=4,
        engine_stats={"prefix_cache_hit_rate": 0.25},
        store_stats={
            "counters": {
                "offload_bytes": 1024**3,
                "restore_bytes": 512 * 1024**2,
                "lookup_restore_bytes": 0,
                "restore_seconds": 0.5,
                "lookup_restore_seconds": 0.0,
            },
            "hit_distribution": [10, 2],
        },
        warmup_requests=1,
    )
    assert agg["requests_measured"] == 5
    assert agg["requests_completed"] == 4
    assert agg["requests_failed"] == 1
    assert agg["goodput_requests"] == 2
    assert agg["goodput_req_s"] == pytest.approx(0.2)
    assert agg["goodput_fraction_of_attempted"] == pytest.approx(0.5)
    assert agg["ttft_ms"]["p50"] == pytest.approx(99.5)
    assert agg["ttft_ms"]["p99"] == pytest.approx(100.97)
    assert agg["prefix_cache_hit_rate"] == 0.25
    assert agg["restore_bandwidth_gib_s"] == pytest.approx(0.5 / 10.0)
    assert agg["offload_bandwidth_gib_s"] == pytest.approx(1.0 / 10.0)
    assert agg["restore_service_gib_s"] == pytest.approx(1.0)
    assert agg["per_tier_hits"] == [10, 2]
