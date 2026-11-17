"""Requirement formulas and the generated table."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from kvoffloadbench.analysis.requirements import (
    LLAMA70B_BYTES_PER_TOKEN,
    build_requirements,
    p99_block_latency_budget_us,
    required_iops,
    required_read_bandwidth_gib_s,
)

GIB = 1024**3
BLOCK = 128 * 1024


def test_llama70b_requirements_on_known_inputs() -> None:
    kv_bytes = 32768 * LLAMA70B_BYTES_PER_TOKEN
    assert kv_bytes == 10 * GIB  # 10 GiB per 32k-token 70B request
    assert required_read_bandwidth_gib_s(kv_bytes, 4.0) == pytest.approx(2.5)
    assert required_iops(kv_bytes, BLOCK, 4.0) == pytest.approx(20480)
    assert p99_block_latency_budget_us(4000.0, 81920) == pytest.approx(48.828125)
    assert p99_block_latency_budget_us(4000.0, 0) == 0.0


def test_local_requirement_formulas() -> None:
    kv_bytes = 4096 * 6144  # 24 MiB
    assert kv_bytes / (1024**2) == 24
    assert required_read_bandwidth_gib_s(kv_bytes, 3.124) == pytest.approx(
        24 / 1024 / 3.124, rel=1e-6
    )
    assert required_iops(kv_bytes, BLOCK, 3.124) == pytest.approx(192 / 3.124)
    with pytest.raises(ValueError):
        required_read_bandwidth_gib_s(kv_bytes, 0)
    with pytest.raises(ValueError):
        required_iops(kv_bytes, 0, 1.0)


def test_build_requirements_from_synthetic_results(tmp_path) -> None:
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "engine_calibration.json").write_text(
        json.dumps({"prefill_ms_per_token": 0.25, "device": "unit-test"})
    )
    (tmp_path / "raw" / "run1").mkdir(parents=True)
    (tmp_path / "raw" / "run1" / "manifest.json").write_text(
        json.dumps(
            {
                "model_shape": {
                    "layers": 12,
                    "kv_heads": 2,
                    "head_dim": 64,
                    "bytes_per_token": 6144,
                },
                "config": {"slo": {"ttft_ms": 4000}},
            }
        )
    )
    pd.DataFrame(
        [
            {
                "run_id": "run1",
                "tier_chain": "dram_nvme",
                "context_tokens": 4096,
                "request_rate": 1.0,
                "restore_service_gib_s": 0.5,
                "restore_bandwidth_gib_s": 0.2,
            }
        ]
    ).to_csv(tmp_path / "index.csv", index=False)
    frame = build_requirements(tmp_path)
    assert len(frame) == 24  # 2 scopes x 3 contexts x 4 rates
    local = frame[(frame["scope"] == "local") & (frame["context_tokens"] == 4096)].iloc[0]
    assert local["bytes_per_token"] == 6144
    assert local["kv_mib_per_request"] == pytest.approx(24.0)
    assert local["prefill_compute_ms_local"] == pytest.approx(1024.0)
    assert local["io_budget_after_local_compute_ms"] == pytest.approx(2976.0)
    assert local["measured_best_restore_gib_s"] == pytest.approx(0.5)
    assert (tmp_path / "tables" / "storage_requirements.md").is_file()
    assert (tmp_path / "tables" / "storage_requirements.csv").is_file()
