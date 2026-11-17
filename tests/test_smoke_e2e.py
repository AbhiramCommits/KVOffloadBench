"""End-to-end smoke: a 2-point emulated sweep produces real artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from kvoffloadbench.harness.sweep import run_sweep

ROOT = Path(__file__).resolve().parents[1]


def test_two_point_smoke_sweep(tmp_path, monkeypatch) -> None:
    trace = (ROOT / "traces" / "agent_react.jsonl").resolve()
    assert trace.is_file(), "run `kvoffloadbench gen-traces` before the smoke test"
    calibration = (ROOT / "results" / "calibration" / "engine_calibration.json").resolve()
    config = {
        "name": "main",
        "description": "smoke",
        "engine": {
            "mode": "emulated",
            "model": "gpt2",
            "calibration_path": str(calibration),
            "model_shape": {
                "layers": 12,
                "kv_heads": 2,
                "head_dim": 64,
                "dtype": "float16",
                "block_tokens": 16,
            },
        },
        "slo": {"ttft_ms": 4000, "tpot_ms": 120},
        "workload_defaults": {
            "max_concurrency": 4,
            "duration_s": 0.5,
            "warmup_s": 0.1,
            "seed": 7,
        },
        "chains": {
            "dram_only": [
                {"kind": "dram", "capacity_mib": 64, "block_size_kib": 128}
            ]
        },
        "axes": {
            "tier_chain": ["dram_only"],
            "request_rate": [1, 2],
            "context_tokens": [512],
            "kv_cache_capacity_mib": {"small": 64},
            "trace": [str(trace)],
        },
        "include": [
            {
                "tier_chain": ["dram_only"],
                "request_rate": [1, 2],
                "context_tokens": [512],
                "kv_cache_capacity_mib": ["small"],
            }
        ],
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "results"
    frame = run_sweep(cfg_path, results_root=out)
    assert len(frame) == 2
    index = pd.read_csv(out / "index.csv")
    assert len(index) == 2
    assert set(index["engine_mode"]) == {"emulated"}
    parquet_files = list((out / "raw").glob("*/requests.parquet"))
    assert len(parquet_files) == 2
    req = pd.read_parquet(parquet_files[0])
    assert not req.empty
    assert {"ttft_ms", "tpot_ms", "status", "warmup"} <= set(req.columns)
    metrics_files = list((out / "raw").glob("*/metrics.json"))
    assert len(metrics_files) == 2
    metrics = json.loads(metrics_files[0].read_text())
    assert metrics["engine_mode"] == "emulated"
    assert metrics["requests_completed"] >= 1

    # A figure must be produced from the two points and be non-trivial in size.
    from kvoffloadbench.analysis import plots

    monkeypatch.setattr(plots, "FIGURES", tmp_path / "figures")
    figure = plots.plot_ttft_vs_rate(
        index[index["name"] == "main"].assign(name="main")
    )
    assert figure.is_file()
    assert figure.stat().st_size > 5000
