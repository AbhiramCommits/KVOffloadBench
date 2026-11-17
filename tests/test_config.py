"""Config validation and sweep expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from kvoffloadbench.config import (
    ConfigError,
    TierConfig,
    load_run_config,
    load_sweep,
)

ROOT = Path(__file__).resolve().parents[1]


def test_main_sweep_expands_to_64_runs() -> None:
    runs = load_sweep(ROOT / "configs" / "sweeps" / "main.yaml")
    assert len(runs) == 64
    assert len({r.run_id() for r in runs}) == 64
    tiers = {r.tier_chain for r in runs}
    assert tiers == {"dram_only", "dram_nvme", "dram_nvme_remote_tcp"}
    assert {r.workload.request_rate for r in runs} == {1.0, 2.0, 4.0, 8.0}
    assert {r.workload.context_tokens for r in runs} == {4096, 16384, 32768}
    assert {r.capacity_label for r in runs} == {"small", "medium", "large"}
    assert {r.workload.trace for r in runs} == {
        "longctx_multiturn",
        "agent_react",
        "shared_prefix_rag",
    }


def test_include_and_exclude_expansion(tmp_path) -> None:
    config = {
        "name": "unit",
        "engine": {"mode": "emulated"},
        "chains": {
            "a": [{"kind": "dram", "capacity_mib": "${capacity}"}],
            "b": [{"kind": "dram", "capacity_mib": "${capacity}"}],
        },
        "axes": {
            "tier_chain": ["a", "b"],
            "request_rate": [1, 2],
            "capacity": {"small": 64, "medium": 512},
            "trace": ["agent_react"],
        },
        "include": [
            {
                "tier_chain": ["a", "b"],
                "request_rate": [1, 2],
                "capacity": ["small", "medium"],
            }
        ],
        "exclude": [{"tier_chain": "b", "request_rate": 2}],
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))
    runs = load_sweep(path)
    assert len(runs) == 6
    assert all(not (r.tier_chain == "b" and r.workload.request_rate == 2) for r in runs)
    assert {t.capacity_mib for r in runs for t in r.tiers} == {64, 512}


def test_run_config_roundtrip(tmp_path) -> None:
    config = {
        "name": "one",
        "engine": {"mode": "auto", "model": "gpt2"},
        "tiers": [
            {"kind": "dram", "capacity_mib": 128, "block_size_kib": 128},
            {
                "kind": "nvme",
                "capacity_mib": 256,
                "path": str(tmp_path / "nvme.bin"),
                "block_size_kib": 128,
            },
        ],
        "workload": {
            "trace": "agent_react",
            "request_rate": 2,
            "context_tokens": 4096,
            "max_concurrency": 4,
            "duration_s": 5,
            "warmup_s": 1,
        },
        "slo": {"ttft_ms": 3000, "tpot_ms": 100},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    run = load_run_config(path)
    assert run.workload.request_rate == 2.0
    assert [t.kind for t in run.tiers] == ["dram", "nvme"]
    payload = run.to_dict()
    assert payload["run_id"].startswith("one-dram-nvme")


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "tape", "capacity_mib": 1},
        {"kind": "dram", "capacity_mib": 0},
        {"kind": "dram", "capacity_mib": 1, "block_size_kib": 7},
        {"kind": "nvme", "capacity_mib": 1, "block_size_kib": 64},
    ],
)
def test_invalid_tiers_raise(bad: dict) -> None:
    with pytest.raises(ConfigError):
        TierConfig(**bad)


def test_unknown_field_raises(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "tiers": [{"kind": "dram", "capacity_mib": 1, "bogus": True}],
            }
        )
    )
    with pytest.raises(ConfigError, match="unknown field"):
        load_run_config(path)


def test_invalid_engine_mode_and_missing_remote_endpoint(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "engine": {"mode": "warp"},
                "tiers": [{"kind": "dram", "capacity_mib": 1}],
            }
        )
    )
    with pytest.raises(ConfigError, match=r"engine\.mode"):
        load_run_config(path)
    with pytest.raises(ConfigError, match="endpoint"):
        TierConfig(kind="remote_tcp", capacity_mib=1)


def test_sweep_rejects_unknown_include_label(tmp_path) -> None:
    config = {
        "name": "unit",
        "chains": {"a": [{"kind": "dram", "capacity_mib": "${cap}"}]},
        "axes": {
            "tier_chain": ["a"],
            "request_rate": [1],
            "cap": {"small": 64},
            "trace": ["agent_react"],
        },
        "include": [{"cap": ["huge"]}],
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match=r"not a label"):
        load_sweep(path)
