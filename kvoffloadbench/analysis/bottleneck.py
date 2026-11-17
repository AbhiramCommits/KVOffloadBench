"""Bottleneck attribution from measured runs.

For every run this reads the per-request parquet, the tier counters, and the
Step-2 tier microbenchmark. It decomposes the measured TTFT into queueing,
calibrated compute, and storage restore time (all per completed request) and
compares achieved restore bandwidth/IOPS with the measured tier ceiling.

The label is deliberately conservative: when the SLO is comfortably met the run
is labelled ``slo_headroom`` rather than blaming a resource.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS = Path("results")
TIERS_CSV = RESULTS / "tiers" / "microbench.csv"
INDEX_CSV = RESULTS / "index.csv"
TABLE_DIR = RESULTS / "tables"


def _tier_map() -> dict[str, str]:
    return {"dram": "dram", "nvme": "nvme_local", "remote_tcp": "remote_tcp", "nvmeof_block": "nvmeof_block"}


def _ceilings() -> dict[str, dict[str, Any]]:
    if not TIERS_CSV.is_file():
        return {}
    df = pd.read_csv(TIERS_CSV)
    df = df[(df["op"] == "read")]
    out: dict[str, dict[str, Any]] = {}
    for tier, group in df.groupby("tier"):
        # The KV block is 96 KiB and the storage slot is 128 KiB; the closest
        # measured points are 64 and 256 KiB. Take the best of the two.
        near = group[group["block_kib"].isin([64, 256])]
        if near.empty:
            near = group
        best_bw = near.iloc[int(near["gib_s"].to_numpy().argmax())]
        best_iops = near.iloc[int(near["iops"].to_numpy().argmax())]
        out[str(tier)] = {
            "ceiling_read_gib_s": float(best_bw["gib_s"]),
            "ceiling_read_iops": float(best_iops["iops"]),
            "ceiling_block_kib": float(best_bw["block_kib"]),
            "ceiling_iops_block_kib": float(best_iops["block_kib"]),
            "ceiling_depth": int(best_bw["depth"]),
            "ceiling_io_mode": str(best_bw["io_mode"]),
        }
    return out


def attribute_run(
    metrics: dict[str, Any],
    tier_stats: dict[str, Any],
    requests: pd.DataFrame | None,
    ceilings: dict[str, dict[str, Any]],
    block_bytes: int,
) -> dict[str, Any]:
    active = list(metrics.get("active_tiers") or [])
    cold_tier = active[-1] if active else "dram"
    mapped = _tier_map().get(cold_tier, cold_tier)
    ceiling = ceilings.get(mapped, {})
    counters = (tier_stats.get("store") or {}).get("counters", {})
    engine = tier_stats.get("engine") or {}
    completed = max(1, int(metrics.get("requests_completed") or 0))
    wall_s = float(metrics.get("wall_clock_s") or 0.0)
    warmup_s = float(metrics.get("warmup_s") or 0.0)
    active_s = max(1e-6, wall_s - warmup_s)
    conc = max(1, int(metrics.get("max_concurrency") or 1))

    compute_s = float(engine.get("prefill_compute_s", 0.0)) + float(
        engine.get("decode_compute_s", 0.0)
    )
    restore_s = float(counters.get("restore_seconds", 0.0))
    queue_s = 0.0
    if requests is not None and not requests.empty and "queue_ms" in requests:
        measured = requests[~requests["warmup"].astype(bool)] if "warmup" in requests else requests
        queue_s = float(measured["queue_ms"].fillna(0).sum()) / 1000.0

    mean_ttft = float(metrics.get("ttft_ms", {}).get("p50", 0.0))
    per_req_compute_ms = compute_s / completed * 1000.0
    per_req_restore_ms = restore_s / completed * 1000.0
    per_req_queue_ms = queue_s / completed * 1000.0
    shares = {
        "compute": per_req_compute_ms / mean_ttft if mean_ttft > 0 else 0.0,
        "restore": per_req_restore_ms / mean_ttft if mean_ttft > 0 else 0.0,
        "queue": per_req_queue_ms / mean_ttft if mean_ttft > 0 else 0.0,
    }
    other_share = max(0.0, 1.0 - sum(shares.values()))
    util = {
        "compute": compute_s / (active_s * conc),
        "restore": restore_s / (active_s * conc),
        "queue": queue_s / (active_s * conc),
    }
    restore_bytes = int(counters.get("restore_bytes", 0))
    achieved_bw = restore_bytes / restore_s / (1024**3) if restore_s > 0 else 0.0
    achieved_iops = restore_bytes / block_bytes / restore_s if restore_s > 0 else 0.0
    pressure_bw = (
        achieved_bw / ceiling["ceiling_read_gib_s"]
        if ceiling.get("ceiling_read_gib_s")
        else 0.0
    )
    pressure_iops = (
        achieved_iops / ceiling["ceiling_read_iops"]
        if ceiling.get("ceiling_read_iops")
        else 0.0
    )
    goodput_fraction = float(metrics.get("goodput_fraction_of_attempted") or 0.0)
    dominant = max(shares, key=lambda k: shares[k])
    if goodput_fraction >= 0.99:
        binding = "slo_headroom"
    elif dominant == "restore":
        binding = (
            f"storage_{cold_tier}" if max(pressure_bw, pressure_iops) >= 0.5 else "client_storage_overhead"
        )
    elif dominant == "compute":
        binding = "compute_calibrated"
    elif dominant == "queue":
        binding = "queueing_concurrency"
    else:
        binding = "client_overhead"
    note = ""
    if metrics.get("engine_mode") == "emulated":
        note = (
            "compute time is the calibrated model on this host "
            f"({engine.get('calibration', {}).get('device', 'n/a')}); storage time is real I/O"
        )
    return {
        "run_id": metrics.get("run_id"),
        "tier_chain": metrics.get("tier_chain"),
        "trace": Path(str(metrics.get("trace", ""))).name,
        "request_rate": metrics.get("request_rate"),
        "context_tokens": metrics.get("context_tokens"),
        "capacity_label": metrics.get("capacity_label"),
        "active_tiers": "+".join(active),
        "cold_tier": cold_tier,
        "wall_s": wall_s,
        "active_s": active_s,
        "max_concurrency": conc,
        "requests_completed": completed,
        "goodput_fraction": goodput_fraction,
        "ttft_p50_ms": mean_ttft,
        "ttft_p99_ms": float(metrics.get("ttft_ms", {}).get("p99", 0.0)),
        "compute_s": compute_s,
        "restore_s": restore_s,
        "queue_s": queue_s,
        "per_req_compute_ms": per_req_compute_ms,
        "per_req_restore_ms": per_req_restore_ms,
        "per_req_queue_ms": per_req_queue_ms,
        "compute_share": shares["compute"],
        "restore_share": shares["restore"],
        "queue_share": shares["queue"],
        "other_share": other_share,
        "compute_util": util["compute"],
        "restore_util": util["restore"],
        "queue_util": util["queue"],
        "restore_bytes": restore_bytes,
        "achieved_restore_gib_s": achieved_bw,
        "ceiling_read_gib_s": ceiling.get("ceiling_read_gib_s", 0.0),
        "bandwidth_pressure": pressure_bw,
        "achieved_restore_iops": achieved_iops,
        "ceiling_read_iops": ceiling.get("ceiling_read_iops", 0.0),
        "iops_pressure": pressure_iops,
        "binding_resource": binding,
        "notes": note,
    }


def analyse(results_root: str | Path = RESULTS) -> pd.DataFrame:
    root = Path(results_root)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    if not (root / "index.csv").is_file():
        raise FileNotFoundError(f"{root / 'index.csv'} missing; run the sweep first")
    index = pd.read_csv(root / "index.csv")
    ceilings = _ceilings()
    rows: list[dict[str, Any]] = []
    for _, run in index.iterrows():
        rid = str(run["run_id"])
        raw = root / "raw" / rid
        metrics_path = raw / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        tier_stats = json.loads((raw / "tier_stats.json").read_text())
        req: pd.DataFrame | None = None
        parquet = raw / "requests.parquet"
        if parquet.is_file():
            try:
                req = pd.read_parquet(parquet)
            except Exception:
                req = None
        block_bytes = 96 * 1024
        if req is not None and not req.empty and "prompt_tokens" in req:
            shape = tier_stats.get("engine", {}).get("model_shape", {})
            if shape:
                block_bytes = int(shape.get("bytes_per_block", block_bytes))
        rows.append(attribute_run(metrics, tier_stats, req, ceilings, block_bytes))
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "bottleneck_attribution.csv", index=False)
    _write_markdown(frame, ceilings, table_dir)
    return frame


def _write_markdown(
    frame: pd.DataFrame,
    ceilings: dict[str, dict[str, Any]],
    table_dir: Path,
) -> None:
    lines = [
        "# Bottleneck attribution",
        "",
        "Generated from every run in `results/index.csv` and `results/raw/**` by",
        "`kvoffloadbench analyze`. All storage time is measured I/O; compute time is the",
        "calibrated model timing recorded in `results/calibration/engine_calibration.json`.",
        "",
        "## Method",
        "",
        "- Per-request decomposition: queue wait (measured in the harness) + calibrated prefill/decode",
        "  compute + storage restore time (measured inside `TieredKVStore.get`) + remainder.",
        "- Shares are relative to measured TTFT p50; utilizations use wall time excluding warmup",
        "  multiplied by `max_concurrency`.",
        "- Achieved restore bandwidth/IOPS are compared with the best measured read ceiling from",
        "  `results/tiers/microbench.csv` at the 64 KiB or 256 KiB point (the KV block is 96 KiB).",
        "- A run that meets the SLO with a large headroom is labelled `slo_headroom`; the tool does",
        "  not invent a bottleneck when none is binding.",
        "",
        "## Measured tier ceilings used",
        "",
        "| tier | read GiB/s | block KiB | depth | IOPS | IOPS block KiB | I/O mode |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tier, c in sorted(ceilings.items()):
        lines.append(
            f"| {tier} | {c['ceiling_read_gib_s']:.3f} | {c['ceiling_block_kib']:.0f} | "
            f"{c['ceiling_depth']} | {c['ceiling_read_iops']:.0f} | "
            f"{c['ceiling_iops_block_kib']:.0f} | {c['ceiling_io_mode']} |"
        )
    lines += ["", "## Binding resource counts", ""]
    if frame.empty:
        lines.append("No runs found.")
    else:
        counts = frame["binding_resource"].value_counts()
        lines += ["| binding resource | runs |", "|---|---:|"]
        for name, count in counts.items():
            lines.append(f"| {name} | {int(count)} |")
        lines += [
            "",
            "## Runs with the highest measured TTFT p99",
            "",
            "| run | tiers | trace | rate | ctx | TTFT p99 ms | goodput frac | restore GiB/s | achieved/ceiling | binding |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        top = frame.sort_values("ttft_p99_ms", ascending=False).head(15)
        for _, r in top.iterrows():
            pressure = max(float(r["bandwidth_pressure"]), float(r["iops_pressure"]))
            lines.append(
                f"| `{r['run_id']}` | {r['active_tiers']} | {r['trace']} | "
                f"{r['request_rate']:g} | {int(r['context_tokens'])} | {r['ttft_p99_ms']:.0f} | "
                f"{r['goodput_fraction']:.2f} | {r['achieved_restore_gib_s']:.3f} | "
                f"{pressure:.2f} | {r['binding_resource']} |"
            )
        lines += [
            "",
            "## Interpretation",
            "",
            "The emulated compute model uses this laptop's measured gpt2 timings, so `compute_calibrated`",
            "labels mean the calibrated prefill/decode budget dominates, not that an H200 was saturated.",
            "`queueing_concurrency` means arrivals outran the in-flight limit before storage could be blamed.",
            "`storage_*` labels are reserved for runs whose measured restore rate actually approached the",
            "measured tier ceiling; the achieved/ceiling column makes that comparison explicit.",
        ]
    (table_dir / "bottleneck_attribution.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    analyse()
