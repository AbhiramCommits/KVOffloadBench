"""Sweep runner: expand the matrix, run points sequentially, build the index.

Isolation between points: every point builds a fresh store (new file/TCP
target), the TCP target is started and stopped per point, and an optional
OS page-cache drop is available on Linux. Resume is by run_id: a point with a
``metrics.json`` is skipped unless ``--force`` is given.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from kvoffloadbench.config import RunConfig, load_sweep
from kvoffloadbench.harness.replay import run_replay

log = logging.getLogger(__name__)
console = Console()

INDEX_COLUMNS = [
    "run_id",
    "name",
    "tier_chain",
    "active_tiers",
    "capacity_label",
    "trace",
    "request_rate",
    "context_tokens",
    "duration_s",
    "warmup_s",
    "max_concurrency",
    "engine_mode",
    "requests_attempted",
    "requests_completed",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "itl_p99_ms",
    "e2e_p50_ms",
    "goodput_req_s",
    "goodput_fraction_of_attempted",
    "prefix_cache_hit_rate",
    "kv_bytes_offloaded",
    "kv_bytes_restored",
    "offload_bandwidth_gib_s",
    "restore_bandwidth_gib_s",
    "restore_service_gib_s",
    "per_tier_hits",
    "wall_clock_s",
    "nvmeof_fell_back",
]


def _row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    def dist(key: str, field: str) -> float:
        return float(metrics.get(key, {}).get(field, 0.0))

    trace = Path(str(metrics.get("trace", ""))).name
    return {
        "run_id": metrics["run_id"],
        "name": metrics.get("name", ""),
        "tier_chain": metrics.get("tier_chain", ""),
        "active_tiers": "+".join(metrics.get("active_tiers", [])),
        "capacity_label": metrics.get("capacity_label", ""),
        "trace": trace,
        "request_rate": metrics.get("request_rate"),
        "context_tokens": metrics.get("context_tokens"),
        "duration_s": metrics.get("duration_s"),
        "warmup_s": metrics.get("warmup_s"),
        "max_concurrency": metrics.get("max_concurrency"),
        "engine_mode": metrics.get("engine_mode"),
        "requests_attempted": metrics.get("requests_attempted"),
        "requests_completed": metrics.get("requests_completed"),
        "ttft_p50_ms": dist("ttft_ms", "p50"),
        "ttft_p95_ms": dist("ttft_ms", "p95"),
        "ttft_p99_ms": dist("ttft_ms", "p99"),
        "tpot_p50_ms": dist("tpot_ms", "p50"),
        "tpot_p99_ms": dist("tpot_ms", "p99"),
        "itl_p99_ms": dist("itl_ms", "p99"),
        "e2e_p50_ms": dist("e2e_ms", "p50"),
        "goodput_req_s": metrics.get("goodput_req_s"),
        "goodput_fraction_of_attempted": metrics.get("goodput_fraction_of_attempted"),
        "prefix_cache_hit_rate": metrics.get("prefix_cache_hit_rate"),
        "kv_bytes_offloaded": metrics.get("kv_bytes_offloaded"),
        "kv_bytes_restored": metrics.get("kv_bytes_restored"),
        "offload_bandwidth_gib_s": metrics.get("offload_bandwidth_gib_s"),
        "restore_bandwidth_gib_s": metrics.get("restore_bandwidth_gib_s"),
        "restore_service_gib_s": metrics.get("restore_service_gib_s"),
        "per_tier_hits": json.dumps(metrics.get("per_tier_hits", [])),
        "wall_clock_s": metrics.get("wall_clock_s"),
        "nvmeof_fell_back": metrics.get("nvmeof_fell_back", False),
    }


def build_index(results_root: str | Path = "results") -> pd.DataFrame:
    """Rebuild results/index.csv from every metrics.json actually on disk."""
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "raw").glob("*/metrics.json")):
        try:
            metrics = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning("skipping unreadable %s", path)
            continue
        rows.append(_row_from_metrics(metrics))
    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["trace", "tier_chain", "context_tokens", "request_rate", "capacity_label"],
            kind="stable",
        )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "index.csv", index=False)
    return frame


def _drop_caches() -> bool:
    if platform.system() != "Linux":
        log.warning("page-cache drop requested but this host is %s; skipped", platform.system())
        return False
    try:
        subprocess.run(["sync"], check=False)
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    except (OSError, PermissionError) as exc:
        log.warning("could not drop page caches: %s", exc)
        return False


def _print_plan(runs: list[RunConfig]) -> None:
    table = Table(title=f"Sweep plan ({len(runs)} runs)")
    for col in ("run_id", "trace", "rate", "context", "cache"):
        table.add_column(col, justify="left" if col in ("run_id", "trace") else "right")
    for run in runs:
        table.add_row(
            run.run_id(),
            Path(run.workload.trace).stem,
            f"{run.workload.request_rate:g}",
            str(run.workload.context_tokens),
            run.capacity_label,
        )
    console.print(table)


def run_sweep(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    limit: int | None = None,
    results_root: str | Path = "results",
) -> pd.DataFrame:
    runs = load_sweep(config_path)
    if limit is not None:
        runs = runs[: max(0, limit)]
    _print_plan(runs)
    root = Path(results_root)
    if dry_run:
        build_index(root)
        return pd.DataFrame()
    raw = root / "raw"
    drop = False
    with contextlib.suppress(OSError, yaml.YAMLError):
        drop = bool(yaml.safe_load(Path(config_path).read_text()).get("drop_caches", False))
    total = len(runs)
    sweep_t0 = time.perf_counter()
    failures: list[dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        rid = run.run_id()
        outdir = raw / rid
        metrics_path = outdir / "metrics.json"
        if metrics_path.is_file() and not force:
            console.print(f"[dim][{i}/{total}] skip existing {rid}[/dim]")
            continue
        elapsed = time.perf_counter() - sweep_t0
        eta = (elapsed / max(1, i - 1)) * (total - i + 1) if i > 1 else 0.0
        console.print(
            f"[bold cyan][{i}/{total}][/bold cyan] {rid} "
            f"[dim](elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min)[/dim]"
        )
        run_t0 = time.perf_counter()
        try:
            asyncio.run(run_replay(run, out_root=raw))
        except Exception as exc:
            log.exception("run %s failed", rid)
            failures.append({"run_id": rid, "error": f"{type(exc).__name__}: {exc}"})
            console.print(f"[red]run failed: {rid}: {exc}[/red]")
        finally:
            build_index(results_root)
        console.print(f"[dim]  point wall clock {time.perf_counter() - run_t0:.1f}s[/dim]")
        if drop:
            _drop_caches()
    frame = build_index(results_root)
    if failures:
        (Path(results_root) / "sweep_failures.json").write_text(
            json.dumps(failures, indent=2)
        )
    console.print(
        f"[green]sweep done: {len(frame)} rows in {Path(results_root) / 'index.csv'} "
        f"in {(time.perf_counter() - sweep_t0) / 60:.1f} min[/green]"
    )
    if not frame.empty:
        summary = (
            frame.groupby(["tier_chain", "request_rate"], dropna=False)
            .agg(
                runs=("run_id", "count"),
                ttft_p50=("ttft_p50_ms", "median"),
                goodput=("goodput_req_s", "median"),
                hit=("prefix_cache_hit_rate", "median"),
            )
            .reset_index()
        )
        console.print(summary.to_string(index=False))
    return frame
