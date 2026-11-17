"""Async replay harness with Poisson arrivals, warmup, and per-request metrics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from kvoffloadbench.config import ConfigError, RunConfig
from kvoffloadbench.engines.base import GenerationRequest
from kvoffloadbench.engines.factory import build_engine
from kvoffloadbench.env import cached_fingerprint
from kvoffloadbench.harness.prom import ReplayMetrics
from kvoffloadbench.metrics import aggregate_requests, request_timings, slo_met
from kvoffloadbench.target.launcher import TargetHandle, start_target
from kvoffloadbench.tiers.tiered_store import build_store
from kvoffloadbench.traces.generate import iter_trace

log = logging.getLogger(__name__)
console = Console()

RAW_ROOT = Path("results/raw")


def resolve_trace(trace: str) -> Path:
    direct = Path(trace)
    if direct.is_file():
        return direct
    candidates = [
        Path("traces") / direct.name,
        Path("traces") / f"{direct.name}.jsonl",
        Path("traces") / f"{direct.stem}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"trace {trace!r} not found; run `kvoffloadbench gen-traces --out traces/` first"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _start_remote_target(run: RunConfig, run_id: str) -> TargetHandle | None:
    for tier in run.tiers:
        if tier.kind == "remote_tcp":
            target_path = tier.extra.get("target_path") or f"results/cache/target/{run_id}.bin"
            return start_target(
                tier.endpoint or "127.0.0.1:4420",
                str(target_path),
                int(tier.extra.get("target_capacity_mib", tier.capacity_mib)),
            )
    return None


async def _drain(tasks: set[asyncio.Task[None]], timeout: float) -> None:
    if not tasks:
        return
    _done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        log.warning("cancelled %d requests that exceeded the %.0fs drain timeout", len(pending), timeout)


async def run_replay(
    run: RunConfig,
    *,
    metrics_port: int = 0,
    duration_override: float | None = None,
    out_root: str | Path = RAW_ROOT,
    drain_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Execute one run, write results/raw/<run_id>/, and return the metrics dict."""
    run_id = run.run_id()
    outdir = Path(out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = resolve_trace(run.workload.trace)
    trace = list(iter_trace(trace_path))
    if not trace:
        raise ConfigError(f"trace {trace_path} is empty")
    duration = float(duration_override if duration_override is not None else run.workload.duration_s)
    warmup = float(run.workload.warmup_s)
    rate = float(run.workload.request_rate)
    if rate <= 0:
        raise ConfigError("request_rate must be > 0")
    max_conc = int(run.workload.max_concurrency)
    seed = int(run.workload.seed)

    started = datetime.now(UTC)
    wall_t0 = time.perf_counter()
    target = _start_remote_target(run, run_id)
    store = None
    engine = None
    metrics = ReplayMetrics(port=metrics_port, run_id=run_id)
    metrics.start()
    records: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max_conc)
    attempted = 0
    warmup_count = 0
    rng = random.Random(seed)
    active_tiers: list[str] = []
    requested_tiers = [t.kind for t in run.tiers]
    try:
        store = build_store(run.tiers, name=run_id)
        active_tiers = [t.kind for t in store.tiers]
        engine = await build_engine(run.engine, store, run_id, list(run.tiers))

        async def serve(rec: dict[str, Any], arrival_ns: int, warmup_flag: bool) -> None:
            queue_t0 = time.perf_counter_ns()
            prompt_tokens = min(
                int(rec["prompt_tokens"]), int(run.workload.context_tokens)
            )
            req = GenerationRequest(
                request_id=f"{run_id}-{rec['id']}-{arrival_ns}",
                text=str(rec["text"]),
                max_tokens=int(rec["output_tokens"]),
                temperature=0.0,
                prompt_tokens=prompt_tokens,
                prefix_group=str(rec["prefix_group"]),
                arrival_ns=arrival_ns,
                seed=seed,
                turn_index=int(rec["turn_index"]),
            )
            row: dict[str, Any] = {
                "run_id": run_id,
                "trace_id": rec["id"],
                "prefix_group": rec["prefix_group"],
                "turn_index": rec["turn_index"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": 0,
                "arrival_ns": arrival_ns,
                "queue_ms": 0.0,
                "start_ns": 0,
                "first_token_ns": None,
                "last_token_ns": None,
                "ttft_ms": None,
                "tpot_ms": None,
                "e2e_ms": None,
                "itl_ms": [],
                "status": "failed",
                "error": "",
                "warmup": warmup_flag,
                "slo_met": False,
            }
            metrics.inflight.inc()
            try:
                async with sem:
                    row["queue_ms"] = (time.perf_counter_ns() - queue_t0) / 1e6
                    row["start_ns"] = time.perf_counter_ns()
                    timestamps: list[int] = []
                    async for event in engine.generate(req):
                        timestamps.append(event.timestamp_ns)
                    if not timestamps:
                        row["status"] = "failed"
                        row["error"] = "engine produced no tokens"
                    else:
                        timings = request_timings(arrival_ns, timestamps)
                        row.update(
                            {
                                "status": "completed",
                                "output_tokens": int(timings["output_tokens"]),
                                "first_token_ns": timestamps[0],
                                "last_token_ns": timestamps[-1],
                                "ttft_ms": timings["ttft_ms"],
                                "tpot_ms": timings["tpot_ms"],
                                "e2e_ms": timings["e2e_ms"],
                                "itl_ms": timings["itl_ms"],
                                "slo_met": slo_met(
                                    timings["ttft_ms"],
                                    timings["tpot_ms"],
                                    run.slo.ttft_ms,
                                    run.slo.tpot_ms,
                                ),
                            }
                        )
            except asyncio.CancelledError:
                row["status"] = "cancelled"
                row["error"] = "drain timeout"
                records.append(row)
                raise
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("request %s failed", req.request_id)
            finally:
                metrics.inflight.dec()
            records.append(row)
            if not warmup_flag:
                metrics.request_finished(
                    row.get("ttft_ms"),
                    row.get("e2e_ms"),
                    row.get("itl_ms") or [],
                    str(row["status"]),
                )

        # -- warmup phase (excluded from metrics) ---------------------------
        warm_tasks: set[asyncio.Task[None]] = set()
        warm_idx = 0
        next_arrival = 0.0
        warm_start = time.perf_counter()
        while True:
            now = time.perf_counter() - warm_start
            if now >= warmup:
                break
            if next_arrival <= now:
                rec = trace[warm_idx % len(trace)]
                warm_idx += 1
                warmup_count += 1
                task = asyncio.create_task(
                    serve(rec, time.perf_counter_ns(), True)
                )
                warm_tasks.add(task)
                task.add_done_callback(warm_tasks.discard)
                next_arrival += rng.expovariate(rate)
            else:
                await asyncio.sleep(min(next_arrival - now, 0.02))
        if warm_tasks:
            await asyncio.wait(warm_tasks, timeout=max(30.0, warmup * 4 + 10))
            for task in list(warm_tasks):
                task.cancel()
            if warm_tasks:
                await asyncio.gather(*warm_tasks, return_exceptions=True)
            warm_tasks.clear()
        if hasattr(store, "reset_stats"):
            store.reset_stats()
        if hasattr(engine, "reset_stats"):
            engine.reset_stats()
        log.warning(
            "run %s: warmup complete (%d requests), starting %.1fs measured window",
            run_id,
            warmup_count,
            duration,
        )

        # -- measured phase --------------------------------------------------
        measure_start = time.perf_counter()
        tasks: set[asyncio.Task[None]] = set()
        trace_idx = 0
        next_arrival = 0.0
        while True:
            now = time.perf_counter() - measure_start
            if now >= duration:
                break
            wait = next_arrival - now
            if wait > 0:
                await asyncio.sleep(min(wait, 0.02))
                continue
            rec = trace[trace_idx % len(trace)]
            trace_idx += 1
            attempted += 1
            task = asyncio.create_task(serve(rec, time.perf_counter_ns(), False))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            next_arrival += rng.expovariate(rate)
        await _drain(tasks, drain_timeout_s)
        window_s = duration

        store_stats = store.stats()
        engine_stats = engine.stats()
        aggregate = aggregate_requests(
            records,
            slo_ttft_ms=run.slo.ttft_ms,
            slo_tpot_ms=run.slo.tpot_ms,
            window_s=window_s,
            attempted=attempted,
            engine_stats=engine_stats,
            store_stats=store_stats,
            warmup_requests=warmup_count,
        )
        metrics.update_store(store_stats)
        metrics.goodput_ratio.set(aggregate["goodput_fraction_of_attempted"])

        wall_s = time.perf_counter() - wall_t0
        finished = datetime.now(UTC)
        mode = getattr(engine, "mode", "unknown")
        reason = getattr(engine, "selection_reason", "")
        result: dict[str, Any] = {
            "run_id": run_id,
            "name": run.name,
            "tier_chain": run.tier_chain or "+".join(active_tiers),
            "capacity_label": run.capacity_label,
            "trace": str(trace_path),
            "trace_sha256": _file_sha256(trace_path),
            "request_rate": rate,
            "context_tokens": run.workload.context_tokens,
            "max_concurrency": max_conc,
            "duration_s": duration,
            "warmup_s": warmup,
            "wall_clock_s": wall_s,
            "engine_mode": mode,
            "engine_selection_reason": reason,
            "requested_tiers": requested_tiers,
            "active_tiers": active_tiers,
            "nvmeof_fell_back": "nvmeof_block" in requested_tiers
            and "nvmeof_block" not in active_tiers,
            **aggregate,
        }

        (outdir / "config.yaml").write_text(
            yaml.safe_dump(run.to_dict(), sort_keys=False)
        )
        (outdir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (outdir / "tier_stats.json").write_text(
            json.dumps(
                {
                    "store": store_stats,
                    "engine": engine_stats,
                    "requested_tiers": requested_tiers,
                    "active_tiers": active_tiers,
                },
                indent=2,
                sort_keys=True,
            )
        )
        manifest = {
            "run_id": run_id,
            "name": run.name,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "wall_clock_s": wall_s,
            "engine_mode": mode,
            "engine_selection_reason": reason,
            "requested_tiers": requested_tiers,
            "active_tiers": active_tiers,
            "trace": str(trace_path),
            "trace_sha256": result["trace_sha256"],
            "trace_requests": len(trace),
            "config": run.to_dict(),
            "calibration": engine_stats.get("calibration"),
            "model_shape": engine_stats.get("model_shape"),
            "env": cached_fingerprint(),
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame.to_parquet(outdir / "requests.parquet", index=False)
        else:
            pd.DataFrame(columns=["run_id", "status"]).to_parquet(
                outdir / "requests.parquet", index=False
            )
        _print_summary(result)
        return result
    finally:
        if engine is not None:
            try:
                await engine.close()
            except Exception:
                log.exception("engine close failed")
        if store is not None:
            try:
                store.close()
            except Exception:
                log.exception("store close failed")
        if target is not None:
            target.stop()


def _print_summary(result: dict[str, Any]) -> None:
    ttft = result["ttft_ms"]
    tpot = result["tpot_ms"]
    table = Table(title=f"run {result['run_id']}")
    table.add_column("metric", justify="left")
    table.add_column("value", justify="right")
    rows = [
        ("engine mode", result["engine_mode"]),
        ("active tiers", " -> ".join(result["active_tiers"])),
        ("requests", f"{result['requests_completed']}/{result['requests_attempted']}"),
        ("TTFT p50/p99 ms", f"{ttft['p50']:.1f} / {ttft['p99']:.1f}"),
        ("TPOT p50/p99 ms", f"{tpot['p50']:.1f} / {tpot['p99']:.1f}"),
        ("goodput req/s", f"{result['goodput_req_s']:.3f}"),
        ("goodput fraction", f"{result['goodput_fraction_of_attempted']:.3f}"),
        ("prefix hit rate", f"{result['prefix_cache_hit_rate']:.3f}"),
        ("restore GiB/s", f"{result['restore_bandwidth_gib_s']:.3f}"),
        ("wall clock s", f"{result['wall_clock_s']:.1f}"),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
