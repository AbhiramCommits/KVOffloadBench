"""Tier microbenchmark: sequential/random, read/write, block sizes, queue depths."""

from __future__ import annotations

import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from kvoffloadbench.config import ConfigError, TierConfig
from kvoffloadbench.target.launcher import TargetHandle, start_target
from kvoffloadbench.tiers.base import KVTier, TierError, TierUnavailable
from kvoffloadbench.tiers.tiered_store import build_tier

log = logging.getLogger(__name__)
console = Console()

DEFAULT_OUT = "results/tiers/microbench.csv"
CSV_FIELDS = [
    "tier",
    "op",
    "block_kib",
    "depth",
    "gib_s",
    "iops",
    "p50_us",
    "p95_us",
    "p99_us",
    "direct_io",
    "mode",
    "pattern",
    "io_mode",
    "ops",
    "seconds",
    "bytes",
]


@dataclass
class TierSpec:
    name: str
    kind: str
    capacity_mib: int
    block_size_kib: int
    path: str | None
    endpoint: str | None
    target_path: str | None
    target_capacity_mib: int
    optional: bool
    working_set_mib: int | None
    extra: dict[str, Any]


@dataclass
class MicroConfig:
    duration_s: float
    working_set_mib: int
    max_bytes_per_point: int
    seed: int
    block_kib: list[int]
    queue_depth: list[int]
    patterns: list[str]
    ops: list[str]
    tiers: list[TierSpec]


def _load_micro_config(path: str | Path) -> MicroConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"microbench config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ConfigError("microbench config must be a mapping")
    allowed = {
        "duration_s",
        "working_set_mib",
        "max_bytes_per_point",
        "seed",
        "block_kib",
        "queue_depth",
        "patterns",
        "ops",
        "tiers",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown microbench fields: {unknown}")
    tiers_raw = raw.get("tiers")
    if not isinstance(tiers_raw, list) or not tiers_raw:
        raise ConfigError("microbench config needs a non-empty 'tiers' list")
    tiers: list[TierSpec] = []
    for i, t in enumerate(tiers_raw):
        if not isinstance(t, dict) or "kind" not in t or "name" not in t:
            raise ConfigError(f"tiers[{i}] must contain 'name' and 'kind'")
        kind = str(t["kind"])
        if kind not in ("dram", "nvme", "remote_tcp", "nvmeof_block"):
            raise ConfigError(f"tiers[{i}].kind={kind!r} is not supported by the microbench")
        tiers.append(
            TierSpec(
                name=str(t["name"]),
                kind=kind,
                capacity_mib=int(t.get("capacity_mib", 2048)),
                block_size_kib=int(t.get("block_size_kib", 64)),
                path=t.get("path"),
                endpoint=t.get("endpoint"),
                target_path=t.get("target_path"),
                target_capacity_mib=int(t.get("target_capacity_mib", 1024)),
                optional=bool(t.get("optional", False)),
                working_set_mib=t.get("working_set_mib"),
                extra=dict(t.get("extra") or {}),
            )
        )
    block_kib = [int(v) for v in raw.get("block_kib", [4, 16, 64, 256, 1024])]
    depths = [int(v) for v in raw.get("queue_depth", [1, 8, 32])]
    patterns = [str(v) for v in raw.get("patterns", ["sequential", "random"])]
    ops = [str(v) for v in raw.get("ops", ["write", "read"])]
    for v in block_kib:
        if v < 4 or (v & (v - 1)):
            raise ConfigError(f"block_kib value {v} must be a power of two >= 4")
    for d in depths:
        if d < 1:
            raise ConfigError(f"queue_depth value {d} must be >= 1")
    for pat in patterns:
        if pat not in ("sequential", "random"):
            raise ConfigError(f"unknown pattern {pat!r}")
    for op in ops:
        if op not in ("read", "write"):
            raise ConfigError(f"unknown op {op!r}")
    return MicroConfig(
        duration_s=float(raw.get("duration_s", 0.5)),
        working_set_mib=int(raw.get("working_set_mib", 64)),
        max_bytes_per_point=int(raw.get("max_bytes_per_point", 256 * 1024 * 1024)),
        seed=int(raw.get("seed", 7)),
        block_kib=block_kib,
        queue_depth=depths,
        patterns=patterns,
        ops=ops,
        tiers=tiers,
    )


def _make_ids(prefix: str, count: int, seed: int, randomize: bool) -> list[str]:
    ids = [f"{prefix}-{i:08d}" for i in range(count)]
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(ids)
    return ids


def _run_point(
    tier: KVTier,
    *,
    op: str,
    pattern: str,
    block_bytes: int,
    depth: int,
    working_set_mib: int,
    duration_s: float,
    max_bytes: int,
    payload: bytes,
    seed: int,
) -> dict[str, Any] | None:
    ws_bytes = working_set_mib * 1024 * 1024
    nblocks = max(1, ws_bytes // block_bytes)
    ids = _make_ids(f"{tier.name}-{pattern}", nblocks, seed + block_bytes + depth, pattern == "random")
    if op == "read":
        for i in range(0, len(ids), depth):
            tier.put_many([(bid, payload) for bid in ids[i : i + depth]])
    tier.reset_stats()
    max_ops = max(depth, min(nblocks * 4, max_bytes // block_bytes))
    ops = 0
    batches = 0
    t0 = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= duration_s or ops >= max_ops:
            break
        start = (ops % len(ids))
        batch = [ids[(start + j) % len(ids)] for j in range(min(depth, max_ops - ops))]
        if op == "read":
            tier.get_many(batch)
        else:
            tier.put_many([(bid, payload) for bid in batch])
        ops += len(batch)
        batches += 1
    elapsed = time.perf_counter() - t0
    if elapsed <= 0 or batches == 0:
        return None
    stats = tier.stats()
    lat = stats.read_latency.summary_us() if op == "read" else stats.write_latency.summary_us()
    total_bytes = ops * block_bytes
    return {
        "tier": tier.name,
        "op": op,
        "block_kib": block_bytes // 1024,
        "depth": depth,
        "gib_s": total_bytes / elapsed / (1024**3),
        "iops": ops / elapsed,
        "p50_us": lat["p50"],
        "p95_us": lat["p95"],
        "p99_us": lat["p99"],
        "direct_io": bool(tier.io_mode in ("odirect", "fnocache", "nvmeof-odirect")),
        "mode": "real",
        "pattern": pattern,
        "io_mode": tier.io_mode,
        "ops": ops,
        "seconds": elapsed,
        "bytes": total_bytes,
    }


def run_microbench(
    config_path: str | Path,
    out_csv: str | Path = DEFAULT_OUT,
) -> list[dict[str, Any]]:
    cfg = _load_micro_config(config_path)
    target: TargetHandle | None = None
    rows: list[dict[str, Any]] = []
    remote_spec = next((t for t in cfg.tiers if t.kind == "remote_tcp"), None)
    try:
        if remote_spec is not None:
            endpoint = remote_spec.endpoint or "127.0.0.1:4420"
            target = start_target(
                endpoint,
                remote_spec.target_path or "results/cache/microbench/target.bin",
                remote_spec.target_capacity_mib,
            )
        for spec in cfg.tiers:
            for block_kib in cfg.block_kib:
                block_bytes = block_kib * 1024
                ws_mib = spec.working_set_mib or cfg.working_set_mib
                if (spec.kind in ("nvme", "nvmeof_block")) and spec.capacity_mib * 1024 * 1024 < ws_mib * 1024 * 1024 * 2:
                    ws_mib = max(8, spec.capacity_mib // 4)
                path = spec.path.format(block_kib=block_kib) if spec.path else None
                tier_cfg = TierConfig(
                    kind=spec.kind,
                    capacity_mib=spec.capacity_mib,
                    path=path,
                    endpoint=spec.endpoint,
                    block_size_kib=block_kib,
                    queue_depth=1,
                    direct_io=bool(spec.extra.get("direct_io", True)),
                    extra=spec.extra,
                )
                console.print(
                    f"[cyan]{spec.name}[/cyan] block={block_kib} KiB "
                    f"working_set={ws_mib} MiB"
                )
                rng = random.Random(cfg.seed + block_kib)
                payload = rng.randbytes(block_bytes)
                try:
                    first = build_tier(tier_cfg, name=spec.name)
                except TierUnavailable as exc:
                    console.print(f"[yellow]skip {spec.name}: {exc}[/yellow]")
                    break
                try:
                    for depth in cfg.queue_depth:
                        if (
                            depth > 1
                            and spec.kind == "dram"
                            and depth != max(cfg.queue_depth)
                        ):
                            # DRAM has no queue to fill; measure the batched path once.
                            continue
                        tier_cfg_depth = TierConfig(
                            kind=spec.kind,
                            capacity_mib=spec.capacity_mib,
                            path=path,
                            endpoint=spec.endpoint,
                            block_size_kib=block_kib,
                            queue_depth=depth,
                            direct_io=bool(spec.extra.get("direct_io", True)),
                            extra=spec.extra,
                        )
                        tier = (
                            first
                            if depth == 1
                            else build_tier(tier_cfg_depth, name=spec.name)
                        )
                        try:
                            for pattern in cfg.patterns:
                                for op in cfg.ops:
                                    row = _run_point(
                                        tier,
                                        op=op,
                                        pattern=pattern,
                                        block_bytes=block_bytes,
                                        depth=depth,
                                        working_set_mib=ws_mib,
                                        duration_s=cfg.duration_s,
                                        max_bytes=cfg.max_bytes_per_point,
                                        payload=payload,
                                        seed=cfg.seed,
                                    )
                                    if row is not None:
                                        row["tier"] = spec.name
                                        rows.append(row)
                                        console.print(
                                            f"  {op:5s} {pattern:10s} d={depth:<3d} "
                                            f"{row['gib_s']:7.2f} GiB/s "
                                            f"{row['iops']:9.0f} IOPS "
                                            f"p50={row['p50_us']:7.1f}us "
                                            f"p99={row['p99_us']:8.1f}us"
                                        )
                        finally:
                            if tier is not first:
                                tier.close()
                finally:
                    first.close()
    finally:
        if target is not None:
            target.stop()

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (str(r["tier"]), int(r["block_kib"]), int(r["depth"]), str(r["op"]), str(r["pattern"])))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})
    console.print(f"[green]wrote {len(rows)} rows to {out}[/green]")
    _print_table(rows)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Tier microbenchmark", show_lines=False)
    for col in ("tier", "op", "pattern", "block", "depth", "GiB/s", "IOPS", "p50 us", "p99 us", "I/O mode"):
        table.add_column(col, justify="right" if col not in ("tier", "op", "pattern", "I/O mode") else "left")
    for r in rows:
        table.add_row(
            str(r["tier"]),
            str(r["op"]),
            str(r["pattern"]),
            f"{r['block_kib']}K",
            str(r["depth"]),
            f"{r['gib_s']:.2f}",
            f"{r['iops']:.0f}",
            f"{r['p50_us']:.1f}",
            f"{r['p99_us']:.1f}",
            str(r["io_mode"]),
        )
    console.print(table)


def main(config: str | Path, out: str | Path = DEFAULT_OUT) -> int:
    try:
        run_microbench(config, out)
    except TierError as exc:
        console.print(f"[red]tier error: {exc}[/red]")
        return 1
    return 0
