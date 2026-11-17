"""Derive the storage requirements table from measured results.

For each context length and target request rate the table answers:

* how many KV bytes a full prefix restore moves;
* the read bandwidth needed for one request to meet the TTFT SLO;
* the sustained bandwidth implied by that rate;
* the IOPS at the 128 KiB storage block size;
* the p99 per-block latency budget.

It is computed twice: once for the declared local shape actually exercised by
the sweep, and once scaled to a 4x H200 Llama-3.3-70B deployment (TP=4). The
scaling assumptions are stated inline in the generated markdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS = Path("results")
TABLE_DIR = RESULTS / "tables"
STORAGE_BLOCK_BYTES = 128 * 1024
CONTEXTS = (4096, 16384, 32768)
RATES = (1.0, 2.0, 4.0, 8.0)

# Declared llama-3.3-70B-like shape: 80 layers, 8 KV heads, head_dim 128, fp16.
LLAMA70B_BYTES_PER_TOKEN = 80 * 2 * 8 * 128 * 2
TP4 = 4


def _calibration(root: Path) -> dict[str, Any]:
    path = root / "calibration" / "engine_calibration.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def required_read_bandwidth_gib_s(kv_bytes: int, budget_s: float) -> float:
    if budget_s <= 0:
        raise ValueError("budget_s must be positive")
    return kv_bytes / budget_s / (1024**3)


def required_iops(kv_bytes: int, block_bytes: int, budget_s: float) -> float:
    if budget_s <= 0 or block_bytes <= 0:
        raise ValueError("budget_s and block_bytes must be positive")
    blocks = math.ceil(kv_bytes / block_bytes)
    return blocks / budget_s


def p99_block_latency_budget_us(budget_ms: float, blocks: int) -> float:
    if blocks <= 0:
        return 0.0
    return budget_ms * 1000.0 / blocks


def _local_shape(root: Path) -> tuple[dict[str, Any], int]:
    for manifest in sorted((root / "raw").glob("*/manifest.json")):
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        shape = data.get("model_shape")
        if shape and shape.get("bytes_per_token"):
            return shape, int(shape["bytes_per_token"])
    return {}, 6144


def _measured_restore(root: Path) -> pd.DataFrame:
    index_path = root / "index.csv"
    if not index_path.is_file():
        return pd.DataFrame()
    index = pd.read_csv(index_path)
    if index.empty:
        return index
    index = index.copy()
    index["best_restore"] = index[["restore_service_gib_s", "restore_bandwidth_gib_s"]].max(axis=1)
    return index


def build_requirements(results_root: str | Path = RESULTS) -> pd.DataFrame:
    root = Path(results_root)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.csv"
    index = pd.read_csv(index_path) if index_path.is_file() else pd.DataFrame()
    calibration = _calibration(root)
    prefill_ms_per_token = float(calibration.get("prefill_ms_per_token", 0.0))
    shape, local_bpt = _local_shape(root)
    slo_ttft = 4000.0
    if not index.empty and "ttft_p50_ms" in index:
        # SLO is constant across the sweep; read it from any manifest config.
        for manifest in sorted((root / "raw").glob("*/manifest.json")):
            try:
                cfg = json.loads(manifest.read_text()).get("config", {})
                slo_ttft = float(cfg.get("slo", {}).get("ttft_ms", slo_ttft))
                break
            except (OSError, json.JSONDecodeError):
                continue

    measured = _measured_restore(root)
    rows: list[dict[str, Any]] = []

    def add_scope(
        scope: str,
        bytes_per_token: int,
        *,
        per_rank: bool,
        full_budget: bool,
        note: str,
    ) -> None:
        for context in CONTEXTS:
            kv_bytes = context * bytes_per_token
            blocks = math.ceil(kv_bytes / STORAGE_BLOCK_BYTES)
            compute_ms = context * prefill_ms_per_token if scope == "local" else 0.0
            io_budget_ms = max(0.0, slo_ttft - compute_ms) if scope == "local" else slo_ttft
            for rate in RATES:
                req_bw = (
                    required_read_bandwidth_gib_s(kv_bytes, slo_ttft / 1000.0)
                    if full_budget
                    else None
                )
                req_bw_after_compute = (
                    required_read_bandwidth_gib_s(kv_bytes, io_budget_ms / 1000.0)
                    if io_budget_ms > 0 and scope == "local"
                    else None
                )
                sustained = kv_bytes * rate / (1024**3)
                iops = required_iops(kv_bytes, STORAGE_BLOCK_BYTES, slo_ttft / 1000.0)
                iops_after = (
                    required_iops(kv_bytes, STORAGE_BLOCK_BYTES, io_budget_ms / 1000.0)
                    if io_budget_ms > 0
                    else None
                )
                p99_us = p99_block_latency_budget_us(slo_ttft, blocks) if blocks else 0.0
                p99_after = (
                    p99_block_latency_budget_us(io_budget_ms, blocks)
                    if io_budget_ms > 0
                    else 0.0
                )
                best_restore = 0.0
                best_tier = ""
                if not measured.empty and scope == "local":
                    subset = measured[
                        (measured["context_tokens"] == context)
                        & (measured["request_rate"] == rate)
                    ]
                    if not subset.empty:
                        row = subset.iloc[int(subset["best_restore"].to_numpy().argmax())]
                        best_restore = float(row["best_restore"])
                        best_tier = str(row["tier_chain"])
                required = req_bw if scope != "local" else req_bw_after_compute
                rows.append(
                    {
                        "scope": scope,
                        "context_tokens": context,
                        "request_rate": rate,
                        "kv_bytes_per_request": kv_bytes,
                        "kv_gib_per_request": kv_bytes / (1024**3),
                        "kv_mib_per_request": kv_bytes / (1024**2),
                        "storage_blocks_128k": blocks,
                        "bytes_per_token": bytes_per_token,
                        "declared_layers": shape.get("layers") if scope == "local" else 80,
                        "declared_kv_heads": shape.get("kv_heads") if scope == "local" else 8,
                        "declared_head_dim": shape.get("head_dim") if scope == "local" else 128,
                        "tensor_parallel": TP4 if scope != "local" else 1,
                        "prefill_compute_ms_local": compute_ms if scope == "local" else None,
                        "ttft_budget_ms": slo_ttft,
                        "io_budget_after_local_compute_ms": io_budget_ms if scope == "local" else None,
                        "required_read_gib_s_per_request": req_bw,
                        "required_read_gib_s_after_local_compute": req_bw_after_compute,
                        "sustained_read_gib_s_at_rate": sustained,
                        "required_read_gib_s_per_rank": req_bw / (TP4 if scope != "local" else 1)
                        if req_bw is not None
                        else None,
                        "required_iops_full_budget": iops,
                        "required_iops_after_local_compute": iops_after,
                        "p99_block_latency_budget_us": p99_us,
                        "p99_block_latency_budget_us_after_compute": p99_after
                        if scope == "local"
                        else None,
                        "p99_block_latency_budget_us_qd8": p99_us * 8,
                        "measured_best_restore_gib_s": best_restore if scope == "local" else None,
                        "measured_best_chain": best_tier if scope == "local" else "",
                        "meets_bandwidth_requirement": (
                            best_restore >= required
                            if scope == "local"
                            and required is not None
                            and required > 0
                            and best_restore > 0
                            else None
                        ),
                        "assumption": note,
                        "per_rank_scope": per_rank,
                    }
                )

    add_scope(
        "local",
        local_bpt,
        per_rank=False,
        full_budget=False,
        note=(
            "Declared local shape exercised by this build; compute budget from the "
            "calibrated gpt2 prefill on this host. Restore is a full prefix miss."
        ),
    )
    add_scope(
        "llama70b_tp4",
        LLAMA70B_BYTES_PER_TOKEN,
        per_rank=True,
        full_budget=True,
        note=(
            "Llama-3.3-70B-like KV (327,680 B/token total, 81,920 B/token per rank at TP=4). "
            "Uses the full TTFT budget for I/O (conservative); per-rank bandwidth is aggregate/4. "
            "Block granularity stays 128 KiB. No H200 timing was available, so no compute "
            "subtraction is claimed."
        ),
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "storage_requirements.csv", index=False)
    _write_markdown(frame, shape, calibration, table_dir)
    return frame


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(
    frame: pd.DataFrame,
    shape: dict[str, Any],
    calibration: dict[str, Any],
    table_dir: Path,
) -> None:
    local = frame[frame["scope"] == "local"]
    scale = frame[frame["scope"] == "llama70b_tp4"]
    lines = [
        "# Storage requirements for KV offload",
        "",
        "Derived by `kvoffloadbench analyze` from the measured sweep "
        "(`results/index.csv`) and the Step-2 tier microbenchmark.",
        "",
        "## Formulas",
        "",
        "```",
        "restore_bytes_per_request = context_tokens * bytes_per_token",
        "storage_blocks             = ceil(restore_bytes_per_request / 128 KiB)",
        "required_read_bandwidth     = restore_bytes_per_request / TTFT_budget",
        "sustained_read_bandwidth    = restore_bytes_per_request * request_rate",
        "required_IOPS               = storage_blocks / TTFT_budget",
        "p99_block_latency_budget    = TTFT_budget / storage_blocks",
        "```",
        "",
        "- TTFT budget: **4000 ms** (the SLO used by the sweep).",
        f"- Local declared shape: {shape.get('layers', 12)} layers, "
        f"{shape.get('kv_heads', 2)} KV heads, head_dim {shape.get('head_dim', 64)}, "
        f"{shape.get('dtype', 'float16')} -> **{shape.get('bytes_per_token', 6144):,} B/token** "
        f"({shape.get('bytes_per_block', 98304):,} B per 16-token KV block).",
        "- Local compute budget uses the measured prefill "
        f"**{calibration.get('prefill_ms_per_token', 0):.4f} ms/token** on "
        f"`{calibration.get('device', 'n/a')}` from `results/calibration/engine_calibration.json`.",
        "- A full prefix miss is assumed; prefix hits reduce the bytes actually restored, which is",
        "  why the measured restore requirement can be lower than this worst case.",
        "- `p99 ... qd8` relaxes the per-block budget by pipelining 8 in-flight block reads.",
        "",
        "## Local shape (as measured in this build)",
        "",
        "| ctx | rate | KV MiB/req | blocks | compute ms | I/O budget ms | req GiB/s | sustained GiB/s | req IOPS | p99 budget us | best measured GiB/s | chain | meets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in local.iterrows():
        lines.append(
            f"| {int(r['context_tokens'])} | {r['request_rate']:g} | "
            f"{r['kv_mib_per_request']:.1f} | {int(r['storage_blocks_128k'])} | "
            f"{_fmt(r['prefill_compute_ms_local'], 0)} | "
            f"{_fmt(r['io_budget_after_local_compute_ms'], 0)} | "
            f"{_fmt(r['required_read_gib_s_after_local_compute'], 3)} | "
            f"{_fmt(r['sustained_read_gib_s_at_rate'], 2)} | "
            f"{_fmt(r['required_iops_after_local_compute'], 0)} | "
            f"{_fmt(r['p99_block_latency_budget_us_after_compute'], 0)} | "
            f"{r['measured_best_restore_gib_s']:.3f} | {r['measured_best_chain']} | "
            f"{_fmt(r['meets_bandwidth_requirement'])} |"
        )
    lines += [
        "",
        "Where the calibrated local compute alone exceeds the 4 s TTFT budget (32k context),",
        "the I/O budget is zero and the requirement columns show `-`: that is an honest statement",
        "that this host's calibrated prefill cannot meet the SLO even with infinitely fast storage.",
        "",
        "## Scaled to Llama-3.3-70B on 4x H200 (TP=4)",
        "",
        "| ctx | rate | KV GiB/req | per-rank GiB/req | blocks | req GiB/s aggregate | per-rank GiB/s | sustained GiB/s | req IOPS | p99 budget us (@QD1 / @QD8) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in scale.iterrows():
        lines.append(
            f"| {int(r['context_tokens'])} | {r['request_rate']:g} | "
            f"{r['kv_gib_per_request']:.2f} | {r['kv_gib_per_request'] / TP4:.2f} | "
            f"{int(r['storage_blocks_128k'])} | "
            f"{_fmt(r['required_read_gib_s_per_request'], 2)} | "
            f"{_fmt(r['required_read_gib_s_per_rank'], 3)} | "
            f"{_fmt(r['sustained_read_gib_s_at_rate'], 1)} | "
            f"{_fmt(r['required_iops_full_budget'], 0)} | "
            f"{_fmt(r['p99_block_latency_budget_us'], 1)} / "
            f"{_fmt(r['p99_block_latency_budget_us_qd8'], 1)} |"
        )
    lines += [
        "",
        "### Scaling assumptions (stated inline)",
        "",
        "- The 70B KV shape is 80 layers x 2 (K and V) x 8 KV heads x 128 head_dim x 2 bytes",
        "  = **327,680 B/token total**; tensor parallel 4 splits it evenly, so each rank stores",
        "  81,920 B/token and must move one quarter of the bytes.",
        "- Restore is assumed to be a full prefix miss, and the full 4 s TTFT budget is allocated",
        "  to storage. On real H200 hardware prefill consumes part of that budget; subtract the",
        "  measured prefill for the target model before sizing the SSD.",
        "- `sustained` is the worst-case aggregate for restoring every request with no prefix",
        "  reuse: bytes/request x request_rate. Prefix caching and KV reuse are what make the",
        "  deployment feasible; the measured hit rates in `results/index.csv` quantify the gap.",
        "- IOPS use 128 KiB storage blocks. A 70B 16-token KV block is 5 MiB, i.e. 40 storage",
        "  blocks per KV block. Larger I/O requests (or NVMe-oF with larger MDTS) reduce the",
        "  IOPS requirement proportionally.",
        "- NVMe-oF adds a network hop: the p99 budget is per 128 KiB read at the host, so the",
        "  target's p99 plus the fabric RTT must fit inside it.",
    ]
    (table_dir / "storage_requirements.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    build_requirements()
