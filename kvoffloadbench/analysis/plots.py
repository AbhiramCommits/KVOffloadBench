"""Figures generated deterministically from committed results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
COLORS = {
    "dram_only": "#1f77b4",
    "dram_nvme": "#2ca02c",
    "dram_nvme_remote_tcp": "#d62728",
}
MARKERS = {"dram_only": "o", "dram_nvme": "s", "dram_nvme_remote_tcp": "^"}
LABELS = {
    "dram_only": "DRAM only",
    "dram_nvme": "DRAM + local NVMe",
    "dram_nvme_remote_tcp": "DRAM + NVMe + remote NVMe-oF (TCP)",
}


def _index(results_root: Path) -> pd.DataFrame:
    path = results_root / "index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the sweep first")
    df = pd.read_csv(path)
    df = df[df["name"] == "main"]
    if df.empty:
        raise ValueError("results/index.csv has no sweep rows (name='main')")
    return df


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _save(fig: Any, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if path.stat().st_size < 5000:
        raise RuntimeError(f"figure {path} looks empty ({path.stat().st_size} bytes)")
    return path


def _curve_frame(df: pd.DataFrame, trace: str = "longctx_multiturn") -> pd.DataFrame:
    sub = df[
        df["trace"].astype(str).str.startswith(trace)
        & (df["context_tokens"] == 16384)
        & (df["capacity_label"] == "medium")
    ]
    if sub.empty:
        sub = df[(df["context_tokens"] == df["context_tokens"].max())]
    return sub


def plot_ttft_vs_rate(df: pd.DataFrame) -> Path:
    sub = _curve_frame(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for chain_raw, group in sub.groupby("tier_chain"):
        chain = str(chain_raw)
        group = group.sort_values("request_rate")
        ax.plot(
            group["request_rate"],
            group["ttft_p50_ms"],
            marker=MARKERS.get(chain, "o"),
            color=COLORS.get(chain),
            label=f"{LABELS.get(chain, chain)} (p50)",
            linewidth=2,
        )
        ax.plot(
            group["request_rate"],
            group["ttft_p99_ms"],
            linestyle="--",
            marker=MARKERS.get(chain, "o"),
            color=COLORS.get(chain),
            alpha=0.6,
            label=f"{LABELS.get(chain, chain)} (p99)",
        )
    ax.axhline(4000, color="black", linestyle=":", linewidth=1, label="TTFT SLO 4 s")
    _style(
        ax,
        "TTFT vs request rate (longctx, 16k context, medium cache)",
        "request rate (req/s)",
        "TTFT (ms)",
    )
    ax.legend(fontsize=8)
    return _save(fig, "ttft_vs_rate_by_tier.png")


def plot_goodput_vs_rate(df: pd.DataFrame) -> Path:
    sub = _curve_frame(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for chain_raw, group in sub.groupby("tier_chain"):
        chain = str(chain_raw)
        group = group.sort_values("request_rate")
        ax.plot(
            group["request_rate"],
            group["goodput_req_s"],
            marker=MARKERS.get(chain, "o"),
            color=COLORS.get(chain),
            label=LABELS.get(chain, chain),
            linewidth=2,
        )
    ax.plot([1, 8], [1, 8], linestyle=":", color="gray", label="offered load")
    _style(
        ax,
        "Goodput vs request rate (longctx, 16k context, medium cache)",
        "request rate (req/s)",
        "SLO-meeting requests/s",
    )
    ax.legend(fontsize=9)
    return _save(fig, "goodput_vs_rate_by_tier.png")


def plot_hit_rate_vs_cache(df: pd.DataFrame) -> Path:
    sub = df[
        df["trace"].astype(str).str.startswith("longctx")
        & (df["context_tokens"] == 16384)
        & (df["request_rate"] == 4)
    ]
    if sub.empty:
        sub = df[df["request_rate"] == 4]
    fig, ax = plt.subplots(figsize=(8, 5))
    for chain_raw, group in sub.groupby("tier_chain"):
        chain = str(chain_raw)
        group = group.copy()
        group["cap_mib"] = pd.to_numeric(
            group["capacity_label"].map({"small": 64, "medium": 512, "large": 2048}),
            errors="coerce",
        )
        group = group.dropna(subset=["cap_mib"]).sort_values("cap_mib")
        if group.empty:
            continue
        ax.plot(
            group["cap_mib"],
            group["prefix_cache_hit_rate"],
            marker=MARKERS.get(chain, "o"),
            color=COLORS.get(chain),
            label=LABELS.get(chain, chain),
            linewidth=2,
        )
    ax.set_xscale("log", base=2)
    _style(
        ax,
        "Prefix-cache hit rate vs hot-cache capacity (longctx, 16k, 4 req/s)",
        "DRAM capacity (MiB, log scale)",
        "prefix-cache hit rate",
    )
    ax.legend(fontsize=9)
    return _save(fig, "hit_rate_vs_cache_size.png")


def plot_ttft_cdf(df: pd.DataFrame, results_root: Path) -> Path:
    sub = df[
        df["trace"].astype(str).str.startswith("longctx")
        & (df["context_tokens"] == 16384)
        & (df["request_rate"] == 4)
        & (df["capacity_label"] == "medium")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for chain_raw, group in sub.groupby("tier_chain"):
        chain = str(chain_raw)
        run_id = str(group.iloc[0]["run_id"])
        parquet = results_root / "raw" / run_id / "requests.parquet"
        if not parquet.is_file():
            continue
        req = pd.read_parquet(parquet)
        req = req[(~req["warmup"].astype(bool)) & (req["status"] == "completed")]
        if req.empty:
            continue
        values = sorted(req["ttft_ms"].astype(float))
        ys = [(i + 1) / len(values) for i in range(len(values))]
        ax.plot(
            values,
            ys,
            color=COLORS.get(chain),
            label=LABELS.get(chain, chain),
            linewidth=2,
        )
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "no per-request parquet data found", ha="center", va="center")
    ax.axvline(4000, color="black", linestyle=":", linewidth=1, label="TTFT SLO 4 s")
    _style(
        ax,
        "TTFT CDF by tier chain (longctx, 16k, 4 req/s)",
        "TTFT (ms)",
        "fraction of requests",
    )
    ax.legend(fontsize=9)
    return _save(fig, "ttft_cdf_by_tier.png")


def plot_restore_bandwidth_vs_context(df: pd.DataFrame) -> Path:
    sub = df[
        df["trace"].astype(str).str.startswith("longctx")
        & (df["request_rate"] == 4)
        & (df["capacity_label"] == "medium")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for chain_raw, group in sub.groupby("tier_chain"):
        chain = str(chain_raw)
        group = group.sort_values("context_tokens")
        ax.plot(
            group["context_tokens"],
            group["restore_service_gib_s"],
            marker=MARKERS.get(chain, "o"),
            color=COLORS.get(chain),
            label=LABELS.get(chain, chain),
            linewidth=2,
        )
    _style(
        ax,
        "Restore service bandwidth vs context length (longctx, 4 req/s)",
        "context tokens",
        "restore bandwidth (GiB/s)",
    )
    ax.legend(fontsize=9)
    return _save(fig, "restore_bandwidth_vs_context.png")


def plot_tier_microbench(results_root: Path) -> Path:
    path = results_root / "tiers" / "microbench.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    df = pd.read_csv(path)
    sub = df[(df["op"] == "read") & (df["pattern"] == "sequential") & (df["depth"] == 8)]
    if sub.empty:
        sub = df[df["op"] == "read"]
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"dram": "#1f77b4", "nvme_local": "#2ca02c", "remote_tcp": "#d62728", "nvmeof_block": "#9467bd"}
    for tier_raw, group in sub.groupby("tier"):
        tier = str(tier_raw)
        group = group.sort_values("block_kib")
        ax.plot(
            group["block_kib"],
            group["p99_us"],
            marker="o",
            color=palette.get(str(tier)),
            label=str(tier),
            linewidth=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(
        ax,
        "Tier read p99 latency vs block size (sequential, QD=8)",
        "block size (KiB, log scale)",
        "p99 latency (us, log scale)",
    )
    ax.legend(fontsize=9)
    return _save(fig, "tier_microbench_latency.png")


def generate_all(results_root: str | Path = RESULTS) -> list[Path]:
    root = Path(results_root)
    df = _index(root)
    paths = [
        plot_ttft_vs_rate(df),
        plot_goodput_vs_rate(df),
        plot_hit_rate_vs_cache(df),
        plot_ttft_cdf(df, root),
        plot_restore_bandwidth_vs_context(df),
        plot_tier_microbench(root),
    ]
    return paths


if __name__ == "__main__":  # pragma: no cover
    for path in generate_all():
        print(path)
