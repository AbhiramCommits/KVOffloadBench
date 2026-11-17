"""Deterministic request-trace generation.

Three traces with genuinely shared prefixes:

* ``longctx_multiturn`` -- 8-turn conversations, one shared system prompt per
  group, logical context growing from 4k to 32k tokens.
* ``agent_react`` -- ReAct tool loops with a long static tool/system preamble
  reused by every step (very high prefix reuse).
* ``shared_prefix_rag`` -- one large retrieved-document prefix per topic with
  many short queries against it.

The ``text`` field is tokenized for real by the emulated engine; the logical
``prompt_tokens`` length is filled with a deterministic, group-seeded token
stream so 32k-token contexts stay representable in a small repository. That
extension is per-group, so different prefix groups never collide while requests
in the same group share an exact prefix.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

WORDS = ["storage", "tier", "cache", "latency", "bandwidth", "queue", "depth", "namespace", "controller", "fabric", "prefix", "token", "block", "tensor", "layer", "head", "dimension", "sequence", "decode", "prefill", "request", "scheduler", "replica", "shard", "cluster", "node", "socket", "kernel", "driver", "filesystem", "direct", "alignment", "sector", "page", "mapping", "eviction", "promotion", "demotion", "residency", "capacity", "throughput", "percentile", "tail", "budget", "target", "region", "zone", "rack", "switch", "link", "credit", "packet", "frame", "protocol", "transport", "handshake", "timeout", "retry", "failure", "recovery", "observe", "metric", "histogram", "counter", "gauge", "trace", "span", "incident", "review", "design", "system", "prompt", "context", "document", "question", "answer", "agent", "tool", "action", "observation", "retrieval", "ranking", "embedding", "vector", "index", "search", "result", "snippet", "citation", "database", "transaction", "journal", "checkpoint", "snapshot", "backup", "restore", "consistency", "compute", "accelerator", "memory", "hierarchy", "offload", "interconnect", "topology", "thermal", "power", "efficiency", "utilization", "saturation", "bottleneck", "attribution", "benchmark"]


def _phrase(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(WORDS) for _ in range(n_words)]
    # Capitalize sentence starts and add punctuation for a realistic tokenizer mix.
    for i in range(0, len(words), 14):
        words[i] = words[i].capitalize()
    return " ".join(words) + "."


def _system_text(rng: random.Random) -> str:
    return (
        "You are an expert assistant for storage-aware LLM serving. "
        + _phrase(rng, 110)
        + " Follow the policy exactly and cite measured evidence. "
        + _phrase(rng, 60)
    )


@dataclass
class TraceSpec:
    name: str
    requests: list[dict[str, Any]]
    block_tokens: int = 16

    def summary(self) -> dict[str, Any]:
        prompt_tokens = [r["prompt_tokens"] for r in self.requests]
        out_tokens = [r["output_tokens"] for r in self.requests]
        groups: dict[str, int] = {}
        for r in self.requests:
            groups[r["prefix_group"]] = max(
                groups.get(r["prefix_group"], 0), int(r["prompt_tokens"])
            )
        total_blocks = sum(math.ceil(t / self.block_tokens) for t in prompt_tokens)
        unique_blocks = sum(math.ceil(t / self.block_tokens) for t in groups.values())
        reuse = 1.0 - unique_blocks / total_blocks if total_blocks else 0.0
        return {
            "trace": self.name,
            "requests": len(self.requests),
            "prefix_groups": len(groups),
            "prompt_tokens_total": sum(prompt_tokens),
            "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "output_tokens_mean": sum(out_tokens) / len(out_tokens),
            "theoretical_prefix_reuse_ratio": reuse,
        }


def generate_longctx_multiturn(seed: int = 2024, conversations: int = 24) -> TraceSpec:
    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    for c in range(conversations):
        group = f"conv-{c:03d}"
        system = _system_text(random.Random(seed * 1000 + c))
        history = ""
        start = c * 2.5
        for turn in range(8):
            target = int(4096 + (32768 - 4096) * turn / 7)
            user = _phrase(rng, 40 + turn * 4)
            text = f"{system}\n\n{history}\nUser: {user}"
            requests.append(
                {
                    "id": f"lc-{c:03d}-t{turn}",
                    "arrival_offset_s": round(start + turn * 2.0, 3),
                    "prefix_group": group,
                    "prompt_tokens": target,
                    "output_tokens": 32 + (turn % 3) * 16,
                    "turn_index": turn,
                    "text": text,
                }
            )
            answer = _phrase(rng, 20 + turn * 2)
            history += f"\nUser: {user}\nAssistant: {answer}"
    return TraceSpec("longctx_multiturn", requests)


def generate_agent_react(seed: int = 2024, agents: int = 40) -> TraceSpec:
    rng = random.Random(seed + 1)
    requests: list[dict[str, Any]] = []
    for a in range(agents):
        group = f"agent-{a:03d}"
        arng = random.Random(seed * 5000 + a)
        tools = (
            "Available tools: search(query), read_file(path), run_sql(statement), "
            "http_get(url), calculator(expression). "
            + _phrase(arng, 260)
        )
        preamble = _system_text(arng) + "\n" + tools
        start = a * 3.0
        steps = 6 + (a % 5)
        for step in range(steps):
            text = (
                f"{preamble}\n\nThought: I need the next fact. "
                f"Action: search({_phrase(rng, 6)})\nObservation: {_phrase(rng, 18)}"
            )
            requests.append(
                {
                    "id": f"react-{a:03d}-s{step}",
                    "arrival_offset_s": round(start + step * 0.75, 3),
                    "prefix_group": group,
                    "prompt_tokens": 4096 + step * 96,
                    "output_tokens": 16 + (step % 3) * 8,
                    "turn_index": step,
                    "text": text,
                }
            )
    return TraceSpec("agent_react", requests)


def generate_shared_prefix_rag(seed: int = 2024, topics: int = 12, queries: int = 10) -> TraceSpec:
    rng = random.Random(seed + 2)
    requests: list[dict[str, Any]] = []
    for t in range(topics):
        group = f"rag-{t:03d}"
        doc_rng = random.Random(seed * 9000 + t)
        doc = (
            "Retrieved document (shared context): "
            + _phrase(doc_rng, 380)
            + " "
            + _phrase(doc_rng, 380)
        )
        for q in range(queries):
            question = _phrase(rng, 18)
            requests.append(
                {
                    "id": f"rag-{t:03d}-q{q}",
                    "arrival_offset_s": round(t * 2.0 + q * 0.9, 3),
                    "prefix_group": group,
                    "prompt_tokens": 16384,
                    "output_tokens": 64,
                    "turn_index": q,
                    "text": f"{doc}\n\nQuestion: {question}\nAnswer:",
                }
            )
    return TraceSpec("shared_prefix_rag", requests)


def write_trace(spec: TraceSpec, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for req in spec.requests:
            fh.write(json.dumps(req, sort_keys=True) + "\n")
    return path


def generate_all(out_dir: str | Path = "traces/", seed: int = 2024) -> list[dict[str, Any]]:
    out = Path(out_dir)
    specs = [
        generate_longctx_multiturn(seed),
        generate_agent_react(seed),
        generate_shared_prefix_rag(seed),
    ]
    summaries = []
    for spec in specs:
        path = write_trace(spec, out)
        summary = spec.summary()
        summary["path"] = str(path)
        summaries.append(summary)
        console.print(f"[green]wrote {len(spec.requests)} requests to {path}[/green]")
    table = Table(title="Generated traces")
    for col in (
        "trace",
        "requests",
        "groups",
        "mean prompt",
        "max prompt",
        "mean out",
        "theoretical reuse",
    ):
        table.add_column(col, justify="left" if col == "trace" else "right")
    for s in summaries:
        table.add_row(
            s["trace"],
            str(s["requests"]),
            str(s["prefix_groups"]),
            f"{s['prompt_tokens_mean']:.0f}",
            str(s["prompt_tokens_max"]),
            f"{s['output_tokens_mean']:.1f}",
            f"{s['theoretical_prefix_reuse_ratio'] * 100:.1f}%",
        )
    console.print(table)
    return summaries


def iter_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(out_dir: str | Path = "traces/", seed: int = 2024) -> int:
    generate_all(out_dir, seed)
    return 0
