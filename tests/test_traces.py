"""Trace determinism, prefix collisions, and declared token bounds."""

from __future__ import annotations

import json
from pathlib import Path

from kvoffloadbench.engines.emulated import chained_block_ids
from kvoffloadbench.traces.generate import generate_all, iter_trace


def _common_prefix_len(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def _read(tmp_path: Path) -> dict[str, list[dict]]:
    return {
        name: list(iter_trace(tmp_path / f"{name}.jsonl"))
        for name in ("longctx_multiturn", "agent_react", "shared_prefix_rag")
    }


def test_generation_is_deterministic(tmp_path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    generate_all(first, seed=2024)
    generate_all(second, seed=2024)
    for name in ("longctx_multiturn", "agent_react", "shared_prefix_rag"):
        a = (first / f"{name}.jsonl").read_bytes()
        b = (second / f"{name}.jsonl").read_bytes()
        assert a == b, f"{name} is not deterministic"
    # A different seed changes the text.
    third = tmp_path / "three"
    generate_all(third, seed=99)
    assert (first / "agent_react.jsonl").read_bytes() != (third / "agent_react.jsonl").read_bytes()


def test_prefix_groups_share_text_prefix(tmp_path) -> None:
    generate_all(tmp_path)
    rows = _read(tmp_path)["longctx_multiturn"]
    by_group: dict[str, list[str]] = {}
    for row in rows:
        by_group.setdefault(row["prefix_group"], []).append(row["text"])
    for texts in by_group.values():
        assert len(texts) == 8
        common_len = min(
            _common_prefix_len(texts[0], text) for text in texts[1:]
        )
        assert common_len > 200, "turns in a conversation must share a system prefix"


def test_chained_block_ids_collide_only_on_true_prefix() -> None:
    prefix = list(range(1, 33))
    a = chained_block_ids([*prefix, 100], 16)
    b = chained_block_ids([*prefix, 200], 16)
    assert a[:2] == b[:2]
    assert a[2] != b[2]
    other = chained_block_ids([999, *prefix, 100], 16)
    assert other[0] != a[0]
    # Same token content at a different position must not collide.
    shifted = chained_block_ids([0, *prefix, 100], 16)
    assert shifted != a


def test_declared_bounds(tmp_path) -> None:
    generate_all(tmp_path)
    data = _read(tmp_path)
    lc = data["longctx_multiturn"]
    assert all(4096 <= r["prompt_tokens"] <= 32768 for r in lc)
    assert all(16 <= r["output_tokens"] <= 96 for r in lc)
    assert all(r["arrival_offset_s"] >= 0 for r in lc)
    react = data["agent_react"]
    assert all(4096 <= r["prompt_tokens"] <= 5000 for r in react)
    assert all(r["turn_index"] >= 0 for r in react)
    rag = data["shared_prefix_rag"]
    assert all(r["prompt_tokens"] == 16384 for r in rag)
    # Shared document prefix per group.
    by_group: dict[str, list[str]] = {}
    for row in rag:
        by_group.setdefault(row["prefix_group"], []).append(row["text"])
    for texts in by_group.values():
        common_len = min(
            _common_prefix_len(texts[0], text) for text in texts[1:]
        )
        assert common_len > 1000


def test_required_fields_and_jsonl(tmp_path) -> None:
    generate_all(tmp_path)
    path = tmp_path / "agent_react.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    required = {
        "id",
        "arrival_offset_s",
        "prefix_group",
        "prompt_tokens",
        "output_tokens",
        "turn_index",
        "text",
    }
    assert set(lines[0]) == required
    assert len({row["id"] for row in lines}) == len(lines)
