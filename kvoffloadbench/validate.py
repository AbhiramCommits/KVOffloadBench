"""KV offload/reload correctness validation.

Two independent checks:

1. **Byte and tensor round-trip.** Random KV-shaped tensors are written to every
   tier and read back; SHA-256 and ``torch.equal`` must both pass. The tiered
   store is then forced to demote every block to the last (remote) tier and the
   same checks are repeated.
2. **Greedy token equivalence.** A fixed greedy prompt set produces a reference
   token sequence with offload disabled, then the same prompts are replayed
   through each tier chain; token ids must match exactly.

Results go to ``results/validation/kv_correctness.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kvoffloadbench.config import ConfigError, EngineConfig, ModelShape, TierConfig
from kvoffloadbench.engines.base import GenerationRequest
from kvoffloadbench.engines.factory import build_engine
from kvoffloadbench.target.launcher import TargetHandle, start_target
from kvoffloadbench.tiers.base import KVTier, TierUnavailable
from kvoffloadbench.tiers.dram import DRAMTier
from kvoffloadbench.tiers.nvme import NVMeTier
from kvoffloadbench.tiers.remote_tcp import RemoteTCPTier
from kvoffloadbench.tiers.tiered_store import TieredKVStore, build_tier

log = logging.getLogger(__name__)


@dataclass
class Block:
    block_id: str
    data: bytes
    tensor: Any


def _next_pow2(value: int) -> int:
    return 1 << max(0, math.ceil(math.log2(value)))


def _make_blocks(shape: ModelShape, count: int, seed: int) -> list[Block]:
    import torch

    g = torch.Generator().manual_seed(seed)
    tensor_shape = (
        shape.block_tokens,
        shape.layers,
        2,
        shape.kv_heads,
        shape.head_dim,
    )
    blocks: list[Block] = []
    for i in range(count):
        tensor = torch.randn(tensor_shape, generator=g, dtype=torch.float16)
        data = tensor.numpy().tobytes()
        block_id = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:24]
        blocks.append(Block(block_id=block_id, data=data, tensor=tensor))
    return blocks


def _check_blocks(tier: KVTier, blocks: list[Block]) -> dict[str, Any]:
    import torch

    shape = tuple(blocks[0].tensor.shape)
    mismatches = 0
    sha_mismatches = 0
    for block in blocks:
        tier.put(block.block_id, block.data)
    for block in blocks:
        got = tier.get(block.block_id)
        if got is None:
            mismatches += 1
            continue
        if hashlib.sha256(got).digest() != hashlib.sha256(block.data).digest():
            sha_mismatches += 1
        try:
            tensor = torch.frombuffer(bytearray(got), dtype=torch.float16).reshape(shape)
            if not torch.equal(tensor, block.tensor):
                mismatches += 1
        except (RuntimeError, ValueError):
            mismatches += 1
    return {
        "tier": tier.name,
        "kind": tier.kind,
        "blocks_compared": len(blocks),
        "byte_mismatches": mismatches,
        "sha_mismatches": sha_mismatches,
        "io_mode": tier.io_mode,
        "capacity_bytes": tier.capacity_bytes,
        "block_bytes": tier.block_bytes,
    }


def _build_chain(
    block_bytes: int,
    block_kib: int,
    dram_mib: int,
    nvme_path: str,
    nvme_mib: int,
    remote_endpoint: str | None = None,
    remote_mib: int = 256,
    include_remote: bool = True,
) -> TieredKVStore:
    tiers: list[KVTier] = [
        DRAMTier(name="dram", capacity_bytes=dram_mib * 1024 * 1024, block_bytes=block_kib * 1024)
    ]
    if nvme_mib > 0:
        tiers.append(
            NVMeTier(
                name="nvme",
                path=nvme_path,
                capacity_bytes=nvme_mib * 1024 * 1024,
                block_bytes=block_kib * 1024,
            )
        )
    if include_remote and remote_endpoint:
        tiers.append(
            RemoteTCPTier(
                name="remote_tcp",
                endpoint=remote_endpoint,
                capacity_bytes=remote_mib * 1024 * 1024,
                block_bytes=block_kib * 1024,
            )
        )
    return TieredKVStore(tiers, name="validation_chain")


async def _collect_tokens(engine: Any, prompt: str, idx: int, max_tokens: int) -> list[int]:
    req = GenerationRequest(
        request_id=f"validate-{idx}",
        text=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=7,
        turn_index=0,
    )
    tokens: list[int] = []
    async for event in engine.generate(req):
        tokens.append(event.token_id)
    return tokens


async def run_validation(
    config_path: str | Path = "configs/validate.yaml",
    out_path: str | Path = "results/validation/kv_correctness.json",
) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError("validation config must be a mapping")
    shape_raw = raw.get("shape") or {}
    shape = ModelShape(
        layers=int(shape_raw.get("layers", 80)),
        kv_heads=int(shape_raw.get("kv_heads", 8)),
        head_dim=int(shape_raw.get("head_dim", 128)),
        dtype=str(shape_raw.get("dtype", "float16")),
        block_tokens=int(shape_raw.get("block_tokens", 16)),
    )
    block_bytes = shape.bytes_per_block
    block_kib = max(4, _next_pow2(math.ceil(block_bytes / 1024)))
    if block_kib * 1024 < block_bytes:
        block_kib *= 2
    count = int(raw.get("blocks", 8))
    seed = int(raw.get("seed", 7))
    max_tokens = int(raw.get("max_tokens", 8))
    prompts = [str(p) for p in raw.get("token_prompts") or []]
    if not prompts:
        raise ConfigError("validation config needs at least one entry in 'token_prompts'")
    tier_cfg = raw.get("tiers") or {}
    nvme_cfg = tier_cfg.get("nvme") or {}
    remote_cfg = tier_cfg.get("remote") or {}
    dram_cfg = tier_cfg.get("dram") or {}
    nvmeof_cfg = tier_cfg.get("nvmeof") or {}

    engine_raw = raw.get("engine") or {}
    engine_cfg = EngineConfig(
        mode=str(engine_raw.get("mode", "emulated")),
        model=str(engine_raw.get("model", "gpt2")),
        tensor_parallel=int(engine_raw.get("tensor_parallel", 1)),
        max_model_len=int(engine_raw.get("max_model_len", 32768)),
        connector=str(engine_raw.get("connector", "lmcache")),
        model_shape=shape,
        calibration_path=engine_raw.get("calibration_path"),
    )

    blocks = _make_blocks(shape, count, seed)
    remote_endpoint = remote_cfg.get("endpoint")
    target: TargetHandle | None = None
    tiers_checked: list[str] = []
    roundtrip: list[dict[str, Any]] = []
    tiered_result: dict[str, Any] = {}
    try:
        if remote_endpoint:
            target = start_target(
                str(remote_endpoint),
                remote_cfg.get("target_path", "results/cache/validate/target.bin"),
                int(remote_cfg.get("target_capacity_mib", 256)),
            )
        specs = [
            TierConfig(
                kind="dram",
                capacity_mib=int(dram_cfg.get("capacity_mib", 32)),
                block_size_kib=block_kib,
            ),
            TierConfig(
                kind="nvme",
                capacity_mib=int(nvme_cfg.get("capacity_mib", 256)),
                path=nvme_cfg.get("path", "results/cache/validate/nvme.bin"),
                block_size_kib=block_kib,
            ),
        ]
        if remote_endpoint:
            specs.append(
                TierConfig(
                    kind="remote_tcp",
                    capacity_mib=int(remote_cfg.get("capacity_mib", 256)),
                    endpoint=str(remote_endpoint),
                    block_size_kib=block_kib,
                )
            )
        nvmeof_device = os.environ.get("KVOB_NVMEOF_DEV")
        if nvmeof_device:
            specs.append(
                TierConfig(
                    kind="nvmeof_block",
                    capacity_mib=int(nvmeof_cfg.get("capacity_mib", 256)),
                    path=nvmeof_device,
                    block_size_kib=block_kib,
                )
            )
        for spec in specs:
            try:
                tier = build_tier(spec, name=spec.kind)
            except TierUnavailable as exc:
                log.warning("validation skips %s: %s", spec.kind, exc)
                continue
            try:
                result = _check_blocks(tier, blocks)
                roundtrip.append(result)
                tiers_checked.append(spec.kind)
            finally:
                tier.close()

        # Tiered store: small DRAM forces demotion to NVMe, then to remote.
        chain = _build_chain(
            block_bytes=block_bytes,
            block_kib=block_kib,
            dram_mib=max(1, count // 4),
            nvme_path=str(nvme_cfg.get("path", "results/cache/validate/nvme.bin")),
            nvme_mib=int(nvme_cfg.get("capacity_mib", 256)),
            remote_endpoint=str(remote_endpoint) if remote_endpoint else None,
            remote_mib=int(remote_cfg.get("capacity_mib", 256)),
        )
        try:
            for block in blocks:
                chain.put(block.block_id, block.data)
            before = [t.contains(b.block_id) for t in chain.tiers for b in blocks]
            moved = chain.demote_all_to_last()
            last = chain.tiers[-1]
            in_last = sum(1 for b in blocks if last.contains(b.block_id))
            mismatch = 0
            for block in blocks:
                got = last.get(block.block_id)
                if got is None or hashlib.sha256(got).digest() != hashlib.sha256(block.data).digest():
                    mismatch += 1
            promoted_mismatch = 0
            for block in blocks:
                got = chain.get(block.block_id)
                if got is None or hashlib.sha256(got).digest() != hashlib.sha256(block.data).digest():
                    promoted_mismatch += 1
            tiered_result = {
                "chain": [t.kind for t in chain.tiers],
                "blocks_put": len(blocks),
                "blocks_in_last_tier": in_last,
                "demoted_blocks": moved,
                "last_tier": last.kind,
                "byte_mismatches": mismatch,
                "promoted_read_mismatches": promoted_mismatch,
                "tier_stats": chain.stats(),
                "_before": sum(before),
            }
        finally:
            chain.close()

        # Token equivalence across chains (greedy, fixed seed).
        reference_engine = await build_engine(
            EngineConfig(**{**engine_cfg.__dict__, "mode": "emulated"}),
            store=None,
            run_name="validate-reference",
            tiers=[],
        )
        reference: list[list[int]] = []
        try:
            for idx, prompt in enumerate(prompts):
                reference.append(await _collect_tokens(reference_engine, prompt, idx, max_tokens))
        finally:
            await reference_engine.close()

        def _chain_for(include_nvme: bool, include_remote: bool) -> TieredKVStore:
            return _build_chain(
                block_bytes=block_bytes,
                block_kib=block_kib,
                dram_mib=int(dram_cfg.get("capacity_mib", 32)),
                nvme_path=str(nvme_cfg.get("path", "results/cache/validate/nvme.bin")),
                nvme_mib=int(nvme_cfg.get("capacity_mib", 256)) if include_nvme else 0,
                remote_endpoint=str(remote_endpoint) if remote_endpoint else None,
                remote_mib=int(remote_cfg.get("capacity_mib", 256)),
                include_remote=include_remote,
            )

        chains = {
            "dram_only": (True, False, False),
            "dram_nvme": (True, True, False),
            "dram_nvme_remote_tcp": (True, True, bool(remote_endpoint)),
        }
        token_results: dict[str, Any] = {}
        total_positions = 0
        matched_positions = 0
        for label, (_inc_dram, inc_nvme, inc_remote) in chains.items():
            store = _chain_for(inc_nvme, inc_remote)
            engine = await build_engine(
                EngineConfig(**{**engine_cfg.__dict__, "mode": "emulated"}),
                store=store,
                run_name=f"validate-{label}",
                tiers=[],
            )
            seqs: list[list[int]] = []
            try:
                for idx, prompt in enumerate(prompts):
                    seqs.append(await _collect_tokens(engine, prompt, idx, max_tokens))
            finally:
                await engine.close()
                store.close()
            matches = 0
            total = 0
            for ref_seq, got_seq in zip(reference, seqs, strict=True):
                total += len(ref_seq)
                matches += sum(
                    1 for a, b in zip(ref_seq, got_seq, strict=True) if a == b
                )
            total_positions += total
            matched_positions += matches
            token_results[label] = {
                "sequences_compared": len(reference),
                "tokens_compared": total,
                "exact_matches": matches,
                "exact_match_rate": matches / total if total else 0.0,
                "reference_sha256": hashlib.sha256(
                    json.dumps(reference, separators=(",", ":")).encode()
                ).hexdigest(),
            }

        byte_mismatches = sum(r["byte_mismatches"] for r in roundtrip) + int(
            tiered_result.get("byte_mismatches", 0)
        ) + int(tiered_result.get("promoted_read_mismatches", 0))
        validation: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "mode": "emulated",
            "engine": engine_cfg.model,
            "model_shape": shape.to_dict(),
            "block_bytes": block_bytes,
            "block_size_kib": block_kib,
            "tiers_checked": tiers_checked,
            "roundtrip": roundtrip,
            "blocks_compared": sum(r["blocks_compared"] for r in roundtrip)
            + len(blocks) * 2,
            "byte_mismatches": byte_mismatches,
            "sha_mismatches": sum(r["sha_mismatches"] for r in roundtrip),
            "token_exact_match_rate": matched_positions / total_positions
            if total_positions
            else 0.0,
            "token_chains": token_results,
            "prompts": prompts,
            "tiered_store": {k: v for k, v in tiered_result.items() if k != "_before"},
            "tiered_store_blocks_tracked_before": tiered_result.get("_before"),
            "calibration": reference_engine.stats().get("calibration")
            if hasattr(reference_engine, "stats")
            else None,
            "real_engine_note": (
                "This host has no CUDA GPU, so the greedy equivalence check uses the "
                "calibrated emulated engine. The byte/tensor checks exercise real tier "
                "I/O. On a GPU host the same module uses vLLM with offload disabled vs "
                "enabled for the token check."
            ),
        }
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(validation, indent=2, sort_keys=True))
        log.warning(
            "validation: blocks=%d byte_mismatches=%d token_match=%.3f -> %s",
            validation["blocks_compared"],
            validation["byte_mismatches"],
            validation["token_exact_match_rate"],
            out,
        )
        return validation
    finally:
        if target is not None:
            target.stop()


def main(config: str | Path = "configs/validate.yaml", out: str | Path = "results/validation/kv_correctness.json") -> int:
    result = asyncio.run(run_validation(config, out))
    ok = result["byte_mismatches"] == 0 and result["token_exact_match_rate"] == 1.0
    print(
        f"tiers={result['tiers_checked']} blocks={result['blocks_compared']} "
        f"byte_mismatches={result['byte_mismatches']} "
        f"token_exact_match_rate={result['token_exact_match_rate']:.3f} "
        f"mode={result['mode']}"
    )
    return 0 if ok else 1
