"""Tiered KV store: HBM/DRAM -> NVMe -> remote NVMe-oF.

The store treats the first tier as the hot cache. When it overflows, the LRU
block is read back and demoted to the next tier (recursively), which means
demotion bytes are real I/O. A hit in a colder tier promotes the block back
toward the hot tier. Per-tier hit counters and the count of bytes moved between
tiers feed the analysis stage.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from kvoffloadbench.config import TierConfig
from kvoffloadbench.tiers.base import KVTier, TierError
from kvoffloadbench.tiers.dram import DRAMTier
from kvoffloadbench.tiers.nvme import NVMeTier
from kvoffloadbench.tiers.nvmeof_block import NVMeOFBlockTier
from kvoffloadbench.tiers.remote_tcp import RemoteTCPTier


@dataclass
class StoreCounters:
    puts: int = 0
    gets: int = 0
    lookups: int = 0
    lookup_hits: int = 0
    lookup_misses: int = 0
    offload_bytes: int = 0
    offload_ops: int = 0
    restore_bytes: int = 0
    restore_ops: int = 0
    restore_seconds: float = 0.0
    lookup_restore_bytes: int = 0
    lookup_restore_seconds: float = 0.0
    promotions: int = 0
    promotion_skips: int = 0
    drops: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "puts": self.puts,
            "gets": self.gets,
            "lookups": self.lookups,
            "lookup_hits": self.lookup_hits,
            "lookup_misses": self.lookup_misses,
            "offload_bytes": self.offload_bytes,
            "offload_ops": self.offload_ops,
            "restore_bytes": self.restore_bytes,
            "restore_ops": self.restore_ops,
            "restore_seconds": self.restore_seconds,
            "lookup_restore_bytes": self.lookup_restore_bytes,
            "lookup_restore_seconds": self.lookup_restore_seconds,
            "promotions": self.promotions,
            "promotion_skips": self.promotion_skips,
            "drops": self.drops,
        }


class TieredKVStore:
    """Chained KV block store with LRU demotion and promotion on hit."""

    def __init__(self, tiers: Sequence[KVTier], name: str = "tiered") -> None:
        if not tiers:
            raise ValueError("a TieredKVStore needs at least one tier")
        self.tiers = list(tiers)
        self.name = name
        self.counters = StoreCounters()
        self._counter_lock = threading.Lock()

    @property
    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]

    @property
    def hot(self) -> KVTier:
        return self.tiers[0]

    @property
    def block_bytes(self) -> int:
        return self.hot.block_bytes

    # -- core operations ----------------------------------------------------

    def put(self, block_id: str, data: bytes | memoryview) -> None:
        payload = bytes(data)
        with self._counter_lock:
            self.counters.puts += 1
        # Invalidate stale copies below the hot tier.
        for tier in self.tiers[1:]:
            if tier.contains(block_id):
                tier.evict(block_id)
        self._insert(0, block_id, payload)

    def _insert(self, index: int, block_id: str, data: bytes) -> None:
        tier = self.tiers[index]
        if len(data) > tier.capacity_bytes:
            if index + 1 < len(self.tiers):
                self._note_offload(len(data))
                self._insert(index + 1, block_id, data)
                return
            raise TierError(
                f"block {block_id!r} ({len(data)} bytes) exceeds the last tier "
                f"capacity ({tier.capacity_bytes} bytes)"
            )
        if index == len(self.tiers) - 1:
            # The last tier owns its own eviction policy (the TCP target uses a
            # server-side LRU); there is nowhere to demote to.
            tier.put(block_id, data)
            return
        while tier.resident_bytes + len(data) > tier.capacity_bytes:
            try:
                evicted = tier.evict_lru(1)
            except TierError:
                break
            if not evicted:
                break
            for victim_id, victim_data in evicted:
                self._note_offload(len(victim_data))
                self._insert(index + 1, victim_id, victim_data)
        tier.put(block_id, data)

    def get(self, block_id: str) -> bytes | None:
        with self._counter_lock:
            self.counters.gets += 1
        t0 = time.perf_counter()
        for index, tier in enumerate(self.tiers):
            data = tier.get(block_id)
            if data is None:
                continue
            if index > 0:
                self._promote(index, block_id, data)
                elapsed = time.perf_counter() - t0
                with self._counter_lock:
                    self.counters.restore_bytes += len(data)
                    self.counters.restore_ops += 1
                    self.counters.restore_seconds += elapsed
            return data
        return None

    def _promote(self, index: int, block_id: str, data: bytes) -> None:
        """Move a block hit in tier ``index`` back toward the hot tier.

        Promotion is admission-controlled: if no hotter tier has room, the
        block stays where it is instead of evicting (and cascading a demotion
        of) another live block. Unconditional promotion on every hit turns a
        sequential prefix scan into a write-amplification storm, which is a
        real cache pathology this benchmark must not accidentally measure.
        """
        promoted = False
        for target in range(index):
            target_tier = self.tiers[target]
            if target_tier.contains(block_id):
                promoted = True
                break
            if target_tier.resident_bytes + len(data) <= target_tier.capacity_bytes:
                target_tier.put(block_id, data)
                promoted = True
                break
        if promoted:
            cold = self.tiers[index]
            if cold.contains(block_id):
                cold.evict(block_id)
            with self._counter_lock:
                self.counters.promotions += 1
        else:
            with self._counter_lock:
                self.counters.promotion_skips += 1

    def lookup(self, block_ids: Sequence[str]) -> tuple[int, int, int, float]:
        """Look up many blocks, promoting hits; return hit/miss/bytes/seconds."""
        import time

        hits = 0
        misses = 0
        nbytes = 0
        t0 = time.perf_counter()
        for block_id in block_ids:
            data = self.get(block_id)
            if data is None:
                misses += 1
            else:
                hits += 1
                nbytes += len(data)
        elapsed = time.perf_counter() - t0
        with self._counter_lock:
            self.counters.lookups += 1
            self.counters.lookup_hits += hits
            self.counters.lookup_misses += misses
            self.counters.lookup_restore_bytes += nbytes
            self.counters.lookup_restore_seconds += elapsed
        return hits, misses, nbytes, elapsed

    def contains(self, block_id: str) -> bool:
        return any(tier.contains(block_id) for tier in self.tiers)

    def evict(self, block_id: str) -> bool:
        removed = False
        for tier in self.tiers:
            removed = tier.evict(block_id) or removed
        return removed

    def demote_hot(self) -> int:
        """Force every hot-tier block down one level (used by validation)."""
        if len(self.tiers) < 2:
            return 0
        moved = 0
        while True:
            evicted = self.hot.evict_lru(1)
            if not evicted:
                break
            for victim_id, data in evicted:
                self._note_offload(len(data))
                self._insert(1, victim_id, data)
                moved += 1
        return moved

    def demote_all_to_last(self) -> int:
        """Force every block in every non-final tier down to the last tier."""
        if len(self.tiers) < 2:
            return 0
        moved = 0
        for index in range(len(self.tiers) - 1):
            source = self.tiers[index]
            while True:
                evicted = source.evict_lru(1)
                if not evicted:
                    break
                for victim_id, data in evicted:
                    self._note_offload(len(data))
                    self._insert(len(self.tiers) - 1, victim_id, data)
                    moved += 1
        return moved

    def _note_offload(self, nbytes: int) -> None:
        with self._counter_lock:
            self.counters.offload_bytes += nbytes
            self.counters.offload_ops += 1

    # -- accounting ---------------------------------------------------------

    def stats(self) -> dict[str, object]:
        with self._counter_lock:
            counters = self.counters.to_dict()
        tiers = [tier.stats_dict() for tier in self.tiers]
        return {
            "name": self.name,
            "tiers": tiers,
            "counters": counters,
            "hit_distribution": [tier.stats().hits for tier in self.tiers],
            "resident_bytes": [tier.resident_bytes for tier in self.tiers],
        }

    def reset_stats(self) -> None:
        with self._counter_lock:
            self.counters = StoreCounters()
        for tier in self.tiers:
            tier.reset_stats()

    def close(self) -> None:
        for tier in self.tiers:
            tier.close()

    def __enter__(self) -> TieredKVStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Construction from TierConfig
# ---------------------------------------------------------------------------


def build_tier(cfg: TierConfig, name: str | None = None) -> KVTier:
    """Instantiate one tier from its config, raising :class:`TierUnavailable`."""
    label = name or cfg.kind
    block_bytes = cfg.block_bytes
    if cfg.kind == "hbm":
        from kvoffloadbench.tiers.hbm import HBMTier

        return HBMTier(
            name=label,
            capacity_bytes=cfg.capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=cfg.queue_depth,
            direct_io=cfg.direct_io,
            device=str(cfg.extra.get("device", "cuda:0")),
        )
    if cfg.kind == "dram":
        return DRAMTier(
            name=label,
            capacity_bytes=cfg.capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=cfg.queue_depth,
            direct_io=cfg.direct_io,
            pinned=cfg.extra.get("pinned"),
        )
    if cfg.kind == "nvme":
        return NVMeTier(
            name=label,
            path=cfg.path or "results/cache/nvme/tier.bin",
            capacity_bytes=cfg.capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=cfg.queue_depth,
            direct_io=cfg.direct_io,
        )
    if cfg.kind == "remote_tcp":
        return RemoteTCPTier(
            name=label,
            endpoint=cfg.endpoint or "127.0.0.1:4420",
            capacity_bytes=cfg.capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=cfg.queue_depth,
            direct_io=cfg.direct_io,
            timeout_s=float(cfg.extra.get("timeout_s", 60.0)),
        )
    if cfg.kind == "nvmeof_block":
        return NVMeOFBlockTier(
            name=label,
            path=cfg.path,
            device_env=str(cfg.extra.get("device_env", "KVOB_NVMEOF_DEV")),
            capacity_bytes=cfg.capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=cfg.queue_depth,
            direct_io=cfg.direct_io,
        )
    raise TierError(f"unknown tier kind {cfg.kind!r}")


def build_store(
    tier_cfgs: Sequence[TierConfig],
    name: str = "tiered",
    fallback_remote: bool = True,
    logger: object | None = None,
) -> TieredKVStore:
    """Build a tier chain, degrading an unavailable NVMe-oF device to TCP.

    The fallback is explicit and logged (and recorded by the caller in the run
    manifest); it never silently changes the chain without a trace.
    """
    import logging

    log = logger or logging.getLogger(__name__)
    remote_cfg = next((c for c in tier_cfgs if c.kind == "remote_tcp"), None)
    tiers: list[KVTier] = []
    for cfg in tier_cfgs:
        try:
            tiers.append(build_tier(cfg))
        except Exception as exc:
            if cfg.kind == "nvmeof_block" and fallback_remote and remote_cfg is not None:
                log.warning(
                    "tier %s unavailable (%s); falling back to remote_tcp at %s",
                    cfg.kind,
                    exc,
                    remote_cfg.endpoint,
                )
                continue
            raise
    if not tiers:
        raise TierError("no tiers could be constructed")
    return TieredKVStore(tiers, name=name)
