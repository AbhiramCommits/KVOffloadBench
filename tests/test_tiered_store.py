"""Tiered store demotion, promotion, and hit accounting."""

from __future__ import annotations

import os

from kvoffloadbench.tiers.dram import DRAMTier
from kvoffloadbench.tiers.tiered_store import TieredKVStore

BLOCK = 4096


def _store(hot_blocks: int = 4, cold_blocks: int = 8) -> TieredKVStore:
    return TieredKVStore(
        [
            DRAMTier(name="dram", capacity_bytes=hot_blocks * BLOCK, block_bytes=BLOCK),
            DRAMTier(name="cold", capacity_bytes=cold_blocks * BLOCK, block_bytes=BLOCK),
        ]
    )


def test_demotion_on_overflow_and_restore_byte_math() -> None:
    store = _store()
    payloads = {f"k{i}": os.urandom(BLOCK) for i in range(6)}
    for key, data in payloads.items():
        store.put(key, data)
    assert store.hot.resident_bytes == 4 * BLOCK
    assert store.tiers[1].resident_bytes == 2 * BLOCK
    assert store.counters.offload_ops == 2
    assert store.counters.offload_bytes == 2 * BLOCK
    assert all(store.contains(key) for key in payloads)

    oldest = "k0"  # first demoted block lives in the cold tier
    assert store.tiers[1].contains(oldest)
    before = store.counters.restore_bytes
    assert store.get(oldest) == payloads[oldest]
    assert store.counters.restore_bytes == before + BLOCK
    assert store.counters.restore_ops >= 1
    assert store.counters.restore_seconds > 0
    # Hot tier is full, so admission control refuses the promotion.
    assert store.counters.promotion_skips >= 1
    assert store.counters.promotions == 0


def test_promotion_when_hot_tier_has_room() -> None:
    store = _store()
    for i in range(5):
        store.put(f"k{i}", os.urandom(BLOCK))
    cold_id = "k0"
    assert store.tiers[1].contains(cold_id)
    store.hot.evict("k1")
    assert store.get(cold_id) is not None
    assert store.counters.promotions == 1
    assert store.hot.contains(cold_id)
    assert not store.tiers[1].contains(cold_id)


def test_hit_distribution_and_lookup_contract() -> None:
    store = _store()
    for i in range(6):
        store.put(f"k{i}", os.urandom(BLOCK))
    hits, misses, nbytes, seconds = store.lookup(["k0", "k5", "missing"])
    assert hits == 2 and misses == 1
    assert nbytes == 2 * BLOCK
    assert seconds >= 0
    assert store.counters.lookup_hits == 2
    assert store.counters.lookup_misses == 1
    dist = store.stats()["hit_distribution"]
    assert sum(dist) >= 2
    assert len(dist) == 2


def test_demote_all_to_last_chain() -> None:
    store = _store(hot_blocks=2, cold_blocks=8)
    payloads = {f"k{i}": os.urandom(BLOCK) for i in range(4)}
    for key, data in payloads.items():
        store.put(key, data)
    moved = store.demote_all_to_last()
    assert moved == 2
    assert store.hot.resident_bytes == 0
    assert store.tiers[1].resident_bytes == 4 * BLOCK
    for key, data in payloads.items():
        assert store.tiers[1].get(key) == data
