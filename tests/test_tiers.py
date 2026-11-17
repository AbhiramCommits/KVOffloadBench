"""Tier backend correctness: byte-exactness, LRU order, capacity, alignment."""

from __future__ import annotations

import ctypes
import os

import pytest
from kvoffloadbench.target.launcher import start_target
from kvoffloadbench.tiers.aligned import AlignedBuffer, round_up
from kvoffloadbench.tiers.base import TierCapacityError
from kvoffloadbench.tiers.dram import DRAMTier
from kvoffloadbench.tiers.nvme import NVMeTier
from kvoffloadbench.tiers.remote_tcp import RemoteTCPTier


def test_dram_roundtrip_lru_and_capacity() -> None:
    block = 4096
    tier = DRAMTier(capacity_bytes=3 * block, block_bytes=block)
    payloads = {name: os.urandom(block) for name in "abc"}
    for name, data in payloads.items():
        tier.put(name, data)
    assert tier.get("a") == payloads["a"]
    tier.put("d", os.urandom(block))  # a is now most-recently used; b is LRU
    assert not tier.contains("b")
    assert tier.contains("a") and tier.contains("c") and tier.contains("d")
    assert tier.resident_bytes == 3 * block
    assert tier.stats().evictions == 1
    assert tier.stats().bytes_written == 4 * block
    assert tier.stats().bytes_read == block
    assert tier.stats().hits == 1
    with pytest.raises(TierCapacityError):
        tier.put("too-big", b"x" * (4 * block))


def test_dram_evict_lru_returns_payload_and_order() -> None:
    block = 4096
    tier = DRAMTier(capacity_bytes=2 * block, block_bytes=block)
    a, b = b"a" * block, b"b" * block
    tier.put("a", a)
    tier.put("b", b)
    evicted = tier.evict_lru(1)
    assert evicted == [("a", a)]
    assert tier.resident_bytes == block
    assert tier.stats().evictions == 1


def test_nvme_roundtrip_with_unaligned_payload(tmp_path) -> None:
    block = 16 * 1024
    tier = NVMeTier(
        path=tmp_path / "nvme.bin",
        capacity_bytes=8 * block,
        block_bytes=block,
        queue_depth=2,
    )
    try:
        payload = os.urandom(block - 1234)  # not a multiple of 4 KiB
        tier.put("x", payload)
        assert tier.get("x") == payload
        assert tier.get("missing") is None
        assert tier.io_mode in {"odirect", "fnocache", "osync", "buffered"}
        stats = tier.stats()
        assert stats.bytes_written == len(payload)
        assert stats.hits == 1 and stats.misses == 1
        # Fill past capacity: oldest blocks must be evicted, newest must survive.
        for i in range(12):
            tier.put(f"k{i}", os.urandom(4096))
        assert tier.resident_bytes == tier.capacity_bytes
        assert tier.stats().evictions >= 4
    finally:
        tier.close()


def test_aligned_buffer_is_page_aligned() -> None:
    buf = AlignedBuffer(5000, 4096)
    try:
        assert buf.size == round_up(5000, 4096)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buf.view()))
        assert address % 4096 == 0
        buf.write(b"x" * 5000)
        assert buf.bytes(5000) == b"x" * 5000
    finally:
        buf.close()


def test_remote_tcp_roundtrip_and_queue_depth(tmp_path, free_port) -> None:
    endpoint = f"127.0.0.1:{free_port}"
    handle = start_target(
        endpoint, tmp_path / "target.bin", capacity_mib=8, direct_io=False
    )
    tier = RemoteTCPTier(
        endpoint=endpoint,
        capacity_bytes=8 * 1024 * 1024,
        block_bytes=4096,
        queue_depth=8,
    )
    try:
        payload = os.urandom(4096)
        tier.put("one", payload)
        assert tier.get("one") == payload
        assert tier.get("nope") is None
        assert tier.ping()
        items = [(f"b{i}", os.urandom(4096)) for i in range(32)]
        tier.put_many(items)
        got = tier.get_many([key for key, _ in items])
        assert all(value == data for value, (_, data) in zip(got, items, strict=True))
        stats = tier.server_stats()
        assert stats["blocks"] >= 33
        assert tier.stats().hits + tier.stats().misses == 34
    finally:
        tier.close()
        handle.stop()
