"""The correctness validator must detect a corrupting tier."""

from __future__ import annotations

from kvoffloadbench.config import ModelShape
from kvoffloadbench.tiers.dram import DRAMTier
from kvoffloadbench.validate import _check_blocks, _make_blocks


class CorruptingTier(DRAMTier):
    """Flips one byte on every read, simulating a bad DMA or SSD path."""

    def _get(self, block_id: str) -> bytes | None:
        data = super()._get(block_id)
        if data is None:
            return None
        mutated = bytearray(data)
        mutated[0] ^= 0xFF
        return bytes(mutated)


def test_good_tier_has_zero_mismatches() -> None:
    shape = ModelShape(layers=2, kv_heads=1, head_dim=16, dtype="float16", block_tokens=4)
    blocks = _make_blocks(shape, 3, seed=11)
    tier = DRAMTier(capacity_bytes=4 * 4096, block_bytes=4096)
    try:
        result = _check_blocks(tier, blocks)
    finally:
        tier.close()
    assert result["blocks_compared"] == 3
    assert result["byte_mismatches"] == 0
    assert result["sha_mismatches"] == 0


def test_corrupting_tier_is_detected() -> None:
    shape = ModelShape(layers=2, kv_heads=1, head_dim=16, dtype="float16", block_tokens=4)
    blocks = _make_blocks(shape, 2, seed=11)
    tier = CorruptingTier(capacity_bytes=4 * 4096, block_bytes=4096)
    try:
        result = _check_blocks(tier, blocks)
    finally:
        tier.close()
    assert result["byte_mismatches"] == 2
    assert result["sha_mismatches"] == 2
