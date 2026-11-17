"""Host DRAM tier: plain or CUDA-pinned buffers with LRU eviction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from kvoffloadbench.tiers.base import KVTier, TierCapacityError

try:  # pragma: no cover - torch is a hard dependency, but keep the tier usable without it
    import torch
except Exception:
    torch = None  # type: ignore[assignment]


class _PlainSlot:
    __slots__ = ("buf", "nbytes")

    def __init__(self, size: int) -> None:
        self.buf = bytearray(size)
        self.nbytes = 0

    def write(self, data: memoryview) -> None:
        self.buf[: len(data)] = data
        self.nbytes = len(data)

    def read(self) -> bytes:
        return bytes(self.buf[: self.nbytes])

    def close(self) -> None:
        self.buf = bytearray(0)


class _PinnedSlot:
    __slots__ = ("nbytes", "tensor")

    def __init__(self, size: int) -> None:
        assert torch is not None
        self.tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.nbytes = 0

    def write(self, data: memoryview) -> None:
        assert torch is not None
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self.tensor[: len(data)].copy_(src)
        self.nbytes = len(data)

    def read(self) -> bytes:
        assert torch is not None
        return bytes(self.tensor[: self.nbytes].cpu().numpy().tobytes())

    def close(self) -> None:
        self.tensor = torch.empty(0, dtype=torch.uint8)


class DRAMTier(KVTier):
    """In-process DRAM cache.

    Buffers are CUDA page-locked (``pin_memory=True``) when a CUDA device is
    present so they can be used for GPUDirect-style transfers; otherwise plain
    host bytearrays are used. There is no copy overhead hidden from the stats:
    the timed region includes the memcpy into/out of the buffer.
    """

    kind = "dram"

    def __init__(
        self,
        name: str = "dram",
        capacity_bytes: int = 1024 * 1024 * 1024,
        block_bytes: int = 64 * 1024,
        queue_depth: int = 8,
        direct_io: bool = False,
        pinned: bool | None = None,
    ) -> None:
        super().__init__(name, capacity_bytes, block_bytes, queue_depth, direct_io)
        self._entries: OrderedDict[str, _PlainSlot | _PinnedSlot] = OrderedDict()
        use_pin = bool(torch is not None and torch.cuda.is_available()) if pinned is None else pinned
        if use_pin and (torch is None or not torch.cuda.is_available()):
            use_pin = False
        self._pinned = use_pin
        self.io_mode = "pinned-dram" if use_pin else "dram"
        self._stats.io_mode = self.io_mode
        self._resident = 0

    def _new_slot(self, size: int) -> _PlainSlot | _PinnedSlot:
        return _PinnedSlot(size) if self._pinned else _PlainSlot(size)

    def _put(self, block_id: str, data: memoryview) -> None:
        n = len(data)
        if n > self.capacity_bytes:
            raise TierCapacityError(
                f"payload {n} bytes exceeds DRAM tier capacity {self.capacity_bytes}"
            )
        with self._lock:
            existing = self._entries.pop(block_id, None)
            if existing is not None:
                self._resident -= existing.nbytes
                existing.close()
            if n > self.block_bytes:
                # Allocate an exactly-sized slot; DRAM is byte-granular.
                slot = self._new_slot(n)
            else:
                slot = self._new_slot(self.block_bytes)
            while self._entries and self._resident + n > self.capacity_bytes:
                _, old = self._entries.popitem(last=False)
                self._resident -= old.nbytes
                old.close()
                self._stats.evictions += 1
            slot.write(data)
            self._entries[block_id] = slot
            self._resident += n

    def _get(self, block_id: str) -> bytes | None:
        with self._lock:
            slot = self._entries.get(block_id)
            if slot is None:
                return None
            self._entries.move_to_end(block_id)
        return slot.read()

    def _contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._entries

    def _evict(self, block_id: str) -> bool:
        with self._lock:
            slot = self._entries.pop(block_id, None)
            if slot is None:
                return False
            self._resident -= slot.nbytes
            slot.close()
            self._stats.evictions += 1
            return True

    def evict_lru(self, count: int = 1) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        with self._lock:
            for _ in range(min(count, len(self._entries))):
                block_id, slot = self._entries.popitem(last=False)
                self._resident -= slot.nbytes
                self._stats.evictions += 1
                data = slot.read()
                slot.close()
                out.append((block_id, data))
        return out

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return self._resident

    def _close(self) -> None:
        with self._lock:
            for slot in self._entries.values():
                slot.close()
            self._entries.clear()
            self._resident = 0

    def debug_state(self) -> dict[str, Any]:
        with self._lock:
            return {"blocks": len(self._entries), "resident_bytes": self._resident}
