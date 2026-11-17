"""Optional HBM tier backed by CUDA tensors.

This tier exists so the tier chain can include GPU memory on a real 4xH200
host; on CPU/MPS-only machines it raises :class:`TierUnavailable` and callers
fall back to the DRAM tier.
"""

from __future__ import annotations

from collections import OrderedDict

from kvoffloadbench.tiers.base import KVTier, TierCapacityError, TierUnavailable

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]


class _CudaSlot:
    __slots__ = ("nbytes", "tensor")

    def __init__(self, size: int, device: str) -> None:
        assert torch is not None
        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)
        self.nbytes = 0

    def write(self, data: memoryview) -> None:
        assert torch is not None
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self.tensor[: len(data)].copy_(src)
        self.nbytes = len(data)

    def read(self) -> bytes:
        return bytes(self.tensor[: self.nbytes].cpu().numpy().tobytes())

    def close(self) -> None:
        self.tensor = self.tensor.new_empty(0)


class HBMTier(KVTier):
    """GPU HBM block store (torch CUDA tensors)."""

    kind = "hbm"

    def __init__(
        self,
        name: str = "hbm",
        capacity_bytes: int = 1024 * 1024 * 1024,
        block_bytes: int = 64 * 1024,
        queue_depth: int = 8,
        direct_io: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if torch is None or not torch.cuda.is_available():
            raise TierUnavailable("HBM tier requires a CUDA device and torch with CUDA support")
        super().__init__(name, capacity_bytes, block_bytes, queue_depth, direct_io)
        self.device = device
        self._entries: OrderedDict[str, _CudaSlot] = OrderedDict()
        self._resident = 0
        self.io_mode = f"torch-{device}"

    def _put(self, block_id: str, data: memoryview) -> None:
        n = len(data)
        if n > self.capacity_bytes:
            raise TierCapacityError(f"payload {n} exceeds HBM tier capacity {self.capacity_bytes}")
        with self._lock:
            old = self._entries.pop(block_id, None)
            if old:
                self._resident -= old.nbytes
                old.close()
            while self._entries and self._resident + n > self.capacity_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._resident -= evicted.nbytes
                evicted.close()
                self._stats.evictions += 1
            slot = _CudaSlot(max(n, self.block_bytes), self.device)
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
