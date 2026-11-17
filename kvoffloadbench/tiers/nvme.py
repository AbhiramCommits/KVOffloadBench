"""Local NVMe tier: preallocated file with a slot allocator and aligned I/O.

``O_DIRECT`` is used when the platform and filesystem accept it. On macOS,
which has no ``O_DIRECT``, ``F_NOCACHE`` is used instead. The mode actually in
force is exposed as ``io_mode`` and lands in every microbenchmark row.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import platform
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kvoffloadbench.tiers.aligned import AlignedBuffer, round_up
from kvoffloadbench.tiers.base import KVTier, TierCapacityError

ALIGNMENT = 4096


class NVMeTier(KVTier):
    kind = "nvme"

    def __init__(
        self,
        name: str = "nvme",
        path: str | Path = "results/cache/nvme/tier.bin",
        capacity_bytes: int = 8 * 1024 * 1024 * 1024,
        block_bytes: int = 64 * 1024,
        queue_depth: int = 8,
        direct_io: bool = True,
    ) -> None:
        super().__init__(name, capacity_bytes, block_bytes, queue_depth, direct_io)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._slots = max(1, capacity_bytes // block_bytes)
        self.capacity_bytes = self._slots * block_bytes
        self._map: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._free: list[int] = list(range(self._slots - 1, -1, -1))
        self._fd = self._open_backing()
        self._scratch = threading.local()
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    # -- backing file -------------------------------------------------------

    def _open_backing(self) -> int:
        flags = os.O_RDWR | os.O_CREAT
        mode = "buffered"
        if self.direct_io_requested:
            if hasattr(os, "O_DIRECT"):
                try:
                    fd = os.open(self.path, flags | os.O_DIRECT, 0o600)
                    os.ftruncate(fd, self.capacity_bytes)
                    self.io_mode = "odirect"
                    self._stats.io_mode = "odirect"
                    return fd
                except OSError:
                    pass
            elif platform.system() == "Darwin":
                fd = os.open(self.path, flags, 0o600)
                try:
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    os.ftruncate(fd, self.capacity_bytes)
                    self.io_mode = "fnocache"
                    self._stats.io_mode = "fnocache"
                    return fd
                except OSError:
                    os.close(fd)
        if hasattr(os, "O_SYNC") and platform.system() == "Linux":
            try:
                fd = os.open(self.path, flags | os.O_SYNC, 0o600)
                os.ftruncate(fd, self.capacity_bytes)
                self.io_mode = "osync"
                self._stats.io_mode = "osync"
                return fd
            except OSError:
                pass
        fd = os.open(self.path, flags, 0o600)
        os.ftruncate(fd, self.capacity_bytes)
        self.io_mode = mode
        self._stats.io_mode = mode
        return fd

    # -- aligned scratch ----------------------------------------------------

    def _buffer(self) -> AlignedBuffer:
        buf = getattr(self._scratch, "buf", None)
        if buf is None:
            buf = AlignedBuffer(self.block_bytes, ALIGNMENT)
            self._scratch.buf = buf
        return buf  # type: ignore[no-any-return]

    # -- slot plumbing ------------------------------------------------------

    def _alloc_slot_locked(self) -> int:
        if self._free:
            return self._free.pop()
        while self._map:
            _, (slot, _) = self._map.popitem(last=False)
            self._free.append(slot)
            self._stats.evictions += 1
            return self._free.pop()
        raise TierCapacityError(f"NVMe tier {self.name!r} has no slots")

    def _write_slot(self, slot: int, data: memoryview) -> None:
        n = len(data)
        buf = self._buffer()
        buf.write(data, 0)
        length = round_up(n, ALIGNMENT)
        os.pwrite(self._fd, buf.view()[:length], slot * self.block_bytes)

    def _read_slot(self, slot: int, nbytes: int) -> bytes:
        length = round_up(min(nbytes, self.block_bytes), ALIGNMENT)
        buf = self._buffer()
        view = buf.view()[:length]
        got = os.preadv(self._fd, [view], slot * self.block_bytes)
        if got < nbytes:
            raise OSError(
                f"short read from {self.path} slot {slot}: wanted {nbytes}, got {got}"
            )
        return bytes(view[:nbytes])

    def _put(self, block_id: str, data: memoryview) -> None:
        n = len(data)
        if n > self.block_bytes:
            raise TierCapacityError(
                f"payload {n} bytes exceeds NVMe block size {self.block_bytes}; "
                "raise tier.block_size_kib"
            )
        with self._lock:
            entry = self._map.pop(block_id, None)
            if entry is not None:
                slot, _ = entry
                self._free.append(slot)
            slot = self._alloc_slot_locked()
            self._map[block_id] = (slot, n)
        self._write_slot(slot, data)

    def _get(self, block_id: str) -> bytes | None:
        with self._lock:
            entry = self._map.get(block_id)
            if entry is None:
                return None
            self._map.move_to_end(block_id)
            slot, nbytes = entry
        return self._read_slot(slot, nbytes)

    def _contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._map

    def _evict(self, block_id: str) -> bool:
        with self._lock:
            entry = self._map.pop(block_id, None)
            if entry is None:
                return False
            self._free.append(entry[0])
            self._stats.evictions += 1
            return True

    def evict_lru(self, count: int = 1) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        for _ in range(count):
            with self._lock:
                if not self._map:
                    break
                block_id, (slot, nbytes) = self._map.popitem(last=False)
                self._stats.evictions += 1
            try:
                data = self._read_slot(slot, nbytes)
                out.append((block_id, data))
            finally:
                with self._lock:
                    self._free.append(slot)
        return out

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return len(self._map) * self.block_bytes

    # -- async-ish batch path ----------------------------------------------

    def _pool(self) -> ThreadPoolExecutor | None:
        if self.queue_depth <= 1:
            return None
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.queue_depth, thread_name_prefix=f"{self.name}-io"
                )
            return self._executor

    def put_many(
        self, items: Sequence[tuple[str, memoryview | bytes]]
    ) -> None:
        pool = self._pool()
        if pool is None:
            for block_id, data in items:
                self.put(block_id, data)
            return
        list(pool.map(lambda item: self.put(item[0], item[1]), items))

    def get_many(self, block_ids: Sequence[str]) -> list[bytes | None]:
        pool = self._pool()
        if pool is None:
            return [self.get(block_id) for block_id in block_ids]
        return list(pool.map(self.get, block_ids))

    # -- lifecycle ----------------------------------------------------------

    def _close(self) -> None:
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
        buf = getattr(self._scratch, "buf", None)
        if buf is not None:
            buf.close()
            self._scratch.buf = None
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1

    def sync(self) -> None:
        if self._fd >= 0:
            os.fsync(self._fd)

    def debug_state(self) -> dict[str, object]:
        with self._lock:
            return {
                "path": str(self.path),
                "io_mode": self.io_mode,
                "blocks": len(self._map),
                "free_slots": len(self._free),
                "slots": self._slots,
                "block_bytes": self.block_bytes,
            }

    # convenience for microbenchmark timing
    def timed_put(self, block_id: str, data: bytes) -> int:
        t0 = time.perf_counter_ns()
        self.put(block_id, data)
        return time.perf_counter_ns() - t0
