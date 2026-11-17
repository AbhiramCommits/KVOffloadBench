"""Server-side block store backed by a preallocated file.

The store uses a 4 KiB-granular extent allocator with coalescing, so the TCP
target can accept any block size from 4 KiB to 1 MiB without allocating one
file per size class. Reads take a shared lock (concurrent queue depth) while
writes are exclusive.
"""

from __future__ import annotations

import fcntl
import os
import platform
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvoffloadbench.metrics import LatencyReservoir
from kvoffloadbench.tiers.aligned import AlignedBuffer, round_up

CHUNK = 4096


class RWLock:
    """Writer-preferring reader/writer lock."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


@dataclass
class _Extent:
    offset_chunks: int
    chunks: int
    length_bytes: int


class FileExtentStore:
    """Block store over a preallocated file or raw device region."""

    def __init__(
        self,
        path: str | Path,
        capacity_bytes: int,
        direct_io: bool = True,
        name: str = "remote_target",
        create: bool = True,
    ) -> None:
        self.path = Path(path)
        self.capacity_bytes = (capacity_bytes // CHUNK) * CHUNK
        if self.capacity_bytes < CHUNK:
            raise ValueError("capacity must be at least 4 KiB")
        self.name = name
        self._total_chunks = self.capacity_bytes // CHUNK
        self._free: list[tuple[int, int]] = [(0, self._total_chunks)]
        self._blocks: OrderedDict[str, _Extent] = OrderedDict()
        self._meta = threading.Lock()
        self._rw = RWLock()
        self._resident = 0
        self._scratch = threading.local()
        self._stats_lock = threading.Lock()
        self.bytes_written = 0
        self.bytes_read = 0
        self.puts = 0
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.write_latency = LatencyReservoir()
        self.read_latency = LatencyReservoir()
        self._fd = self._open(create=create)

    # -- backing file -------------------------------------------------------

    def _open(self, create: bool) -> int:
        flags = os.O_RDWR
        if create:
            flags |= os.O_CREAT
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.io_mode = "buffered"
        if os.getenv("KVOB_TARGET_DIRECT_IO", "1") not in ("0", "false", "no"):
            if hasattr(os, "O_DIRECT"):
                try:
                    fd = os.open(self.path, flags | os.O_DIRECT, 0o600)
                    if create:
                        os.ftruncate(fd, self.capacity_bytes)
                    self.io_mode = "odirect"
                    return fd
                except OSError:
                    pass
            elif platform.system() == "Darwin":
                fd = os.open(self.path, flags, 0o600)
                try:
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    if create:
                        os.ftruncate(fd, self.capacity_bytes)
                    self.io_mode = "fnocache"
                    return fd
                except OSError:
                    os.close(fd)
        fd = os.open(self.path, flags, 0o600)
        if create:
            os.ftruncate(fd, self.capacity_bytes)
        return fd

    # -- aligned scratch ----------------------------------------------------

    def _buffer(self, size: int) -> AlignedBuffer:
        buf = getattr(self._scratch, "buf", None)
        need = round_up(max(size, CHUNK), 4096)
        if buf is None or buf.size < need:
            if buf is not None:
                buf.close()
            buf = AlignedBuffer(need, 4096)
            self._scratch.buf = buf
        return buf  # type: ignore[no-any-return]

    def _read_at(self, offset: int, nbytes: int) -> bytes:
        length = round_up(nbytes, CHUNK) if self.io_mode == "odirect" else nbytes
        buf = self._buffer(length)
        view = buf.view()[:length]
        got = 0
        while got < length:
            n = os.preadv(self._fd, [view[got:]], offset + got)
            if n <= 0:
                raise OSError(f"short read at offset {offset}: got {got}/{length}")
            got += n
        return bytes(view[:nbytes])

    def _write_at(self, offset: int, data: bytes) -> None:
        length = round_up(len(data), CHUNK) if self.io_mode == "odirect" else len(data)
        buf = self._buffer(length)
        buf.write(data)
        view = buf.view()[:length]
        written = 0
        while written < length:
            n = os.pwrite(self._fd, view[written:], offset + written)
            written += n

    # -- allocator ----------------------------------------------------------

    def _alloc_locked(self, chunks: int) -> int | None:
        for idx, (start, length) in enumerate(self._free):
            if length >= chunks:
                if length == chunks:
                    del self._free[idx]
                else:
                    self._free[idx] = (start + chunks, length - chunks)
                return start
        return None

    def _free_locked(self, start: int, chunks: int) -> None:
        merged_start, merged_chunks = start, chunks
        new_free: list[tuple[int, int]] = []
        for s, n in self._free:
            if s + n == merged_start:
                merged_start, merged_chunks = s, n + merged_chunks
            elif merged_start + merged_chunks == s:
                merged_chunks += n
            else:
                new_free.append((s, n))
        new_free.append((merged_start, merged_chunks))
        new_free.sort()
        # second pass in case the insertion bridged two runs
        coalesced: list[tuple[int, int]] = []
        for s, n in new_free:
            if coalesced and coalesced[-1][0] + coalesced[-1][1] == s:
                prev_s, prev_n = coalesced[-1]
                coalesced[-1] = (prev_s, prev_n + n)
            else:
                coalesced.append((s, n))
        self._free = coalesced

    # -- public API ---------------------------------------------------------

    def put(self, block_id: str, data: bytes) -> None:
        if not data:
            raise ValueError("cannot store an empty block")
        chunks = (len(data) + CHUNK - 1) // CHUNK
        if chunks > self._total_chunks:
            raise ValueError(f"block of {len(data)} bytes exceeds target capacity")
        t0 = time.perf_counter_ns()
        with self._rw.write():
            with self._meta:
                old = self._blocks.pop(block_id, None)
                if old is not None:
                    self._resident -= old.length_bytes
                    self._free_locked(old.offset_chunks, old.chunks)
                offset = self._alloc_locked(chunks)
                while offset is None:
                    if not self._blocks:
                        raise ValueError("target is full and holds no evictable blocks")
                    _, victim = self._blocks.popitem(last=False)
                    self._resident -= victim.length_bytes
                    self.evictions += 1
                    self._free_locked(victim.offset_chunks, victim.chunks)
                    offset = self._alloc_locked(chunks)
                self._blocks[block_id] = _Extent(offset, chunks, len(data))
                self._resident += len(data)
            self._write_at(offset * CHUNK, data)
        dt = time.perf_counter_ns() - t0
        with self._stats_lock:
            self.puts += 1
            self.bytes_written += len(data)
            self.write_latency.add_ns(dt)

    def get(self, block_id: str) -> bytes | None:
        t0 = time.perf_counter_ns()
        with self._rw.read():
            with self._meta:
                extent = self._blocks.get(block_id)
                if extent is not None:
                    self._blocks.move_to_end(block_id)
            data = (
                self._read_at(extent.offset_chunks * CHUNK, extent.length_bytes)
                if extent is not None
                else None
            )
        dt = time.perf_counter_ns() - t0
        with self._stats_lock:
            self.gets += 1
            if data is None:
                self.misses += 1
            else:
                self.hits += 1
                self.bytes_read += len(data)
            self.read_latency.add_ns(dt)
        return data

    def contains(self, block_id: str) -> bool:
        with self._meta:
            return block_id in self._blocks

    def evict(self, block_id: str) -> bool:
        with self._rw.write(), self._meta:
            extent = self._blocks.pop(block_id, None)
            if extent is None:
                return False
            self._resident -= extent.length_bytes
            self.evictions += 1
            self._free_locked(extent.offset_chunks, extent.chunks)
            return True

    def stats(self) -> dict[str, Any]:
        with self._meta:
            blocks = len(self._blocks)
            free_chunks = sum(n for _, n in self._free)
            fragments = len(self._free)
        with self._stats_lock:
            return {
                "backend": "file",
                "path": str(self.path),
                "io_mode": self.io_mode,
                "capacity_bytes": self.capacity_bytes,
                "resident_bytes": self._resident,
                "blocks": blocks,
                "free_chunks": free_chunks,
                "free_fragments": fragments,
                "puts": self.puts,
                "gets": self.gets,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "bytes_written": self.bytes_written,
                "bytes_read": self.bytes_read,
                "write_latency_us": self.write_latency.summary_us(),
                "read_latency_us": self.read_latency.summary_us(),
            }

    def close(self) -> None:
        buf = getattr(self._scratch, "buf", None)
        if buf is not None:
            buf.close()
            self._scratch.buf = None
        if self._fd >= 0:
            with suppress(OSError):
                os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> FileExtentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
