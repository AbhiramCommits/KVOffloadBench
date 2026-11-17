"""Core tier abstraction, statistics, and exceptions."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from kvoffloadbench.metrics import LatencyReservoir


class TierError(RuntimeError):
    """Base error for tier operations."""


class TierUnavailable(TierError):
    """Raised when a backend cannot be used on this host (missing device, etc.)."""


class TierCapacityError(TierError):
    """Raised when a payload can never fit in a tier."""


@dataclass
class TierStats:
    """Counters and latency samples for one tier instance."""

    tier: str
    io_mode: str = "buffered"
    bytes_written: int = 0
    bytes_read: int = 0
    puts: int = 0
    gets: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    write_latency: LatencyReservoir = field(default_factory=LatencyReservoir)
    read_latency: LatencyReservoir = field(default_factory=LatencyReservoir)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "io_mode": self.io_mode,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "puts": self.puts,
            "gets": self.gets,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "errors": self.errors,
            "write_latency_us": self.write_latency.summary_us(),
            "read_latency_us": self.read_latency.summary_us(),
        }


class KVTier(ABC):
    """A block store for KV-cache chunks.

    Subclasses implement ``_put`` / ``_get`` / ``_contains`` / ``_evict``; the
    public wrappers time every operation and maintain statistics.
    """

    kind: str = "base"

    def __init__(
        self,
        name: str,
        capacity_bytes: int,
        block_bytes: int,
        queue_depth: int = 8,
        direct_io: bool = True,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if block_bytes <= 0:
            raise ValueError("block_bytes must be positive")
        if block_bytes % 4096:
            raise ValueError("block_bytes must be a multiple of 4096")
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.block_bytes = block_bytes
        self.queue_depth = max(1, int(queue_depth))
        self.direct_io_requested = direct_io
        self.io_mode = "buffered"
        self._stats = TierStats(tier=name, io_mode="buffered")
        self._lock = threading.RLock()
        self._closed = False

    # -- public API ---------------------------------------------------------

    def put(self, block_id: str, data: memoryview | bytes | bytearray) -> None:
        self._ensure_open()
        view = data if isinstance(data, memoryview) else memoryview(data)
        t0 = time.perf_counter_ns()
        try:
            self._put(block_id, view)
        except TierCapacityError:
            with self._lock:
                self._stats.errors += 1
            raise
        dt = time.perf_counter_ns() - t0
        with self._lock:
            self._stats.puts += 1
            self._stats.bytes_written += len(view)
            self._stats.write_latency.add_ns(dt)

    def get(self, block_id: str) -> bytes | None:
        self._ensure_open()
        t0 = time.perf_counter_ns()
        data = self._get(block_id)
        dt = time.perf_counter_ns() - t0
        with self._lock:
            self._stats.gets += 1
            if data is None:
                self._stats.misses += 1
            else:
                self._stats.hits += 1
                self._stats.bytes_read += len(data)
            self._stats.read_latency.add_ns(dt)
        return data

    def put_many(self, items: Sequence[tuple[str, memoryview | bytes]]) -> None:
        for block_id, data in items:
            self.put(block_id, data)

    def get_many(self, block_ids: Sequence[str]) -> list[bytes | None]:
        return [self.get(block_id) for block_id in block_ids]

    def contains(self, block_id: str) -> bool:
        self._ensure_open()
        return self._contains(block_id)

    def evict(self, block_id: str) -> bool:
        self._ensure_open()
        return self._evict(block_id)

    def evict_lru(self, count: int = 1) -> list[tuple[str, bytes]]:
        """Remove and return up to ``count`` least-recently-used entries.

        Only tiers that can act as a demotion source implement this.
        """
        raise TierError(f"tier {self.name!r} does not support LRU eviction export")

    @property
    @abstractmethod
    def resident_bytes(self) -> int:
        """Bytes currently occupying the tier (slot-accounted)."""

    def stats(self) -> TierStats:
        with self._lock:
            clone = TierStats(
                tier=self._stats.tier,
                io_mode=self.io_mode,
                bytes_written=self._stats.bytes_written,
                bytes_read=self._stats.bytes_read,
                puts=self._stats.puts,
                gets=self._stats.gets,
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                errors=self._stats.errors,
                write_latency=LatencyReservoir(capacity=self._stats.write_latency.capacity),
                read_latency=LatencyReservoir(capacity=self._stats.read_latency.capacity),
            )
            clone.write_latency.extend_ns(self._stats.write_latency.values_ns())
            clone.read_latency.extend_ns(self._stats.read_latency.values_ns())
            return clone

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = TierStats(tier=self.name, io_mode=self.io_mode)

    def stats_dict(self) -> dict[str, object]:
        d = self.stats().to_dict()
        d["resident_bytes"] = self.resident_bytes
        d["capacity_bytes"] = self.capacity_bytes
        d["block_bytes"] = self.block_bytes
        d["queue_depth"] = self.queue_depth
        d["direct_io_requested"] = self.direct_io_requested
        return d

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._close()

    def _close(self) -> None:  # noqa: B027 - optional hook, not every tier owns resources
        """Release resources; default no-op."""

    def __enter__(self) -> KVTier:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- hooks --------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise TierError(f"tier {self.name!r} is closed")

    def _record_write(self, nbytes: int, latency_ns: int) -> None:
        with self._lock:
            self._stats.puts += 1
            self._stats.bytes_written += nbytes
            self._stats.write_latency.add_ns(latency_ns)

    def _record_read(self, nbytes: int | None, latency_ns: int) -> None:
        with self._lock:
            self._stats.gets += 1
            if nbytes is None:
                self._stats.misses += 1
            else:
                self._stats.hits += 1
                self._stats.bytes_read += nbytes
            self._stats.read_latency.add_ns(latency_ns)

    @abstractmethod
    def _put(self, block_id: str, data: memoryview) -> None: ...

    @abstractmethod
    def _get(self, block_id: str) -> bytes | None: ...

    @abstractmethod
    def _contains(self, block_id: str) -> bool: ...

    @abstractmethod
    def _evict(self, block_id: str) -> bool: ...


def split_batches(items: Sequence[str], depth: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), max(1, depth)):
        yield items[i : i + max(1, depth)]
