"""NVMe-oF-style remote tier over TCP with a pipelined request protocol.

The tier is a client for ``python -m kvoffloadbench.target.server``. It keeps
``queue_depth`` sockets per process/thread and pipelines batches: all READ or
WRITE frames for a batch are sent before any response is consumed. That makes
the queue-depth axis in the microbenchmark a real concurrency knob rather than
a label.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from kvoffloadbench.tiers.base import KVTier, TierError
from kvoffloadbench.tiers.wire import (
    OP_CONTAINS,
    OP_EVICT,
    OP_GET,
    OP_PING,
    OP_PUT,
    OP_STATS,
    RESPONSE_HEADER,
    STATUS_NOT_FOUND,
    STATUS_OK,
    encode_request,
)


class _Connection:
    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            with contextlib.suppress(OSError):
                self.sock.close()
            self.sock = None

    def _require(self) -> socket.socket:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        return self.sock

    def send_raw(self, data: bytes | bytearray | memoryview) -> None:
        self._require().sendall(data)

    def recv_exact(self, n: int) -> bytes:
        sock = self._require()
        chunks = bytearray()
        while len(chunks) < n:
            part = sock.recv(n - len(chunks))
            if not part:
                raise ConnectionError("remote target closed the connection")
            chunks.extend(part)
        return bytes(chunks)

    def recv_response(self) -> tuple[int, bytes]:
        header = self.recv_exact(RESPONSE_HEADER.size)
        status, length = RESPONSE_HEADER.unpack(header)
        payload = self.recv_exact(length) if length else b""
        return status, payload

    def request(self, op: int, block_id: str = "", payload: bytes = b"") -> tuple[int, bytes]:
        self.send_raw(encode_request(op, block_id, payload))
        return self.recv_response()


class RemoteTCPTier(KVTier):
    kind = "remote_tcp"

    def __init__(
        self,
        name: str = "remote_tcp",
        endpoint: str = "127.0.0.1:4420",
        capacity_bytes: int = 4 * 1024 * 1024 * 1024,
        block_bytes: int = 64 * 1024,
        queue_depth: int = 8,
        direct_io: bool = False,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(name, capacity_bytes, block_bytes, queue_depth, direct_io)
        host, _, port = endpoint.rpartition(":")
        if not host or not port.isdigit():
            raise TierError(f"invalid remote endpoint {endpoint!r}; expected host:port")
        self.host = host
        self.port = int(port)
        self.timeout_s = timeout_s
        self.io_mode = "tcp"
        self._stats.io_mode = "tcp"
        self._local = threading.local()
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    # -- connections --------------------------------------------------------

    def _conn(self) -> _Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _Connection(self.host, self.port, self.timeout_s)
            self._local.conn = conn
        if conn.sock is None:
            conn.connect()
        return conn  # type: ignore[no-any-return]

    def _reset_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()

    def _request(self, op: int, block_id: str = "", payload: bytes = b"") -> tuple[int, bytes]:
        try:
            return self._conn().request(op, block_id, payload)
        except (OSError, ConnectionError):
            self._reset_conn()
            return self._conn().request(op, block_id, payload)

    def _pool(self) -> ThreadPoolExecutor | None:
        if self.queue_depth <= 1:
            return None
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.queue_depth, thread_name_prefix=f"{self.name}-net"
                )
            return self._executor

    # -- single-op hooks ----------------------------------------------------

    def _put(self, block_id: str, data: memoryview) -> None:
        status, payload = self._request(OP_PUT, block_id, bytes(data))
        if status != STATUS_OK:
            raise TierError(
                f"remote PUT {block_id!r} failed: status={status} {payload.decode(errors='replace')}"
            )

    def _get(self, block_id: str) -> bytes | None:
        status, payload = self._request(OP_GET, block_id)
        if status == STATUS_NOT_FOUND:
            return None
        if status != STATUS_OK:
            raise TierError(
                f"remote GET {block_id!r} failed: status={status} "
                f"{payload.decode(errors='replace')}"
            )
        return payload

    def _contains(self, block_id: str) -> bool:
        status, payload = self._request(OP_CONTAINS, block_id)
        return status == STATUS_OK and payload == b"\x01"

    def _evict(self, block_id: str) -> bool:
        status, payload = self._request(OP_EVICT, block_id)
        return status == STATUS_OK and payload == b"\x01"

    # -- pipelined batch path ----------------------------------------------

    def _split(self, items: Sequence[str]) -> list[Sequence[str]]:
        n = min(self.queue_depth, len(items))
        if n <= 1:
            return [items]
        size = (len(items) + n - 1) // n
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _batch_get(self, ids: Sequence[str]) -> list[bytes | None]:
        conn = self._conn()
        t0 = time.perf_counter_ns()
        try:
            conn.send_raw(b"".join(encode_request(OP_GET, block_id) for block_id in ids))
            out: list[bytes | None] = []
            for block_id in ids:
                status, payload = conn.recv_response()
                dt = time.perf_counter_ns() - t0
                if status == STATUS_NOT_FOUND:
                    self._record_read(None, dt)
                    out.append(None)
                elif status == STATUS_OK:
                    self._record_read(len(payload), dt)
                    out.append(payload)
                else:
                    self._record_read(None, dt)
                    raise TierError(f"remote GET {block_id!r} failed: {status}")
            return out
        except (OSError, ConnectionError):
            self._reset_conn()
            raise

    def _batch_put(self, items: Sequence[tuple[str, bytes]]) -> None:
        conn = self._conn()
        t0 = time.perf_counter_ns()
        try:
            buf = bytearray()
            for block_id, data in items:
                buf.extend(encode_request(OP_PUT, block_id, data))
            conn.send_raw(buf)
            for block_id, data in items:
                status, _payload = conn.recv_response()
                dt = time.perf_counter_ns() - t0
                self._record_write(len(data), dt)
                if status != STATUS_OK:
                    raise TierError(f"remote PUT {block_id!r} failed: {status}")
        except (OSError, ConnectionError):
            self._reset_conn()
            raise

    def get_many(self, block_ids: list[str]) -> list[bytes | None]:
        pool = self._pool()
        if pool is None or len(block_ids) <= 1:
            return [self.get(block_id) for block_id in block_ids]
        chunks = self._split(block_ids)
        futures = [pool.submit(self._batch_get, chunk) for chunk in chunks]
        results: list[bytes | None] = []
        for future in futures:
            results.extend(future.result())
        return results

    def put_many(self, items: list[tuple[str, memoryview | bytes]]) -> None:
        pool = self._pool()
        payload = [(block_id, bytes(data)) for block_id, data in items]
        if pool is None or len(payload) <= 1:
            for block_id, data in payload:
                self.put(block_id, data)
            return
        chunks = self._split_put(payload)
        futures = [pool.submit(self._batch_put, chunk) for chunk in chunks]
        for future in futures:
            future.result()

    def _split_put(
        self, items: Sequence[tuple[str, bytes]]
    ) -> list[Sequence[tuple[str, bytes]]]:
        n = min(self.queue_depth, len(items))
        if n <= 1:
            return [items]
        size = (len(items) + n - 1) // n
        return [items[i : i + size] for i in range(0, len(items), size)]

    # -- extras -------------------------------------------------------------

    def ping(self) -> bool:
        status, payload = self._request(OP_PING)
        return status == STATUS_OK and payload == b"pong"

    def server_stats(self) -> dict[str, object]:
        import json

        status, payload = self._request(OP_STATS)
        if status != STATUS_OK:
            raise TierError(f"remote STATS failed with status {status}")
        return json.loads(payload.decode())  # type: ignore[no-any-return]

    @property
    def resident_bytes(self) -> int:
        try:
            return int(self.server_stats().get("resident_bytes", 0))
        except (TierError, OSError, ValueError):
            return 0

    def _close(self) -> None:
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
        self._reset_conn()
