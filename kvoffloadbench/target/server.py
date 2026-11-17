"""NVMe-oF-style TCP block target.

Run it with::

    python -m kvoffloadbench.target.server --port 4420 \
        --path results/cache/target/server.bin --capacity-mib 4096

The target is deliberately small and self-contained: length-prefixed binary
frames (``kvoffloadbench.tiers.wire``), a file-backed extent allocator, and
pipelined request processing. It speaks a real storage access protocol over a
real TCP socket -- only the RDMA transport is substituted.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from pathlib import Path

from kvoffloadbench.target.store import FileExtentStore
from kvoffloadbench.tiers.wire import (
    MAX_BLOCK_ID,
    MAX_PAYLOAD,
    OP_CONTAINS,
    OP_EVICT,
    OP_GET,
    OP_PING,
    OP_PUT,
    OP_STATS,
    REQUEST_HEADER,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    encode_response,
)

log = logging.getLogger("kvoffloadbench.target")


class TargetServer:
    def __init__(
        self,
        store: FileExtentStore,
        host: str = "127.0.0.1",
        port: int = 4420,
        max_inflight: int = 64,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.max_inflight = max(1, max_inflight)
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port, backlog=256
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.port
        return int(self._server.sockets[0].getsockname()[1])

    async def _dispatch(self, op: int, block_id: str, payload: bytes) -> bytes:
        try:
            if op == OP_PING:
                return encode_response(STATUS_OK, b"pong")
            if op == OP_STATS:
                stats = await asyncio.to_thread(self.store.stats)
                return encode_response(STATUS_OK, json.dumps(stats).encode())
            if op == OP_PUT:
                if not block_id or not payload:
                    raise ValueError("PUT requires a block id and a payload")
                await asyncio.to_thread(self.store.put, block_id, payload)
                return encode_response(STATUS_OK)
            if op == OP_GET:
                if not block_id:
                    raise ValueError("GET requires a block id")
                data = await asyncio.to_thread(self.store.get, block_id)
                if data is None:
                    return encode_response(STATUS_NOT_FOUND)
                return encode_response(STATUS_OK, data)
            if op == OP_CONTAINS:
                found = await asyncio.to_thread(self.store.contains, block_id)
                return encode_response(STATUS_OK, b"\x01" if found else b"\x00")
            if op == OP_EVICT:
                removed = await asyncio.to_thread(self.store.evict, block_id)
                return encode_response(STATUS_OK, b"\x01" if removed else b"\x00")
            return encode_response(STATUS_ERROR, f"unknown opcode {op}".encode())
        except Exception as exc:
            log.exception("request failed: op=%s block=%r", op, block_id)
            return encode_response(STATUS_ERROR, f"{type(exc).__name__}: {exc}".encode())

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        log.info("connection from %s", peer)
        queue: asyncio.Queue[asyncio.Future[bytes] | None] = asyncio.Queue(maxsize=self.max_inflight)
        writer_done = asyncio.create_task(self._writer_loop(writer, queue))

        try:
            while True:
                header = await reader.readexactly(REQUEST_HEADER.size)
                op, id_len, payload_len = REQUEST_HEADER.unpack(header)
                if id_len > MAX_BLOCK_ID or payload_len > MAX_PAYLOAD:
                    raise ValueError(f"frame too large: id_len={id_len} payload_len={payload_len}")
                id_bytes = await reader.readexactly(id_len) if id_len else b""
                payload = await reader.readexactly(payload_len) if payload_len else b""
                try:
                    block_id = id_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    block_id = ""
                if op not in (OP_PUT, OP_GET, OP_CONTAINS, OP_EVICT, OP_STATS, OP_PING):
                    fut: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
                    fut.set_result(encode_response(STATUS_ERROR, b"unknown opcode"))
                    await queue.put(fut)
                    continue
                task = asyncio.create_task(self._dispatch(op, block_id, payload))
                await queue.put(task)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            log.exception("connection loop failed for %s", peer)
        finally:
            await queue.put(None)
            try:
                await writer_done
            except Exception:
                log.exception("writer loop failed for %s", peer)
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
            log.info("connection closed: %s", peer)

    async def _writer_loop(
        self, writer: asyncio.StreamWriter, queue: asyncio.Queue[asyncio.Future[bytes] | None]
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            try:
                response = await item
            except Exception as exc:
                log.exception("dispatch task failed")
                response = encode_response(STATUS_ERROR, f"{type(exc).__name__}: {exc}".encode())
            writer.write(response)
            await writer.drain()


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = Path(args.path)
    store = FileExtentStore(
        path,
        capacity_bytes=int(args.capacity_mib) * 1024 * 1024,
        direct_io=not args.no_direct_io,
    )
    server = TargetServer(
        store, host=args.host, port=args.port, max_inflight=args.max_inflight
    )
    await server.start()
    port = server.bound_port
    print(
        f"READY host={args.host} port={port} path={path} "
        f"capacity_mib={args.capacity_mib} io_mode={store.io_mode}",
        flush=True,
    )
    stop = asyncio.Event()

    def _request_stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover - Windows
            signal.signal(sig, lambda *_: _request_stop())

    serve = asyncio.create_task(server.serve_forever())
    await stop.wait()
    serve.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await serve
    await server.close()
    store.close()
    print("STOPPED", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kvoffloadbench.target.server",
        description="NVMe-oF-style TCP block target backed by a preallocated file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4420)
    parser.add_argument("--path", default="results/cache/target/server.bin")
    parser.add_argument("--capacity-mib", type=int, default=4096)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument(
        "--no-direct-io",
        action="store_true",
        help="use buffered file I/O instead of O_DIRECT/F_NOCACHE",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":
    sys.exit(main())
