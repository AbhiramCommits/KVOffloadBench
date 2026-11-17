"""Helpers to start and stop the TCP target from benchmarks and tests."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class TargetHandle:
    endpoint: str
    path: Path
    process: subprocess.Popen[str] | None = None
    started: bool = False

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def __enter__(self) -> TargetHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_target(
    endpoint: str,
    path: str | Path,
    capacity_mib: int,
    direct_io: bool = True,
    wait_s: float = 20.0,
) -> TargetHandle:
    host, _, port_s = endpoint.rpartition(":")
    port = int(port_s)
    handle = TargetHandle(endpoint=endpoint, path=Path(path))
    if port_open(host, port):
        log.info("reusing TCP target already listening on %s", endpoint)
        return handle
    handle.path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kvoffloadbench.target.server",
        "--host",
        host,
        "--port",
        str(port),
        "--path",
        str(path),
        "--capacity-mib",
        str(capacity_mib),
    ]
    if not direct_io:
        cmd.append("--no-direct-io")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    handle.process = proc
    handle.started = True
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if line.startswith("READY"):
            log.info("target ready: %s", line.strip())
            return handle
        if proc.poll() is not None:
            raise RuntimeError(
                f"target exited with code {proc.returncode}: {line.strip()}"
            )
    proc.terminate()
    raise RuntimeError(f"target did not become ready on {endpoint} within {wait_s}s")
