"""NVMe-oF block-device tier.

This is the real network block path: ``KVOB_NVMEOF_DEV`` points at a namespace
produced by ``nvme connect`` (TCP or RDMA/RoCE), and this tier performs aligned
``O_DIRECT`` reads and writes against the raw block device.

It is destructive by design: the benchmark owns the namespace starting at
offset 0 for ``capacity_mib``. Point it only at a scratch namespace created by
``scripts/setup_nvmeof_tcp.sh``.
"""

from __future__ import annotations

import logging
import os
import platform
import stat
from pathlib import Path

from kvoffloadbench.tiers.base import TierUnavailable
from kvoffloadbench.tiers.nvme import NVMeTier

log = logging.getLogger(__name__)


def _device_size_bytes(path: Path) -> int | None:
    if platform.system() == "Linux":
        try:
            import fcntl
            import struct

            # BLKGETSIZE64 = _IOR(0x12, 114, size_t)
            with open(path, "rb") as fh:
                buf = fcntl.ioctl(fh.fileno(), 0x80081272, b"\x00" * 8)
            return int(struct.unpack("Q", buf)[0])
        except (OSError, ValueError):
            pass
    try:
        return path.stat().st_size or None
    except OSError:
        return None


class NVMeOFBlockTier(NVMeTier):
    kind = "nvmeof_block"

    def __init__(
        self,
        name: str = "nvmeof_block",
        path: str | Path | None = None,
        device_env: str = "KVOB_NVMEOF_DEV",
        capacity_bytes: int = 1024 * 1024 * 1024,
        block_bytes: int = 64 * 1024,
        queue_depth: int = 8,
        direct_io: bool = True,
    ) -> None:
        device = path or os.environ.get(device_env)
        if not device:
            raise TierUnavailable(
                f"NVMe-oF block tier unavailable: set {device_env} to a device from "
                "`nvme connect` (run scripts/setup_nvmeof_tcp.sh) or set tier.path"
            )
        dev_path = Path(device)
        if not dev_path.exists():
            raise TierUnavailable(
                f"NVMe-oF device {dev_path} does not exist; connect the namespace first "
                "(scripts/setup_nvmeof_tcp.sh)"
            )
        try:
            mode = dev_path.stat().st_mode
        except OSError as exc:
            raise TierUnavailable(f"cannot stat NVMe-oF device {dev_path}: {exc}") from exc
        if platform.system() == "Linux" and not stat.S_ISBLK(mode):
            raise TierUnavailable(
                f"{dev_path} is not a block device; KVOB_NVMEOF_DEV must point at "
                "/dev/nvmeXnY from `nvme connect`"
            )
        size = _device_size_bytes(dev_path)
        if size is not None and capacity_bytes > size:
            log.warning(
                "requested NVMe-oF capacity %d MiB exceeds device %s (%d MiB); clamping",
                capacity_bytes // (1024 * 1024),
                dev_path,
                size // (1024 * 1024),
            )
            capacity_bytes = size
        capacity_bytes = (capacity_bytes // block_bytes) * block_bytes
        if capacity_bytes <= 0:
            raise TierUnavailable(
                f"NVMe-oF device {dev_path} is too small for a {block_bytes}-byte block"
            )
        self.device_path = dev_path
        log.warning(
            "NVMe-oF tier will overwrite the first %d MiB of %s",
            capacity_bytes // (1024 * 1024),
            dev_path,
        )
        super().__init__(
            name=name,
            path=dev_path,
            capacity_bytes=capacity_bytes,
            block_bytes=block_bytes,
            queue_depth=queue_depth,
            direct_io=direct_io,
        )

    def _open_backing(self) -> int:
        flags = os.O_RDWR
        if self.direct_io_requested:
            if hasattr(os, "O_DIRECT"):
                try:
                    fd = os.open(self.path, flags | os.O_DIRECT)
                    self.io_mode = "nvmeof-odirect"
                    self._stats.io_mode = self.io_mode
                    return fd
                except OSError as exc:
                    log.warning("O_DIRECT on %s failed: %s", self.path, exc)
            elif platform.system() == "Darwin":
                raise TierUnavailable(
                    "NVMe-oF block I/O needs a Linux nvme-tcp/rdma namespace; macOS has "
                    "no block-device path. Use the remote_tcp tier here."
                )
        fd = os.open(self.path, flags)
        self.io_mode = "nvmeof-buffered"
        self._stats.io_mode = self.io_mode
        return fd

    def _close(self) -> None:
        super()._close()

    def debug_state(self) -> dict[str, object]:
        state = super().debug_state()
        state["device"] = str(self.device_path)
        return state
