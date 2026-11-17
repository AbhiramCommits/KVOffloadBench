"""Environment fingerprint captured in every run manifest.

The fingerprint is deliberately cheap (a few subprocess calls plus one small
O_DIRECT probe) and never raises: missing tools or packages become ``None`` /
empty values so that a manifest can always be written, even on a broken host.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import mmap
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PACKAGE_NAMES = (
    "kvoffloadbench",
    "torch",
    "transformers",
    "vllm",
    "lmcache",
    "nixl",
    "numpy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "prometheus-client",
)


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git(repo: Path | None = None) -> dict[str, Any]:
    cwd = str(repo) if repo else str(Path(__file__).resolve().parents[1])
    sha = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(status) if status is not None else None,
    }


def _cpu_model() -> str | None:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            return None
    return platform.processor() or None


def _cpu_counts() -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    physical: int | None = None
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        physical = int(out) if out and out.isdigit() else None
    elif system == "Linux":
        try:
            info = Path("/proc/cpuinfo").read_text()
            ids = {
                (line.split(":")[1].strip(),)
                for line in info.splitlines()
                if line.startswith("physical id")
            }
            cores = sum(
                1 for line in info.splitlines() if line.startswith("cpu cores")
            )
            physical = cores or (len(ids) if ids else None)
        except OSError:
            physical = None
    return physical, logical


def _total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # type: ignore[no-any-return]
    except (ValueError, OSError, AttributeError):
        return None


def _gpus() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "mps_available": False,
        "device": "cpu",
        "gpus": [],
    }
    try:
        import torch

        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        if hasattr(torch.backends, "mps"):
            info["mps_available"] = bool(torch.backends.mps.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
        elif info["mps_available"]:
            info["device"] = "mps"
    except Exception as exc:  # noqa: BLE001 - fingerprint must never fail
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _pip_freeze() -> tuple[str | None, int]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, 0
    if out.returncode != 0:
        return None, 0
    lines = sorted(line.strip() for line in out.stdout.splitlines() if line.strip())
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return digest, len(lines)


def _probe_direct_io() -> dict[str, Any]:
    """Check whether the filesystem honors O_DIRECT (Linux) / F_NOCACHE (macOS)."""
    result: dict[str, Any] = {
        "o_direct_flag": hasattr(os, "O_DIRECT"),
        "supported": False,
        "method": None,
        "detail": "",
    }
    tmpdir = Path(tempfile.gettempdir())
    path = tmpdir / f"kvob-odirect-probe-{os.getpid()}.bin"
    try:
        if hasattr(os, "O_DIRECT"):
            flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT
            fd = os.open(path, flags, 0o600)
            result["method"] = "O_DIRECT"
            try:
                buf = mmap.mmap(-1, 4096)
                buf.write(b"k" * 4096)
                os.pwrite(fd, buf, 0)
                os.fsync(fd)
                os.pread(fd, 4096, 0)
                result["supported"] = True
                result["detail"] = "aligned 4 KiB write/read with O_DIRECT"
                buf.close()
            finally:
                os.close(fd)
        elif platform.system() == "Darwin":
            import fcntl

            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            result["method"] = "F_NOCACHE"
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                os.write(fd, b"k" * 4096)
                os.fsync(fd)
                os.pread(fd, 4096, 0)
                result["supported"] = True
                result["detail"] = (
                    "F_NOCACHE accepted; macOS has no O_DIRECT, so the page cache is "
                    "bypassed on read but writes remain buffered"
                )
            finally:
                os.close(fd)
        else:
            result["detail"] = "platform exposes neither O_DIRECT nor F_NOCACHE"
    except OSError as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    return result


def _nvme_inventory() -> dict[str, Any]:
    devices = sorted(glob.glob("/dev/nvme*n*"))
    env_dev = os.environ.get("KVOB_NVMEOF_DEV")
    return {
        "nvme_devices": devices,
        "nvme_cli": shutil.which("nvme"),
        "nvmeof_dev_env": env_dev,
        "nvmeof_dev_present": bool(env_dev and Path(env_dev).exists()),
    }


def fingerprint(repo: Path | None = None, include_pip_freeze: bool = True) -> dict[str, Any]:
    """Return a JSON-serialisable environment fingerprint."""
    physical, logical = _cpu_counts()
    gpu = _gpus()
    freeze_sha, freeze_count = _pip_freeze() if include_pip_freeze else (None, 0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_model": _cpu_model(),
        "cpu_count_physical": physical,
        "cpu_count_logical": logical,
        "total_ram_bytes": _total_ram_bytes(),
        **gpu,
        "versions": _versions(),
        "pip_freeze_sha256": freeze_sha,
        "pip_package_count": freeze_count,
        "direct_io": _probe_direct_io(),
        **(_nvme_inventory()),
        **_git(repo),
    }


def fingerprint_json(**kwargs: Any) -> str:
    return json.dumps(fingerprint(**kwargs), indent=2, sort_keys=True, default=str)
