"""Page-aligned buffers for O_DIRECT-capable I/O."""

from __future__ import annotations

import contextlib
import mmap
from typing import Any


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class AlignedBuffer:
    """Anonymous ``mmap`` buffer whose start address is page aligned.

    ``mmap`` allocations are page aligned on Linux and macOS, which satisfies
    the alignment requirements of ``O_DIRECT`` (logical block size) and NVMe
    namespace I/O.
    """

    def __init__(self, size: int, alignment: int = 4096) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.alignment = alignment
        self.size = round_up(size, alignment)
        self._mm = mmap.mmap(-1, self.size)
        self._mv = memoryview(self._mm)

    def view(self) -> memoryview:
        return self._mv

    def write(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        n = len(data)
        if offset + n > self.size:
            raise ValueError("data does not fit in aligned buffer")
        self._mv[offset : offset + n] = data

    def zero(self, start: int = 0, end: int | None = None) -> None:
        self._mv[start : self.size if end is None else end] = b"\x00" * (
            (self.size if end is None else end) - start
        )

    def bytes(self, n: int | None = None) -> bytes:
        return bytes(self._mv[: self.size if n is None else n])

    def close(self) -> None:
        with contextlib.suppress(ValueError, BufferError):
            self._mv.release()
        with contextlib.suppress(ValueError, BufferError):
            self._mm.close()

    def __len__(self) -> int:
        return self.size

    def __enter__(self) -> AlignedBuffer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
