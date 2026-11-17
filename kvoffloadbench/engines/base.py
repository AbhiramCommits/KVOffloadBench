"""Engine abstraction shared by the vLLM/LMCache, NIXL, and emulated engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class EngineUnavailable(RuntimeError):
    """Raised when an engine cannot be constructed on this host."""


@dataclass(frozen=True)
class GenerationRequest:
    """One request to serve.

    ``prompt_tokens`` is the target prompt length. The emulated engine
    tokenizes ``text`` for real and truncates/pads to this length so that the
    sweep's context axis is honoured exactly.
    """

    request_id: str
    text: str
    max_tokens: int = 64
    temperature: float = 0.0
    prompt_tokens: int | None = None
    prefix_group: str = ""
    arrival_ns: int = 0
    seed: int = 0
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    timestamp_ns: int
    is_first: bool


@runtime_checkable
class Engine(Protocol):
    mode: str
    name: str
    selection_reason: str

    def generate(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]: ...

    def stats(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def event_is_goodput(ttft_ms: float, tpot_ms: float, slo_ttft_ms: float, slo_tpot_ms: float) -> bool:
    """Single source of truth for the goodput boundary (inclusive)."""
    return ttft_ms <= slo_ttft_ms and tpot_ms <= slo_tpot_ms
