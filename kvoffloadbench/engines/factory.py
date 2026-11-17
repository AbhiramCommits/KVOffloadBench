"""Engine factory: choose the real vLLM/NIXL engine or the calibrated emulator.

Selection rule, logged loudly and recorded in every result row:

1. ``engine.mode == "emulated"`` -> emulated.
2. ``engine.mode == "real"``    -> requested connector; raises if unavailable.
3. ``engine.mode == "auto"``    -> vLLM+NIXL when requested and importable,
   else vLLM+LMCache when CUDA and vLLM are present, else emulated.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging

from kvoffloadbench.config import EngineConfig, TierConfig
from kvoffloadbench.engines.base import Engine, EngineUnavailable
from kvoffloadbench.engines.emulated import EmulatedEngine
from kvoffloadbench.tiers.tiered_store import TieredKVStore

log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def availability() -> dict[str, bool]:
    return {
        "cuda": _cuda_available(),
        "vllm": _has("vllm"),
        "lmcache": _has("lmcache"),
        "nixl": _has("nixl"),
    }


async def build_engine(
    cfg: EngineConfig,
    store: TieredKVStore | None,
    run_name: str = "run",
    tiers: list[TierConfig] | None = None,
) -> Engine:
    tiers = tiers or []
    avail = availability()
    reason_bits = [f"{k}={v}" for k, v in avail.items()]

    engine: Engine
    if cfg.mode == "emulated":
        engine = await asyncio.to_thread(
            EmulatedEngine,
            cfg,
            store,
            run_name,
            cfg.calibration_path,
            "engine.mode='emulated' requested; tier I/O remains real",
        )
        log.warning(
            "ENGINE DECISION run=%s mode=emulated (%s)", run_name, ", ".join(reason_bits)
        )
        return engine

    if cfg.mode == "real":
        engine = await _build_real(cfg, run_name, tiers)
        log.warning("ENGINE DECISION run=%s mode=real engine=%s", run_name, engine.name)
        return engine

    # auto
    if avail["vllm"] and avail["cuda"]:
        try:
            engine = await _build_real(cfg, run_name, tiers)
            log.warning(
                "ENGINE DECISION run=%s mode=real engine=%s (%s)",
                run_name,
                engine.name,
                ", ".join(reason_bits),
            )
            return engine
        except EngineUnavailable as exc:
            log.warning(
                "real engine unavailable for run=%s: %s; degrading to calibrated emulation",
                run_name,
                exc,
            )
            reason = f"real engine unavailable ({exc}); using calibrated emulation"
    else:
        missing = [k for k in ("cuda", "vllm") if not avail[k]]
        reason = (
            f"auto mode: missing {', '.join(missing)}; using calibrated emulation "
            f"({', '.join(reason_bits)})"
        )
    engine = await asyncio.to_thread(
        EmulatedEngine, cfg, store, run_name, cfg.calibration_path, reason
    )
    log.warning("ENGINE DECISION run=%s mode=emulated: %s", run_name, reason)
    return engine


async def _build_real(
    cfg: EngineConfig, run_name: str, tiers: list[TierConfig]
) -> Engine:
    if cfg.connector == "nixl":
        from kvoffloadbench.engines.nixl import NIXLEngine

        engine: Engine = NIXLEngine(cfg, tiers, run_name)
    else:
        from kvoffloadbench.engines.vllm_lmcache import VLLMEngine

        engine = VLLMEngine(cfg, tiers, run_name)
    await engine.start()  # type: ignore[attr-defined]
    return engine
