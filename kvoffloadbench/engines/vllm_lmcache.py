"""vLLM + LMCache engine adapter.

The adapter builds ``AsyncLLMEngine`` (vLLM V1 preferred) with the LMCache KV
connector, writes an LMCache YAML derived from the benchmark's tier configs,
exports ``LMCACHE_CONFIG_FILE`` / ``VLLM_USE_V1``, and converts vLLM's output
stream into :class:`TokenEvent` timestamps.

This module imports cleanly without vLLM installed; construction raises
:class:`EngineUnavailable` so the factory can degrade to the emulated engine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

from kvoffloadbench.config import EngineConfig, TierConfig
from kvoffloadbench.engines.base import (
    EngineUnavailable,
    GenerationRequest,
    TokenEvent,
)

log = logging.getLogger(__name__)


def lmcache_yaml(tiers: list[TierConfig], chunk_size: int = 256) -> dict[str, Any]:
    """Translate tier configs into an LMCache configuration mapping."""
    cfg: dict[str, Any] = {"chunk_size": chunk_size}
    for tier in tiers:
        if tier.kind in ("hbm", "dram"):
            cfg.setdefault("local_cpu", True)
            cfg["max_local_cpu_size"] = tier.capacity_mib / 1024.0
        elif tier.kind == "nvme":
            cfg["local_disk"] = str(tier.path or "results/cache/lmcache-disk")
            cfg["max_local_disk_size"] = tier.capacity_mib / 1024.0
        elif tier.kind == "remote_tcp":
            cfg["remote_url"] = tier.endpoint
            cfg["remote_serde"] = str(tier.extra.get("remote_serde", "naive"))
        elif tier.kind == "nvmeof_block":
            cfg["local_disk"] = str(tier.path or os.environ.get("KVOB_NVMEOF_DEV", ""))
            cfg["max_local_disk_size"] = tier.capacity_mib / 1024.0
    return cfg


def write_lmcache_config(
    tiers: list[TierConfig], path: str | Path, chunk_size: int = 256
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(lmcache_yaml(tiers, chunk_size), sort_keys=True))
    return p


def _import_vllm() -> tuple[Any, Any, Any]:
    try:
        import vllm
        from vllm import SamplingParams
        from vllm.config import KVTransferConfig
    except Exception as exc:
        raise EngineUnavailable(f"vLLM is not importable: {type(exc).__name__}: {exc}") from exc
    return vllm, SamplingParams, KVTransferConfig


class VLLMEngine:
    """Real serving engine (vLLM + LMCache or NIXL connector)."""

    mode = "real"
    name = "vllm"

    def __init__(
        self,
        cfg: EngineConfig,
        tiers: list[TierConfig],
        run_name: str = "run",
        connector: str = "LMCacheConnectorV1",
    ) -> None:
        self.cfg = cfg
        self.tiers = tiers
        self.run_name = run_name
        self.connector = connector
        self.selection_reason = f"CUDA + vLLM available; connector={connector}"
        self._engine: Any = None
        self._SamplingParams: Any = None
        self._requests = 0
        self._tokens = 0
        self._cached_tokens = 0
        self._prompt_tokens = 0
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        import torch

        if not torch.cuda.is_available():
            raise EngineUnavailable(
                "vLLM engine requires CUDA; no CUDA device is visible. "
                "Use mode='emulated' or run on a GPU host."
            )
        _vllm, SamplingParams, KVTransferConfig = _import_vllm()
        self._SamplingParams = SamplingParams
        config_dir = Path("results/cache/lmcache")
        config_path = config_dir / f"{self.run_name}.yaml"
        write_lmcache_config(self.tiers, config_path)
        os.environ["LMCACHE_CONFIG_FILE"] = str(config_path)
        os.environ["VLLM_USE_V1"] = "1"
        os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")
        log.info("LMCache config written to %s", config_path)

        extra = dict(self.cfg.nixl_agent_config) if self.connector == "NixlConnector" else {}
        kv_config = KVTransferConfig(
            kv_connector=self.connector,
            kv_role="kv_both",
            kv_connector_extra_config=extra or None,
        )
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "tensor_parallel_size": self.cfg.tensor_parallel,
            "max_model_len": self.cfg.max_model_len,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "max_num_seqs": self.cfg.max_num_seqs,
            "enable_prefix_caching": True,
            "kv_transfer_config": kv_config,
            "trust_remote_code": False,
        }
        if self.cfg.dtype != "auto":
            kwargs["dtype"] = self.cfg.dtype
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(**kwargs)
            self._engine = AsyncLLM.from_engine_args(engine_args)
            log.info("started vLLM V1 AsyncLLM with %s", self.connector)
        except Exception as exc:
            log.warning("AsyncLLM unavailable (%s); trying AsyncLLMEngine", exc)
            try:
                from vllm import AsyncEngineArgs, AsyncLLMEngine

                self._engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
                log.info("started vLLM AsyncLLMEngine with %s", self.connector)
            except Exception as exc2:
                raise EngineUnavailable(
                    f"could not construct vLLM engine: {type(exc2).__name__}: {exc2}"
                ) from exc2
        self._started = True

    async def close(self) -> None:
        engine = self._engine
        self._engine = None
        self._started = False
        if engine is None:
            return
        for method in ("shutdown", "shutdown_background_loop"):
            fn = getattr(engine, method, None)
            if fn is None:
                continue
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
                return
            except Exception:
                log.exception("engine shutdown via %s failed", method)

    # -- generation ---------------------------------------------------------

    async def generate(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        if self._engine is None:
            await self.start()
        assert self._SamplingParams is not None
        sampling = self._SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.seed,
            ignore_eos=False,
        )
        emitted = 0
        prompt = req.text
        self._requests += 1
        async for output in self._engine.generate(prompt, sampling, req.request_id):
            if not output.outputs:
                continue
            out = output.outputs[0]
            token_ids = list(out.token_ids)
            cached = getattr(output, "num_cached_tokens", None)
            if cached:
                self._cached_tokens += int(cached)
                self._prompt_tokens += int(cached)
            while emitted < len(token_ids):
                tid = int(token_ids[emitted])
                self._tokens += 1
                yield TokenEvent(
                    token_id=tid,
                    timestamp_ns=time.perf_counter_ns(),
                    is_first=(emitted == 0),
                )
                emitted += 1

    # -- stats --------------------------------------------------------------

    def prefix_cache_hit_rate(self) -> float:
        if self._prompt_tokens <= 0:
            return 0.0
        return min(1.0, self._cached_tokens / self._prompt_tokens)

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "engine": self.name,
            "connector": self.connector,
            "mode_selection_reason": self.selection_reason,
            "requests": self._requests,
            "tokens_generated": self._tokens,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate(),
            "blocks_offloaded": 0,
            "blocks_restored": 0,
            "bytes_offloaded": 0,
            "bytes_restored": 0,
            "note": (
                "LMCache owns its internal counters; per-tier byte counters for the real "
                "engine are exported by LMCache/Prometheus on the GPU host, not by this "
                "client. The emulated engine reports TieredKVStore counters directly."
            ),
        }


def config_hash(tiers: list[TierConfig]) -> str:
    blob = yaml.safe_dump(lmcache_yaml(tiers), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None
