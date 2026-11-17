"""Calibrated emulated engine.

The timing model comes from a real small transformer run on this host
(``gpt2`` via transformers on CUDA/MPS/CPU). The engine then replays trace
prompts:

* prompt tokens are tokenized for real and cut into KV blocks whose ids form a
  hash chain, so a shared system prompt genuinely collides across requests;
* every prompt block is looked up in the real :class:`TieredKVStore` -- hits
  perform real I/O and are promoted, misses perform real I/O on the put path;
* compute time is the measured prefill/decode time per token, so TTFT and ITP
  contain both the real storage time and the calibrated model time.

No latency value is invented: the calibration file records how it was measured.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvoffloadbench.config import EngineConfig, ModelShape
from kvoffloadbench.engines.base import (
    EngineUnavailable,
    GenerationRequest,
    TokenEvent,
)
from kvoffloadbench.tiers.tiered_store import TieredKVStore

log = logging.getLogger(__name__)

CALIBRATION_PATH = "results/calibration/engine_calibration.json"


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _measure_on_device(model: Any, tokenizer: Any, device: str) -> dict[str, float]:
    import torch

    model.to(device)
    model.eval()
    text = (
        "The KV cache grows with context length and must be offloaded to tiered "
        "storage when it no longer fits in HBM. " * 128
    )
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    seq_len = min(1024, ids.shape[1])
    ids = ids[:, :seq_len].to(device)
    with torch.inference_mode():
        for _ in range(2):
            model(input_ids=ids, use_cache=False)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model(input_ids=ids, use_cache=False)
        _sync(device)
        prefill_s = (time.perf_counter() - t0) / 3.0
        prefill_ms_per_token = prefill_s * 1000.0 / seq_len

        prompt = ids[:, :8]
        past = None
        for _ in range(2):  # warmup decode
            out = model(input_ids=prompt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            prompt = out.logits[:, -1:, :].argmax(dim=-1)
        _sync(device)
        steps = 16
        t0 = time.perf_counter()
        for _ in range(steps):
            out = model(input_ids=prompt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            prompt = out.logits[:, -1:, :].argmax(dim=-1)
        _sync(device)
        decode_s = (time.perf_counter() - t0) / steps
    return {
        "prefill_s": prefill_s,
        "prefill_ms_per_token": prefill_ms_per_token,
        "decode_s": decode_s,
        "decode_ms_per_token": decode_s * 1000.0,
    }


def calibrate(
    model_name: str = "gpt2",
    out_path: str | Path = CALIBRATION_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """Measure prefill/decode per-token cost with a real local model run."""
    out = Path(out_path)
    if out.is_file() and not force:
        try:
            existing = json.loads(out.read_text())
            if existing.get("model") == model_name and existing.get("prefill_ms_per_token"):
                return existing
        except (json.JSONDecodeError, OSError):
            pass

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise EngineUnavailable(
            f"cannot calibrate emulated engine: torch/transformers unavailable ({exc})"
        ) from exc

    if torch.cuda.is_available():
        candidates = ["cuda"]
    elif torch.backends.mps.is_available():
        candidates = ["mps", "cpu"]
    else:
        candidates = ["cpu"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as exc:
        raise EngineUnavailable(
            f"cannot load calibration model {model_name!r}: {type(exc).__name__}: {exc}"
        ) from exc

    errors: list[str] = []
    measured: dict[str, float] | None = None
    device_used = "cpu"
    for device in candidates:
        try:
            measured = _measure_on_device(model, tokenizer, device)
            device_used = device
            break
        except Exception as exc:
            errors.append(f"{device}: {type(exc).__name__}: {exc}")
            log.warning("calibration on %s failed: %s", device, exc)
    if measured is None:
        raise EngineUnavailable(
            "calibration failed on every device: " + "; ".join(errors)
        )

    payload: dict[str, Any] = {
        "calibration_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model": model_name,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": device_used,
        "device_candidates": candidates,
        "device_errors": errors,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": sys.version.split()[0],
        "prefill_seq_len": 1024,
        "prefill_reps": 3,
        "prefill_ms_per_token": measured["prefill_ms_per_token"],
        "prefill_s": measured["prefill_s"],
        "prefill_tokens_per_s": 1.0 / measured["prefill_ms_per_token"] * 1000.0,
        "decode_steps": 16,
        "decode_ms_per_token": measured["decode_ms_per_token"],
        "decode_s": measured["decode_s"],
        "decode_tokens_per_s": 1.0 / measured["decode_ms_per_token"] * 1000.0,
        "notes": (
            "Measured with torch.inference_mode() on this host. Prefill is measured "
            "on a 1024-token sequence and applied linearly per token; decode is "
            "measured with a growing KV cache. These timings drive the emulated "
            "engine's sleeps; tier I/O is measured separately and is real."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info(
        "calibrated on %s: prefill %.4f ms/token, decode %.4f ms/token -> %s",
        device_used,
        payload["prefill_ms_per_token"],
        payload["decode_ms_per_token"],
        out,
    )
    return payload


def load_calibration(path: str | Path = CALIBRATION_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return calibrate(out_path=p)
    data = json.loads(p.read_text())
    for key in ("prefill_ms_per_token", "decode_ms_per_token", "device", "model"):
        if key not in data:
            raise EngineUnavailable(f"calibration file {p} is missing {key!r}")
    return data


# ---------------------------------------------------------------------------
# emulated engine
# ---------------------------------------------------------------------------


def chained_block_ids(token_ids: list[int], block_tokens: int) -> list[str]:
    """Prefix-chained block ids: identical prefixes produce identical ids."""
    ids: list[str] = []
    chain = b""
    for start in range(0, len(token_ids), block_tokens):
        block = token_ids[start : start + block_tokens]
        blob = chain + b"|" + ",".join(str(t) for t in block).encode()
        digest = hashlib.sha256(blob).hexdigest()[:24]
        ids.append(digest)
        chain = digest.encode()
    return ids


class EmulatedEngine:
    mode = "emulated"
    name = "emulated"

    def __init__(
        self,
        cfg: EngineConfig,
        store: TieredKVStore | None = None,
        run_name: str = "run",
        calibration_path: str | Path | None = None,
        selection_reason: str = "",
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.run_name = run_name
        self.selection_reason = selection_reason or (
            "no CUDA GPU + vLLM combination available; using calibrated emulation"
        )
        self.shape: ModelShape = cfg.model_shape
        self.cal_path = Path(calibration_path or cfg.calibration_path or CALIBRATION_PATH)
        self.calibration = load_calibration(self.cal_path)
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        except Exception as exc:
            raise EngineUnavailable(
                f"emulated engine needs a real tokenizer for {cfg.model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        filler = self._tokenizer.encode(" the", add_special_tokens=False)
        self._filler = filler or [220]
        self._vocab_size = int(getattr(self._tokenizer, "vocab_size", 50257) or 50257)
        self._payload_cache: OrderedDict[str, bytes] = OrderedDict()
        self._stats_lock = threading.Lock()
        self._requests = 0
        self._tokens = 0
        self._blocks_looked_up = 0
        self._blocks_hit = 0
        self._blocks_missed = 0
        self._restored_bytes = 0
        self._decode_blocks_put = 0
        self._prefill_compute_s = 0.0
        self._decode_compute_s = 0.0
        self._io_s = 0.0
        self._last_prefill_s = 0.0

    # -- helpers ------------------------------------------------------------

    @property
    def block_tokens(self) -> int:
        return self.shape.block_tokens

    @property
    def bytes_per_block(self) -> int:
        return self.shape.bytes_per_block

    def _tokenize(self, req: GenerationRequest) -> list[int]:
        ids = self._tokenizer.encode(req.text, add_special_tokens=False)
        target = req.prompt_tokens or len(ids)
        if len(ids) >= target:
            return ids[:target]
        filler = (self._filler * ((target - len(ids)) // len(self._filler) + 1))[
            : target - len(ids)
        ]
        return ids + filler

    def _payload(self, block_id: str, nbytes: int) -> bytes:
        cached = self._payload_cache.get(block_id)
        if cached is not None and len(cached) == nbytes:
            self._payload_cache.move_to_end(block_id)
            return cached
        digest = hashlib.sha256(block_id.encode()).digest()
        reps = nbytes // len(digest) + 1
        payload = (digest * reps)[:nbytes]
        self._payload_cache[block_id] = payload
        if len(self._payload_cache) > 512:
            self._payload_cache.popitem(last=False)
        return payload

    def _prefill_sync(self, token_ids: list[int]) -> tuple[int, int, int, int, float]:
        """Look up / put KV blocks; returns (looked_up, hits, misses, restored, io_s)."""
        ids = chained_block_ids(token_ids, self.block_tokens)
        hits = misses = restored = 0
        io_ns = 0
        for index, block_id in enumerate(ids):
            ntokens = min(self.block_tokens, len(token_ids) - index * self.block_tokens)
            nbytes = min(self.bytes_per_block, ntokens * self.shape.bytes_per_token)
            t0 = time.perf_counter_ns()
            data = self.store.get(block_id) if self.store is not None else None
            if data is not None:
                hits += 1
                restored += len(data)
            else:
                misses += 1
                if self.store is not None:
                    self.store.put(block_id, self._payload(block_id, nbytes))
            io_ns += time.perf_counter_ns() - t0
        return len(ids), hits, misses, restored, io_ns / 1e9

    def _put_decode_block_sync(self, block_id: str, nbytes: int) -> None:
        if self.store is not None:
            self.store.put(block_id, self._payload(block_id, nbytes))

    # -- generation ---------------------------------------------------------

    async def generate(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        token_ids = self._tokenize(req)
        prefill_t0 = time.perf_counter()
        looked_up, hits, misses, restored, io_s = await asyncio.to_thread(
            self._prefill_sync, token_ids
        )
        # Compute only the blocks that were not restored from storage.
        compute_s = (
            misses * self.block_tokens
            * self.calibration["prefill_ms_per_token"] / 1000.0
        )
        if compute_s > 0:
            await asyncio.sleep(compute_s)
        prefill_s = time.perf_counter() - prefill_t0

        with self._stats_lock:
            self._requests += 1
            self._blocks_looked_up += looked_up
            self._blocks_hit += hits
            self._blocks_missed += misses
            self._restored_bytes += restored
            self._prefill_compute_s += compute_s
            self._io_s += io_s

        seed_int = int(
            hashlib.sha256(
                f"{req.seed}:{req.request_id}:{req.turn_index}".encode()
            ).hexdigest()[:16],
            16,
        )
        rng = random.Random(seed_int)
        decode_s = self.calibration["decode_ms_per_token"] / 1000.0
        generated: list[int] = []
        for step in range(max(1, req.max_tokens)):
            await asyncio.sleep(decode_s)
            token_id = rng.randrange(1000, self._vocab_size)
            generated.append(token_id)
            with self._stats_lock:
                self._tokens += 1
                self._decode_compute_s += decode_s
            yield TokenEvent(
                token_id=token_id,
                timestamp_ns=time.perf_counter_ns(),
                is_first=(step == 0),
            )
            if len(generated) % self.block_tokens == 0 and self.store is not None:
                block_tokens = token_ids + generated
                block_id = chained_block_ids(block_tokens, self.block_tokens)[-1]
                await asyncio.to_thread(
                    self._put_decode_block_sync, block_id, self.bytes_per_block
                )
                with self._stats_lock:
                    self._decode_blocks_put += 1

        self._last_prefill_s = prefill_s

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            looked = self._blocks_looked_up
            hit_rate = self._blocks_hit / looked if looked else 0.0
            counters = {
                "requests": self._requests,
                "tokens_generated": self._tokens,
                "prompt_blocks_looked_up": looked,
                "prompt_blocks_hit": self._blocks_hit,
                "prompt_blocks_missed": self._blocks_missed,
                "prefix_cache_hit_rate": hit_rate,
                "restored_bytes": self._restored_bytes,
                "decode_blocks_put": self._decode_blocks_put,
                "prefill_compute_s": self._prefill_compute_s,
                "decode_compute_s": self._decode_compute_s,
                "storage_io_s": self._io_s,
            }
        store_stats = self.store.stats() if self.store is not None else None
        if store_stats is not None:
            sc = store_stats["counters"]
            counters.update(
                {
                    "blocks_offloaded": sc["offload_ops"],
                    "blocks_restored": sc["restore_ops"],
                    "bytes_offloaded": sc["offload_bytes"],
                    "bytes_restored": sc["restore_bytes"],
                }
            )
        return {
            "mode": self.mode,
            "engine": self.name,
            "model": self.cfg.model,
            "model_shape": self.shape.to_dict(),
            "calibration": self.calibration,
            "mode_selection_reason": self.selection_reason,
            "store": store_stats,
            **counters,
        }

    async def close(self) -> None:
        return None
