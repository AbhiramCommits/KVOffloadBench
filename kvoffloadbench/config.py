"""Typed configuration model for KVOffloadBench.

The YAML surface has two entry points:

* :func:`load_run_config` -- a single run specification.
* :func:`load_sweep` -- a cross-product matrix that expands into many
  :class:`RunConfig` objects, optionally trimmed with ``include`` / ``exclude``
  blocks.

All validation errors raise :class:`ConfigError` with a message that names the
offending field and the accepted values.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIER_KINDS = ("hbm", "dram", "nvme", "remote_tcp", "nvmeof_block")
ENGINE_MODES = ("auto", "real", "emulated")
CONNECTORS = ("lmcache", "nixl", "none")
DTYPES = ("float16", "bfloat16", "float32", "int8")


class ConfigError(ValueError):
    """Raised when a run or sweep configuration is invalid."""


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierConfig:
    """One level of the storage hierarchy.

    ``path`` is used by ``nvme`` and ``nvmeof_block`` (the latter may also read
    ``KVOB_NVMEOF_DEV``); ``endpoint`` is a ``host:port`` pair for ``remote_tcp``.
    """

    kind: str
    capacity_mib: int = 1024
    path: str | None = None
    endpoint: str | None = None
    block_size_kib: int = 64
    queue_depth: int = 8
    direct_io: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TIER_KINDS:
            raise ConfigError(
                f"tier.kind={self.kind!r} is invalid; choose one of {list(TIER_KINDS)}"
            )
        if not isinstance(self.capacity_mib, int) or self.capacity_mib <= 0:
            raise ConfigError(
                f"tier.capacity_mib must be a positive integer (got {self.capacity_mib!r})"
            )
        if not isinstance(self.block_size_kib, int) or self.block_size_kib < 4:
            raise ConfigError(
                f"tier.block_size_kib must be an integer >= 4 (got {self.block_size_kib!r})"
            )
        if self.block_size_kib & (self.block_size_kib - 1):
            raise ConfigError(
                f"tier.block_size_kib must be a power of two (got {self.block_size_kib})"
            )
        if not isinstance(self.queue_depth, int) or not 1 <= self.queue_depth <= 1024:
            raise ConfigError(
                f"tier.queue_depth must be between 1 and 1024 (got {self.queue_depth!r})"
            )
        if self.kind == "nvme" and not self.path:
            raise ConfigError("tier kind='nvme' requires a 'path' for its backing file")
        if self.kind == "remote_tcp" and not self.endpoint:
            raise ConfigError(
                "tier kind='remote_tcp' requires an 'endpoint' such as '127.0.0.1:4420'"
            )
        if self.kind == "nvmeof_block" and not (self.path or self.extra.get("device_env")):
            raise ConfigError(
                "tier kind='nvmeof_block' requires 'path' (a /dev/nvmeXnY device) or "
                "extra.device_env naming an environment variable"
            )
        if self.endpoint and self.kind == "remote_tcp":
            host, _, port = self.endpoint.rpartition(":")
            if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
                raise ConfigError(
                    f"tier.endpoint={self.endpoint!r} must look like 'host:port' with a "
                    "port in 1..65535"
                )

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_mib * 1024 * 1024

    @property
    def block_bytes(self) -> int:
        return self.block_size_kib * 1024

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModelShape:
    """Declared transformer KV shape, used for byte accounting in emulation.

    Defaults describe a Llama-3.3-70B-class model: 80 layers, 8 KV heads,
    head_dim 128, fp16. Per token that is 80 * 2 * 8 * 128 * 2 = 327,680 bytes.
    Local runs on a laptop declare a smaller shape in their YAML so that real
    I/O volumes remain feasible; the shape always lands in the manifest.
    """

    layers: int = 80
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"
    block_tokens: int = 16
    tensor_parallel: int = 1

    def __post_init__(self) -> None:
        for name in ("layers", "kv_heads", "head_dim", "block_tokens", "tensor_parallel"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"model_shape.{name} must be a positive integer")
        if self.dtype not in DTYPES:
            raise ConfigError(
                f"model_shape.dtype={self.dtype!r} is invalid; choose one of {list(DTYPES)}"
            )

    @property
    def dtype_bytes(self) -> int:
        return {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}[self.dtype]

    @property
    def bytes_per_token(self) -> int:
        """Bytes of KV cache per token, summed across all layers, for one rank."""
        return (
            self.layers
            * 2
            * self.kv_heads
            * self.head_dim
            * self.dtype_bytes
            // self.tensor_parallel
        )

    @property
    def bytes_per_block(self) -> int:
        return self.bytes_per_token * self.block_tokens

    @property
    def total_bytes_per_token(self) -> int:
        """Bytes per token for the whole model (all tensor-parallel ranks)."""
        return self.layers * 2 * self.kv_heads * self.head_dim * self.dtype_bytes

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["bytes_per_token"] = self.bytes_per_token
        d["bytes_per_block"] = self.bytes_per_block
        return d


@dataclass(frozen=True)
class EngineConfig:
    mode: str = "auto"
    model: str = "gpt2"
    tensor_parallel: int = 1
    max_model_len: int = 32768
    connector: str = "lmcache"
    model_shape: ModelShape = field(default_factory=ModelShape)
    calibration_path: str | None = None
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 128
    nixl_agent_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ENGINE_MODES:
            raise ConfigError(
                f"engine.mode={self.mode!r} is invalid; choose one of {list(ENGINE_MODES)}"
            )
        if self.connector not in CONNECTORS:
            raise ConfigError(
                f"engine.connector={self.connector!r} is invalid; choose one of "
                f"{list(CONNECTORS)}"
            )
        if not isinstance(self.model, str) or not self.model:
            raise ConfigError("engine.model must be a non-empty HuggingFace id or path")
        if not isinstance(self.tensor_parallel, int) or self.tensor_parallel < 1:
            raise ConfigError("engine.tensor_parallel must be an integer >= 1")
        if not isinstance(self.max_model_len, int) or self.max_model_len < 128:
            raise ConfigError("engine.max_model_len must be an integer >= 128")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ConfigError("engine.gpu_memory_utilization must be in (0, 1]")
        if self.mode == "real" and self.connector == "none":
            raise ConfigError("engine.mode='real' requires connector 'lmcache' or 'nixl'")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["model_shape"] = self.model_shape.to_dict()
        return d


@dataclass(frozen=True)
class WorkloadConfig:
    trace: str = "longctx_multiturn"
    request_rate: float = 4.0
    context_tokens: int = 8192
    max_concurrency: int = 16
    duration_s: float = 30.0
    seed: int = 2024
    warmup_s: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.trace, str) or not self.trace:
            raise ConfigError("workload.trace must name a trace file or trace id")
        if not isinstance(self.request_rate, (int, float)) or self.request_rate <= 0:
            raise ConfigError("workload.request_rate must be > 0")
        if not isinstance(self.context_tokens, int) or self.context_tokens < 128:
            raise ConfigError("workload.context_tokens must be an integer >= 128")
        if not isinstance(self.max_concurrency, int) or self.max_concurrency < 1:
            raise ConfigError("workload.max_concurrency must be an integer >= 1")
        if not isinstance(self.duration_s, (int, float)) or self.duration_s <= 0:
            raise ConfigError("workload.duration_s must be > 0")
        if not 0 <= self.warmup_s < self.duration_s:
            raise ConfigError("workload.warmup_s must be >= 0 and less than duration_s")
        if not isinstance(self.seed, int):
            raise ConfigError("workload.seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SLOConfig:
    ttft_ms: float = 2000.0
    tpot_ms: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.ttft_ms, (int, float)) or self.ttft_ms <= 0:
            raise ConfigError("slo.ttft_ms must be > 0")
        if not isinstance(self.tpot_ms, (int, float)) or self.tpot_ms <= 0:
            raise ConfigError("slo.tpot_ms must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunConfig:
    name: str
    engine: EngineConfig = field(default_factory=EngineConfig)
    tiers: tuple[TierConfig, ...] = field(default_factory=tuple)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    description: str = ""
    tier_chain: str = ""
    capacity_label: str = ""
    axis_values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("run.name must be a non-empty string")
        if not self.tiers:
            raise ConfigError("run.tiers must contain at least one tier")
        kinds = [t.kind for t in self.tiers]
        if len(set(kinds)) != len(kinds):
            raise ConfigError(f"run.tiers has duplicate kinds: {kinds}")
        if kinds[0] not in ("hbm", "dram"):
            raise ConfigError(
                "the first tier must be 'hbm' or 'dram' (the hot cache); got "
                f"{kinds[0]!r}"
            )

    def run_id(self) -> str:
        """Stable identifier used for result directories and resume logic."""
        w = self.workload
        rate = f"{w.request_rate:g}"
        cap = f"-k{self.capacity_label}" if self.capacity_label else ""
        chain = self.tier_chain or "+".join(t.kind for t in self.tiers)
        return _slug(
            f"{self.name}-{chain}-r{rate}-c{w.context_tokens}{cap}-{_trace_stem(w.trace)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id(),
            "description": self.description,
            "tier_chain": self.tier_chain,
            "capacity_label": self.capacity_label,
            "axis_values": self.axis_values,
            "engine": self.engine.to_dict(),
            "tiers": [t.to_dict() for t in self.tiers],
            "workload": self.workload.to_dict(),
            "slo": self.slo.to_dict(),
        }


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-").lower()


def _trace_stem(trace: str) -> str:
    return Path(trace).stem


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level")
    return raw


def _check_keys(section: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown field(s) in {where}: {unknown}; allowed: {sorted(allowed)}")


def _build_tier(raw: Mapping[str, Any], where: str) -> TierConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
    allowed = {
        "kind",
        "capacity_mib",
        "path",
        "endpoint",
        "block_size_kib",
        "queue_depth",
        "direct_io",
        "extra",
    }
    _check_keys(raw, allowed, where)
    if "kind" not in raw:
        raise ConfigError(f"{where} is missing required field 'kind'")
    extra = raw.get("extra") or {}
    if not isinstance(extra, dict):
        raise ConfigError(f"{where}.extra must be a mapping")
    return TierConfig(
        kind=str(raw["kind"]),
        capacity_mib=raw.get("capacity_mib", 1024),
        path=raw.get("path"),
        endpoint=raw.get("endpoint"),
        block_size_kib=raw.get("block_size_kib", 64),
        queue_depth=raw.get("queue_depth", 8),
        direct_io=raw.get("direct_io", True),
        extra=extra,
    )


def _build_engine(raw: Mapping[str, Any] | None, where: str) -> EngineConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    allowed = {
        "mode",
        "model",
        "tensor_parallel",
        "max_model_len",
        "connector",
        "model_shape",
        "calibration_path",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "nixl_agent_config",
    }
    _check_keys(raw, allowed, where)
    shape_raw = raw.get("model_shape") or {}
    if not isinstance(shape_raw, dict):
        raise ConfigError(f"{where}.model_shape must be a mapping")
    shape_allowed = {
        "layers",
        "kv_heads",
        "head_dim",
        "dtype",
        "block_tokens",
        "tensor_parallel",
    }
    _check_keys(shape_raw, shape_allowed, f"{where}.model_shape")
    shape = ModelShape(**shape_raw)
    if "tensor_parallel" in raw and "tensor_parallel" not in shape_raw:
        shape = dataclasses.replace(shape, tensor_parallel=int(raw["tensor_parallel"]))
    nixl_cfg = raw.get("nixl_agent_config") or {}
    if not isinstance(nixl_cfg, dict):
        raise ConfigError(f"{where}.nixl_agent_config must be a mapping")
    return EngineConfig(
        mode=str(raw.get("mode", "auto")),
        model=str(raw.get("model", "gpt2")),
        tensor_parallel=int(raw.get("tensor_parallel", 1)),
        max_model_len=int(raw.get("max_model_len", 32768)),
        connector=str(raw.get("connector", "lmcache")),
        model_shape=shape,
        calibration_path=raw.get("calibration_path"),
        dtype=str(raw.get("dtype", "auto")),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9)),
        max_num_seqs=int(raw.get("max_num_seqs", 128)),
        nixl_agent_config=nixl_cfg,
    )


def _build_workload(raw: Mapping[str, Any] | None, where: str) -> WorkloadConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    allowed = {
        "trace",
        "request_rate",
        "context_tokens",
        "max_concurrency",
        "duration_s",
        "seed",
        "warmup_s",
    }
    _check_keys(raw, allowed, where)
    return WorkloadConfig(
        trace=str(raw.get("trace", "longctx_multiturn")),
        request_rate=float(raw.get("request_rate", 4.0)),
        context_tokens=int(raw.get("context_tokens", 8192)),
        max_concurrency=int(raw.get("max_concurrency", 16)),
        duration_s=float(raw.get("duration_s", 30.0)),
        seed=int(raw.get("seed", 2024)),
        warmup_s=float(raw.get("warmup_s", 3.0)),
    )


def _build_slo(raw: Mapping[str, Any] | None, where: str) -> SLOConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    _check_keys(raw, {"ttft_ms", "tpot_ms"}, where)
    return SLOConfig(
        ttft_ms=float(raw.get("ttft_ms", 2000.0)),
        tpot_ms=float(raw.get("tpot_ms", 100.0)),
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a single-run YAML file."""
    raw = _read_yaml(path)
    _check_keys(raw, {"name", "description", "engine", "tiers", "workload", "slo"}, "run")
    if "name" not in raw:
        raise ConfigError(f"{path}: run config is missing required field 'name'")
    tiers_raw = raw.get("tiers")
    if not isinstance(tiers_raw, list) or not tiers_raw:
        raise ConfigError(f"{path}: 'tiers' must be a non-empty list")
    tiers = tuple(
        _build_tier(t, f"tiers[{i}]") for i, t in enumerate(tiers_raw)
    )
    return RunConfig(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        engine=_build_engine(raw.get("engine"), "engine"),
        tiers=tiers,
        workload=_build_workload(raw.get("workload"), "workload"),
        slo=_build_slo(raw.get("slo"), "slo"),
    )


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------


def _axis_values(value: Any, axis: str) -> list[tuple[Any, str]]:
    """Normalise an axis to a list of (value, label) pairs."""
    if isinstance(value, list):
        return [(v, str(v)) for v in value]
    if isinstance(value, dict):
        if not value:
            raise ConfigError(f"axes.{axis} is an empty mapping")
        return [(v, str(k)) for k, v in value.items()]
    raise ConfigError(
        f"axes.{axis} must be a list of values or a mapping of label -> value"
    )


def _subset_match(assignment: Mapping[str, Any], pattern: Mapping[str, Any]) -> bool:
    for key, value in pattern.items():
        if key not in assignment:
            raise ConfigError(f"exclude/include references unknown axis {key!r}")
        if assignment[key] != value:
            return False
    return True


def _substitute(value: Any, assignment: Mapping[str, Any]) -> Any:
    """Recursively substitute ``${axis}`` placeholders with axis values."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in assignment:
                raise ConfigError(
                    f"tier-chain template references unknown axis {key!r}"
                )
            return str(assignment[key])

        # A value that is exactly a placeholder keeps its native type.
        exact = re.fullmatch(r"\$\{([A-Za-z0-9_]+)\}", value)
        if exact:
            key = exact.group(1)
            if key not in assignment:
                raise ConfigError(
                    f"tier-chain template references unknown axis {key!r}"
                )
            return assignment[key]
        return re.sub(r"\$\{([A-Za-z0-9_]+)\}", repl, value)
    if isinstance(value, list):
        return [_substitute(v, assignment) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, assignment) for k, v in value.items()}
    return value


def load_sweep(path: str | Path) -> list[RunConfig]:
    """Expand a sweep YAML matrix into validated :class:`RunConfig` objects.

    Schema::

        name: main
        engine: {...}                 # EngineConfig fields
        slo: {...}
        workload_defaults: {...}      # WorkloadConfig fields, used as defaults
        chains:                       # named tier chains, templated on axes
          dram_only: [{kind: dram, capacity_mib: "${kv_cache_capacity_mib}"}]
        axes:
          tier_chain: [dram_only, ...]
          request_rate: [1, 2, 4, 8]
          context_tokens: [4096, 16384, 32768]
          kv_cache_capacity_mib: {small: 96, medium: 512, large: 2048}
          trace: [longctx_multiturn, ...]
        include:                      # optional partial cross-products
          - {tier_chain: [...], request_rate: [...], ...}
        exclude:                      # optional partial assignments to drop
          - {tier_chain: dram_only, request_rate: 8}
    """
    raw = _read_yaml(path)
    _check_keys(
        raw,
        {"name", "description", "engine", "slo", "workload_defaults", "chains", "axes", "include", "exclude", "drop_caches", "drain_timeout_s"},
        "sweep",
    )
    if "name" not in raw:
        raise ConfigError(f"{path}: sweep config is missing required field 'name'")
    axes_raw = raw.get("axes")
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise ConfigError(f"{path}: 'axes' must be a non-empty mapping")
    chains_raw = raw.get("chains")
    if not isinstance(chains_raw, dict) or not chains_raw:
        raise ConfigError(f"{path}: 'chains' must be a non-empty mapping")

    axis_values: dict[str, list[tuple[Any, str]]] = {
        axis: _axis_values(value, axis) for axis, value in axes_raw.items()
    }
    for chain_name, chain in chains_raw.items():
        if not isinstance(chain, list) or not chain:
            raise ConfigError(f"chains.{chain_name} must be a non-empty list of tiers")

    engine = _build_engine(raw.get("engine"), "engine")
    slo = _build_slo(raw.get("slo"), "slo")
    workload_defaults = raw.get("workload_defaults") or {}
    if not isinstance(workload_defaults, dict):
        raise ConfigError("workload_defaults must be a mapping")
    _check_keys(
        workload_defaults,
        {"trace", "request_rate", "context_tokens", "max_concurrency", "duration_s", "seed", "warmup_s"},
        "workload_defaults",
    )

    include_raw = raw.get("include")
    exclude_raw = raw.get("exclude") or []
    if not isinstance(exclude_raw, list):
        raise ConfigError("exclude must be a list of partial axis assignments")

    if include_raw is None:
        blocks = [dict(axes_raw)]
        axis_sets: list[list[str]] = [list(axis_values)]
    else:
        if not isinstance(include_raw, list) or not include_raw:
            raise ConfigError("include must be a non-empty list when present")
        blocks = []
        axis_sets = []
        for i, block in enumerate(include_raw):
            if not isinstance(block, dict) or not block:
                raise ConfigError(f"include[{i}] must be a non-empty mapping")
            unknown = sorted(set(block) - set(axis_values))
            if unknown:
                raise ConfigError(
                    f"include[{i}] references unknown axes {unknown}; known: {sorted(axis_values)}"
                )
            blocks.append(block)
            axis_sets.append(list(block))

    seen: set[str] = set()
    run_configs: list[RunConfig] = []

    def _resolve_include(axis: str, value: Any) -> list[tuple[Any, str]]:
        items = _axis_values(value, f"include.{axis}")
        mapping = axes_raw[axis]
        if isinstance(mapping, dict):
            resolved: list[tuple[Any, str]] = []
            for item, _ in items:
                if item not in mapping:
                    raise ConfigError(
                        f"include value {item!r} for axis {axis!r} is not a label in "
                        f"axes.{axis} ({sorted(mapping)})"
                    )
                resolved.append((mapping[item], str(item)))
            return resolved
        return items

    for block, axis_names in zip(blocks, axis_sets, strict=False):
        value_lists = []
        for axis in axis_names:
            if axis in block:
                value_lists.append(_resolve_include(axis, block[axis]))
            else:
                value_lists.append(axis_values[axis])
        for combo in itertools.product(*value_lists):
            assignment: dict[str, Any] = {}
            labels: dict[str, str] = {}
            for axis, (value, label) in zip(axis_names, combo, strict=False):
                assignment[axis] = value
                labels[axis] = label
            if any(_subset_match(assignment, pat) for pat in exclude_raw if isinstance(pat, dict)):
                continue
            if "tier_chain" not in assignment:
                raise ConfigError("every sweep run must set the 'tier_chain' axis")
            chain_name = assignment["tier_chain"]
            if chain_name not in chains_raw:
                raise ConfigError(
                    f"tier_chain={chain_name!r} is not defined in 'chains' "
                    f"(have {sorted(chains_raw)})"
                )
            substituted = _substitute(chains_raw[chain_name], assignment)
            tiers = tuple(
                _build_tier(t, f"chains.{chain_name}[{j}]")
                for j, t in enumerate(substituted)
            )
            wl_raw = dict(workload_defaults)
            for key in ("trace", "request_rate", "context_tokens"):
                if key in assignment:
                    wl_raw[key] = assignment[key]
            if "trace" not in wl_raw:
                raise ConfigError("sweep must define a trace (via axis or workload_defaults)")
            workload = _build_workload(wl_raw, "workload")

            cap_label = labels.get("kv_cache_capacity_mib", "")
            run = RunConfig(
                name=str(raw["name"]),
                description=str(raw.get("description", "")),
                engine=engine,
                tiers=tiers,
                workload=workload,
                slo=slo,
                tier_chain=str(chain_name),
                capacity_label=cap_label,
                axis_values=dict(assignment),
            )
            rid = run.run_id()
            if rid in seen:
                raise ConfigError(
                    f"duplicate run_id {rid!r} after expansion; two axis combinations "
                    "collide -- check axis labels and tier capacities"
                )
            seen.add(rid)
            run_configs.append(run)

    if not run_configs:
        raise ConfigError("sweep expansion produced zero runs; check include/exclude blocks")
    return run_configs
