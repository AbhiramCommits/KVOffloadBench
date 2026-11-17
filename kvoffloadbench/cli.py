"""Command-line interface for KVOffloadBench."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from kvoffloadbench import __version__
from kvoffloadbench.env import fingerprint

_DESCRIPTION = """\
KVOffloadBench - measure the cost of pushing an LLM KV cache down the storage
hierarchy (HBM -> DRAM -> local NVMe -> remote NVMe-oF) and turn it into a
storage requirements table (bandwidth, IOPS, tail latency).

Engine modes are recorded in every result row:
  real      actual vLLM + LMCache (or NIXL) engine, used when CUDA + vLLM exist
  emulated  calibrated analytic decode/prefill driving REAL tier I/O
"""

_EPILOG = """\
Typical sequence:
  kvoffloadbench env
  kvoffloadbench gen-traces --out traces/
  kvoffloadbench bench-tiers --config configs/tiers_micro.yaml
  kvoffloadbench validate
  kvoffloadbench replay --config configs/runs/dram_only.yaml
  kvoffloadbench sweep --config configs/sweeps/main.yaml
  kvoffloadbench analyze
"""


def _not_implemented(args: argparse.Namespace) -> int:
    print(
        f"kvoffloadbench {args.command}: not implemented yet "
        "(this subcommand is wired up in a later build step)",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvoffloadbench",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p_env = sub.add_parser(
        "env", help="print the environment fingerprint JSON used in every manifest"
    )
    p_env.add_argument(
        "--no-freeze",
        action="store_true",
        help="skip `pip freeze` (faster; pip_freeze_sha256 will be null)",
    )
    p_env.set_defaults(func=_cmd_env)

    p_bt = sub.add_parser(
        "bench-tiers", help="run the storage tier microbenchmark"
    )
    p_bt.add_argument("--config", default="configs/tiers_micro.yaml", help="microbench YAML")
    p_bt.add_argument(
        "--out", default="results/tiers/microbench.csv", help="output CSV path"
    )
    p_bt.set_defaults(func=_not_implemented)

    p_replay = sub.add_parser("replay", help="replay one traced workload against an engine")
    p_replay.add_argument("--config", required=True, help="run config YAML")
    p_replay.add_argument("--metrics-port", type=int, default=0, help="Prometheus port (0=off)")
    p_replay.add_argument("--duration", type=float, default=None, help="override duration_s")
    p_replay.set_defaults(func=_not_implemented)

    p_sweep = sub.add_parser("sweep", help="expand and run a sweep matrix")
    p_sweep.add_argument("--config", required=True, help="sweep config YAML")
    p_sweep.add_argument("--dry-run", action="store_true", help="print run ids and exit")
    p_sweep.add_argument("--metrics-port", type=int, default=0, help="Prometheus port (0=off)")
    p_sweep.set_defaults(func=_not_implemented)

    p_gen = sub.add_parser("gen-traces", help="generate deterministic request traces")
    p_gen.add_argument("--out", default="traces/", help="output directory")
    p_gen.add_argument("--seed", type=int, default=2024, help="generator seed")
    p_gen.set_defaults(func=_not_implemented)

    p_val = sub.add_parser("validate", help="verify KV offload/reload correctness")
    p_val.add_argument("--out", default="results/validation/kv_correctness.json")
    p_val.add_argument("--config", default="configs/validate.yaml")
    p_val.set_defaults(func=_not_implemented)

    p_an = sub.add_parser("analyze", help="regenerate tables and figures from results/")
    p_an.add_argument("--results", default="results/", help="results directory")
    p_an.set_defaults(func=_not_implemented)

    return parser


def _cmd_env(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            fingerprint(include_pip_freeze=not args.no_freeze),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return int(result) if result is not None else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
