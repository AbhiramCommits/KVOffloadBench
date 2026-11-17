# KVOffloadBench runbook

This runbook is for the target deployment host: **4x NVIDIA H200** serving a
Llama-3.3-70B-class model with tensor parallel 4, LMCache and/or NIXL KV
connectors, and an NVMe-oF namespace exported over RDMA/RoCE to a
BlueField-3-equipped storage server. It also records exactly what this
repository's build could and could not measure on a CPU/MPS laptop.

## 0. What was measured in this build

| Component | This build (Apple M3 laptop, 16 GB, macOS) | Target host |
|---|---|---|
| Serving engine | `emulated`: calibrated gpt2 prefill/decode; real per-block KV traffic through the tier stack | `real`: vLLM + LMCache/NIXL |
| Calibration | gpt2 on MPS, `results/calibration/engine_calibration.json` | same procedure on the H200 or use the committed calibration |
| DRAM tier | real in-process buffers | real pinned host buffers |
| Local NVMe tier | real file on the laptop SSD, `F_NOCACHE` (macOS has no `O_DIRECT`) | real NVMe with `O_DIRECT` |
| Remote tier | real TCP target process on loopback, real file I/O behind it | NVMe-oF over RDMA/RoCE to the storage server |
| `nvmeof_block` tier | unavailable (no `/dev/nvme*`), skipped with an explicit log | enabled when `KVOB_NVMEOF_DEV` is exported |

Every number in `README.md` traces to a CSV/JSON/Parquet file produced by the
commands below. Nothing was synthesized.

## 1. Target host prerequisites

- Ubuntu 24.04, kernel 6.8+, NVIDIA driver 550+, CUDA 12.4.
- Python 3.11 or 3.12, `git`, build tooling.
- vLLM >= 0.6.3 (V1 engine), LMCache >= 0.2.5.
- NIXL (`nvidia-nixl`) only if using the Dynamo/NIXL connector path.
- `nvme-cli`, `nvmet`, `nvmet-rdma`, `nvme-rdma` (or `nvmet-tcp`/`nvme-tcp`).
- rdma-core and the NIC driver (`mlx5` for BlueField-3/ConnectX-7).
- A scratch NVMe-oF namespace you are allowed to overwrite.

```bash
# Host sanity checks before spending GPU time
nvidia-smi -L
ibv_devinfo -l
nvme list-subsys
nvme list
```

## 2. Install

```bash
git clone <your fork> KVOffloadBench && cd KVOffloadBench
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[gpu,dev]"
python -c "import torch, vllm, lmcache; print(torch.cuda.is_available(), vllm.__version__)"
kvoffloadbench env > results/env_target.json
```

HuggingFace access for the model:

```bash
huggingface-cli login
# accept the Llama-3.3-70B-Instruct license on the model page first
```

## 3. NVMe-oF namespace

### 3a. TCP loopback (works without an RDMA fabric; already scripted)

```bash
sudo scripts/setup_nvmeof_tcp.sh
# prints: export KVOB_NVMEOF_DEV=/dev/nvmeXnY
export KVOB_NVMEOF_DEV=/dev/nvme1n1
```

Verify:

```bash
nvme list-subsys
nvme id-ns "$KVOB_NVMEOF_DEV"
kvoffloadbench bench-tiers --config configs/tiers_micro.yaml   # the nvmeof_block rows appear
```

### 3b. RDMA/RoCE to a BlueField-3 storage server

On the **storage server** (same commands; replace the IP and backing device):

```bash
sudo modprobe nvmet nvmet-rdma
sudo mkdir -p /sys/kernel/config/nvmet/subsystems/nqn.2026-01.io.kvoffloadbench:storage
echo 1 | sudo tee .../attr_allow_any_host
sudo mkdir -p .../namespaces/1
echo -n /dev/nvme0n1p2 | sudo tee .../namespaces/1/device_path   # dedicated scratch partition
echo 1 | sudo tee .../namespaces/1/enable
sudo mkdir -p /sys/kernel/config/nvmet/ports/1
echo rdma   | sudo tee .../ports/1/addr_trtype
echo 10.0.0.10 | sudo tee .../ports/1/addr_traddr
echo ipv4   | sudo tee .../ports/1/addr_adrfam
echo 4420   | sudo tee .../ports/1/addr_trsvcid
sudo ln -s .../subsystems/nqn.2026-01.io.kvoffloadbench:storage .../ports/1/subsystems/
```

On the **GPU host**:

```bash
sudo modprobe nvme-rdma
sudo nvme connect -t rdma -n nqn.2026-01.io.kvoffloadbench:storage \
    -a 10.0.0.10 -s 4420
export KVOB_NVMEOF_DEV=/dev/nvme2n1
nvme list-subsys
```

RoCEv2 requirements: lossless priority (PFC) on the storage VLAN, ECN enabled,
MTU 9000 end to end. Verify with:

```bash
ibv_devinfo
ib_write_bw -d mlx5_0 -a --report_gbits    # server side first
```

### 3c. Teardown

```bash
sudo nvme disconnect -n nqn.2026-01.io.kvoffloadbench:storage
sudo scripts/teardown_nvmeof_tcp.sh
```

## 4. Traces and calibration

```bash
kvoffloadbench gen-traces --out traces/ --seed 2024
# Calibrates once with gpt2 on the best local device and writes:
#   results/calibration/engine_calibration.json
kvoffloadbench validate --config configs/validate.yaml
```

`validate` must report `byte_mismatches=0` and `token_exact_match_rate=1.0`. It
exercises DRAM, local NVMe, the TCP target, and (when `KVOB_NVMEOF_DEV` is set)
the real NVMe-oF block tier, including a forced demotion to the last tier.

## 5. Key storage microbenchmark

```bash
kvoffloadbench bench-tiers --config configs/tiers_micro.yaml \
  --out results/tiers/microbench.csv
```

Every tier is real I/O. The `direct_io` column records whether `O_DIRECT`
(Linux), `F_NOCACHE` (macOS), or a buffered fallback was in force. On the GPU
host the `nvmeof_block` rows are the ones that matter for the fabric.

## 6. Single-run replay

```bash
# DRAM-only baseline
kvoffloadbench replay --config configs/runs/dram_only.yaml

# DRAM -> NVMe -> TCP target
kvoffloadbench replay --config configs/runs/dram_nvme_remote.yaml \
  --metrics-port 9100
```

Open `http://<host>:9100/metrics` or scrape it with Prometheus. Each run writes
`results/raw/<run_id>/{requests.parquet,metrics.json,tier_stats.json,config.yaml,manifest.json}`.

## 7. Real-engine configuration on 4x H200

Edit the sweep's `engine:` block (or make a copy under `configs/sweeps/`):

```yaml
engine:
  mode: real
  model: meta-llama/Llama-3.3-70B-Instruct
  tensor_parallel: 4
  max_model_len: 32768
  connector: lmcache          # or nixl for the Dynamo path
  gpu_memory_utilization: 0.90
  max_num_seqs: 64
```

For NIXL:

```yaml
engine:
  mode: real
  connector: nixl
  nixl_agent_config:
    agent_name: kvob-agent
    # the NIXL connector passes this through as kv_connector_extra_config
```

LMCache is configured at runtime from the tier list (the adapter writes
`results/cache/lmcache/<run_id>.yaml` and exports `LMCACHE_CONFIG_FILE` and
`VLLM_USE_V1=1`): `local_cpu`, `max_local_cpu_size`, `local_disk`,
`max_local_disk_size`, `remote_url`, `remote_serde`.

Make the sweep use the real NVMe-oF block tier by adding this chain to
`configs/sweeps/main.yaml` (and to the include blocks):

```yaml
chains:
  dram_nvme_nvmeof:
    - {kind: hbm, capacity_mib: 8192, block_size_kib: 128}
    - {kind: dram, capacity_mib: "${kv_cache_capacity_mib}", block_size_kib: 128}
    - {kind: nvmeof_block, capacity_mib: 65536, block_size_kib: 128,
       extra: {device_env: KVOB_NVMEOF_DEV}}
    - {kind: remote_tcp, capacity_mib: 262144, block_size_kib: 128,
       endpoint: "10.0.0.10:4420", queue_depth: 32}
```

Notes for the real engine:

- The `hbm` tier is only constructible when CUDA is visible; it is optional.
- For 32k contexts a single 70B request holds ~10 GiB of KV; keep at least
  64 GiB of DRAM/LMCache per rank or the experiment measures eviction, not the
  SLO.
- Start with `duration_s: 60`, `warmup_s: 5`, `max_concurrency: 32`, then raise
  concurrency until goodput saturates.
- Pin the process (`numactl --cpunodebind=0 --membind=0`) and isolate the
  storage NIC IRQs.

## 8. Full experiment matrix

```bash
kvoffloadbench sweep --config configs/sweeps/main.yaml
kvoffloadbench analyze --results results/
```

Expected artifacts:

```
results/tiers/microbench.csv
results/index.csv
results/raw/<64 run ids>/{requests.parquet,metrics.json,tier_stats.json,config.yaml,manifest.json}
results/validation/kv_correctness.json
results/calibration/engine_calibration.json
results/tables/bottleneck_attribution.{csv,md}
results/tables/storage_requirements.{csv,md}
results/figures/{ttft_vs_rate_by_tier,goodput_vs_rate_by_tier,
                 hit_rate_vs_cache_size,ttft_cdf_by_tier,
                 restore_bandwidth_vs_context,tier_microbench_latency}.png
```

`analyze` is deterministic from committed results: rerunning it must not change
the tables or figure contents for the same inputs.

## 9. Docker / Compose

```bash
docker compose build
docker compose up target prometheus grafana
docker compose run --rm engine env
docker compose up engine          # replay catalogued in configs/runs/compose.yaml
```

- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (admin / kvoffloadbench), dashboard
  "KVOffloadBench serving".

The engine service requests all GPUs and the target service owns a named volume
for its backing file. `configs/runs/compose.yaml` points the remote tier at
`target:4420`.

## 10. Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| Engine mode is `emulated` on the GPU host | `kvoffloadbench env` shows `cuda_available`, `vllm` version | install `[gpu]`, check `torch.cuda.is_available()`; explicit `mode: real` will fail loudly instead of degrading |
| `nvme connect` hangs/fails | `dmesg`, `nvmetcli ls`, `ss -ltnp | grep 4420`, GID index | fix `addr_trtype`/`addr_traddr`, use `--host-traddr`, open the fabric VLAN |
| `nvmeof_block` tier silently absent | run log warning, `KVOB_NVMEOF_DEV` | export the device printed by the setup script; the tier deliberately degrades to `remote_tcp` and records `nvmeof_fell_back` |
| LMCache not used | `VLLM_USE_V1=1`, connector name | vLLM V1 uses `LMCacheConnectorV1`; older builds use `LMCacheConnector` |
| OOM in vLLM | `nvidia-smi`, `gpu_memory_utilization` | lower utilization / `max_num_seqs`, or raise TP to 4 |
| p99 restore latency high | `results/tiers/microbench.csv`, target queue depth | raise queue depth, check PFC/ECN, move target closer, use larger block sizes |
| `goodput=0` with high hit rate | `results/tables/bottleneck_attribution.md` | check whether calibrated compute or queueing dominates; the table names the binding resource |

## 11. Measurement definitions

- **TTFT**: arrival timestamp at the harness to the first token event.
- **TPOT**: `(last_token_ts - first_token_ts) / (output_tokens - 1)`.
- **ITL**: consecutive token timestamp differences.
- **Goodput**: requests with `TTFT <= slo.ttft_ms AND TPOT <= slo.tpot_ms`
  (inclusive boundary), reported as req/s and as a fraction of attempted.
- **Prefix-cache hit rate**: KV blocks found in the tier chain divided by
  prompt blocks looked up.
- **Restore bandwidth**: bytes fetched from any tier below the hot tier divided
  by the measured restore time inside `TieredKVStore.get`; the window-average
  rate is reported separately.
- **Warmup**: runs before the measured window; counters are reset afterwards so
  warmup traffic is not in the reported statistics.

## 12. Hardware caveats

- This build's compute timings come from gpt2 on an M3 MPS device. They are not
  H200 timings. Every result row says `mode=emulated`; no result claims to be a
  real vLLM serving measurement.
- The TCP target is a Python asyncio server. Its throughput ceiling is a
  property of this implementation and is measured honestly in the microbench,
  not a property of NVMe-oF hardware.
- macOS has no `O_DIRECT`; the local NVMe tier uses `F_NOCACHE` and reports
  `io_mode=fnocache`. The `O_DIRECT` path is exercised by the Linux CI host and
  the GPU host.
- The local NVMe "device" is a preallocated file on the laptop SSD, not a raw
  namespace. It is real file I/O with cache bypass, which is the strongest
  statement this host supports.
