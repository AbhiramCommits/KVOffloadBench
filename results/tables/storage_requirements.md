# Storage requirements for KV offload

Derived by `kvoffloadbench analyze` from the measured sweep (`results/index.csv`) and the Step-2 tier microbenchmark.

## Formulas

```
restore_bytes_per_request = context_tokens * bytes_per_token
storage_blocks             = ceil(restore_bytes_per_request / 128 KiB)
required_read_bandwidth     = restore_bytes_per_request / TTFT_budget
sustained_read_bandwidth    = restore_bytes_per_request * request_rate
required_IOPS               = storage_blocks / TTFT_budget
p99_block_latency_budget    = TTFT_budget / storage_blocks
```

- TTFT budget: **4000 ms** (the SLO used by the sweep).
- Local declared shape: 12 layers, 2 KV heads, head_dim 64, float16 -> **6,144 B/token** (98,304 B per 16-token KV block).
- Local compute budget uses the measured prefill **0.2139 ms/token** on `mps` from `results/calibration/engine_calibration.json`.
- A full prefix miss is assumed; prefix hits reduce the bytes actually restored, which is
  why the measured restore requirement can be lower than this worst case.
- `p99 ... qd8` relaxes the per-block budget by pipelining 8 in-flight block reads.

## Local shape (as measured in this build)

| ctx | rate | KV MiB/req | blocks | compute ms | I/O budget ms | req GiB/s | sustained GiB/s | req IOPS | p99 budget us | best measured GiB/s | chain | meets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4096 | 1 | 24.0 | 192 | 876 | 3124 | 0.008 | 0.02 | 61 | 16270 | 0.000 | dram_nvme | - |
| 4096 | 2 | 24.0 | 192 | 876 | 3124 | 0.008 | 0.05 | 61 | 16270 | 0.000 | dram_nvme | - |
| 4096 | 4 | 24.0 | 192 | 876 | 3124 | 0.008 | 0.09 | 61 | 16270 | 0.000 | dram_nvme | - |
| 4096 | 8 | 24.0 | 192 | 876 | 3124 | 0.008 | 0.19 | 61 | 16270 | 0.000 | dram_nvme | - |
| 16384 | 1 | 96.0 | 768 | 3504 | 496 | 0.189 | 0.09 | 1550 | 645 | 0.000 | dram_nvme | - |
| 16384 | 2 | 96.0 | 768 | 3504 | 496 | 0.189 | 0.19 | 1550 | 645 | 0.000 | dram_nvme | - |
| 16384 | 4 | 96.0 | 768 | 3504 | 496 | 0.189 | 0.38 | 1550 | 645 | 0.266 | dram_nvme | True |
| 16384 | 8 | 96.0 | 768 | 3504 | 496 | 0.189 | 0.75 | 1550 | 645 | 0.663 | dram_nvme_remote_tcp | True |
| 32768 | 1 | 192.0 | 1536 | 7009 | 0 | - | 0.19 | - | 0 | 0.000 | dram_nvme | - |
| 32768 | 2 | 192.0 | 1536 | 7009 | 0 | - | 0.38 | - | 0 | 0.000 | dram_nvme | - |
| 32768 | 4 | 192.0 | 1536 | 7009 | 0 | - | 0.75 | - | 0 | 0.000 | dram_nvme | - |
| 32768 | 8 | 192.0 | 1536 | 7009 | 0 | - | 1.50 | - | 0 | 0.733 | dram_nvme | - |

Where the calibrated local compute alone exceeds the 4 s TTFT budget (32k context),
the I/O budget is zero and the requirement columns show `-`: that is an honest statement
that this host's calibrated prefill cannot meet the SLO even with infinitely fast storage.

## Scaled to Llama-3.3-70B on 4x H200 (TP=4)

| ctx | rate | KV GiB/req | per-rank GiB/req | blocks | req GiB/s aggregate | per-rank GiB/s | sustained GiB/s | req IOPS | p99 budget us (@QD1 / @QD8) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 1.25 | 0.31 | 10240 | 0.31 | 0.078 | 1.2 | 2560 | 390.6 / 3125.0 |
| 4096 | 2 | 1.25 | 0.31 | 10240 | 0.31 | 0.078 | 2.5 | 2560 | 390.6 / 3125.0 |
| 4096 | 4 | 1.25 | 0.31 | 10240 | 0.31 | 0.078 | 5.0 | 2560 | 390.6 / 3125.0 |
| 4096 | 8 | 1.25 | 0.31 | 10240 | 0.31 | 0.078 | 10.0 | 2560 | 390.6 / 3125.0 |
| 16384 | 1 | 5.00 | 1.25 | 40960 | 1.25 | 0.312 | 5.0 | 10240 | 97.7 / 781.2 |
| 16384 | 2 | 5.00 | 1.25 | 40960 | 1.25 | 0.312 | 10.0 | 10240 | 97.7 / 781.2 |
| 16384 | 4 | 5.00 | 1.25 | 40960 | 1.25 | 0.312 | 20.0 | 10240 | 97.7 / 781.2 |
| 16384 | 8 | 5.00 | 1.25 | 40960 | 1.25 | 0.312 | 40.0 | 10240 | 97.7 / 781.2 |
| 32768 | 1 | 10.00 | 2.50 | 81920 | 2.50 | 0.625 | 10.0 | 20480 | 48.8 / 390.6 |
| 32768 | 2 | 10.00 | 2.50 | 81920 | 2.50 | 0.625 | 20.0 | 20480 | 48.8 / 390.6 |
| 32768 | 4 | 10.00 | 2.50 | 81920 | 2.50 | 0.625 | 40.0 | 20480 | 48.8 / 390.6 |
| 32768 | 8 | 10.00 | 2.50 | 81920 | 2.50 | 0.625 | 80.0 | 20480 | 48.8 / 390.6 |

### Scaling assumptions (stated inline)

- The 70B KV shape is 80 layers x 2 (K and V) x 8 KV heads x 128 head_dim x 2 bytes
  = **327,680 B/token total**; tensor parallel 4 splits it evenly, so each rank stores
  81,920 B/token and must move one quarter of the bytes.
- Restore is assumed to be a full prefix miss, and the full 4 s TTFT budget is allocated
  to storage. On real H200 hardware prefill consumes part of that budget; subtract the
  measured prefill for the target model before sizing the SSD.
- `sustained` is the worst-case aggregate for restoring every request with no prefix
  reuse: bytes/request x request_rate. Prefix caching and KV reuse are what make the
  deployment feasible; the measured hit rates in `results/index.csv` quantify the gap.
- IOPS use 128 KiB storage blocks. A 70B 16-token KV block is 5 MiB, i.e. 40 storage
  blocks per KV block. Larger I/O requests (or NVMe-oF with larger MDTS) reduce the
  IOPS requirement proportionally.
- NVMe-oF adds a network hop: the p99 budget is per 128 KiB read at the host, so the
  target's p99 plus the fabric RTT must fit inside it.
