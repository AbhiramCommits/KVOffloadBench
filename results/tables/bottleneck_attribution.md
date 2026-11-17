# Bottleneck attribution

Generated from every run in `results/index.csv` and `results/raw/**` by
`kvoffloadbench analyze`. All storage time is measured I/O; compute time is the
calibrated model timing recorded in `results/calibration/engine_calibration.json`.

## Method

- Per-request decomposition: queue wait (measured in the harness) + calibrated prefill/decode
  compute + storage restore time (measured inside `TieredKVStore.get`) + remainder.
- Shares are relative to measured TTFT p50; utilizations use wall time excluding warmup
  multiplied by `max_concurrency`.
- Achieved restore bandwidth/IOPS are compared with the best measured read ceiling from
  `results/tiers/microbench.csv` at the 64 KiB or 256 KiB point (the KV block is 96 KiB).
- A run that meets the SLO with a large headroom is labelled `slo_headroom`; the tool does
  not invent a bottleneck when none is binding.

## Measured tier ceilings used

| tier | read GiB/s | block KiB | depth | IOPS | IOPS block KiB | I/O mode |
|---|---:|---:|---:|---:|---:|---|
| dram | 12.186 | 256 | 32 | 113459 | 64 | dram |
| nvme_local | 3.557 | 256 | 32 | 24407 | 64 | fnocache |
| remote_tcp | 1.647 | 256 | 32 | 11259 | 64 | tcp |

## Binding resource counts

| binding resource | runs |
|---|---:|
| slo_headroom | 34 |
| compute_calibrated | 20 |
| queueing_concurrency | 12 |

## Runs with the highest measured TTFT p99

| run | tiers | trace | rate | ctx | TTFT p99 ms | goodput frac | restore GiB/s | achieved/ceiling | binding |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `main-dram_nvme_remote_tcp-r8-c32768-kmedium-shared_prefix_rag` | dram+nvme+remote_tcp | shared_prefix_rag.jsonl | 8 | 32768 | 62484 | 0.08 | 0.182 | 0.18 | queueing_concurrency |
| `main-dram_nvme_remote_tcp-r8-c32768-kmedium-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 8 | 32768 | 30662 | 0.13 | 0.695 | 0.67 | queueing_concurrency |
| `main-dram_only-r8-c32768-kmedium-longctx_multiturn` | dram | longctx_multiturn.jsonl | 8 | 32768 | 21357 | 0.12 | 0.000 | 0.00 | queueing_concurrency |
| `main-dram_nvme_remote_tcp-r4-c16384-klarge-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 4 | 16384 | 21173 | 0.12 | 0.000 | 0.00 | queueing_concurrency |
| `main-dram_nvme-r8-c32768-kmedium-longctx_multiturn` | dram+nvme | longctx_multiturn.jsonl | 8 | 32768 | 21121 | 0.17 | 0.733 | 0.33 | queueing_concurrency |
| `main-dram_only-r8-c32768-kmedium-shared_prefix_rag` | dram | shared_prefix_rag.jsonl | 8 | 32768 | 17304 | 0.14 | 0.000 | 0.00 | queueing_concurrency |
| `main-dram_nvme-r8-c32768-kmedium-shared_prefix_rag` | dram+nvme | shared_prefix_rag.jsonl | 8 | 32768 | 17192 | 0.22 | 0.668 | 0.30 | queueing_concurrency |
| `main-dram_nvme_remote_tcp-r8-c16384-kmedium-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 8 | 16384 | 16476 | 0.14 | 0.663 | 0.64 | queueing_concurrency |
| `main-dram_nvme_remote_tcp-r4-c16384-ksmall-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 4 | 16384 | 13685 | 0.16 | 0.240 | 0.23 | compute_calibrated |
| `main-dram_nvme_remote_tcp-r4-c32768-kmedium-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 4 | 32768 | 13603 | 0.22 | 0.000 | 0.00 | compute_calibrated |
| `main-dram_only-r8-c16384-kmedium-longctx_multiturn` | dram | longctx_multiturn.jsonl | 8 | 16384 | 12130 | 0.22 | 0.000 | 0.00 | queueing_concurrency |
| `main-dram_nvme-r8-c16384-kmedium-longctx_multiturn` | dram+nvme | longctx_multiturn.jsonl | 8 | 16384 | 11845 | 0.28 | 0.652 | 0.29 | queueing_concurrency |
| `main-dram_nvme-r4-c32768-kmedium-longctx_multiturn` | dram+nvme | longctx_multiturn.jsonl | 4 | 32768 | 8953 | 0.45 | 0.000 | 0.00 | compute_calibrated |
| `main-dram_only-r4-c32768-kmedium-longctx_multiturn` | dram | longctx_multiturn.jsonl | 4 | 32768 | 8886 | 0.45 | 0.000 | 0.00 | compute_calibrated |
| `main-dram_nvme_remote_tcp-r2-c32768-kmedium-longctx_multiturn` | dram+nvme+remote_tcp | longctx_multiturn.jsonl | 2 | 32768 | 8840 | 0.43 | 0.000 | 0.00 | compute_calibrated |

## Interpretation

The emulated compute model uses this laptop's measured gpt2 timings, so `compute_calibrated`
labels mean the calibrated prefill/decode budget dominates, not that an H200 was saturated.
`queueing_concurrency` means arrivals outran the in-flight limit before storage could be blamed.
`storage_*` labels are reserved for runs whose measured restore rate actually approached the
measured tier ceiling; the achieved/ceiling column makes that comparison explicit.
