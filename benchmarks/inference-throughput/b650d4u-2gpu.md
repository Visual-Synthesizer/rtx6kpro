# 2× RTX PRO 6000 Blackwell — B650D4U + PLX Gen5 Switch

Inference throughput benchmarks for a 2-GPU Blackwell rig sharing a PLX
PM50100 Gen5 switch (PIX topology). Distinct from the 4× and 8× GPU
measurements on the 397B model elsewhere in this repo — different
scaling regime, different flag set, different bottlenecks.

## Test Environment

| Parameter | Value |
|-----------|-------|
| **Board** | AsRock Rack B650D4U-2L2T/BCM, BIOS 22.13 |
| **CPU** | AMD EPYC 4564P (16c Zen4c) |
| **RAM** | 128 GB DDR5 ECC |
| **GPUs** | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W) |
| **PCIe fabric** | c-payne PM50100 PLX Gen5 switch, PIX topology, both GPUs behind the switch |
| **Storage** | 7.3 TB local NVMe (models on-device, never NFS) |
| **Engine** | SGLang via `voipmonitor/sglang:cu130` docker image (b12x 0.8.3) |
| **Benchmark tool** | `benchmark_sglang.py` (this repo) |
| **Power limit** | 600 W per GPU, CPU performance governor |
| **Date** | 2026-04-12 |
| **Methodology** | 3x full-server-restart iterations averaged, 5-request warmup per run, 30 s per cell |

## Headline results

### MiniMax-M2.7 NVFP4 (lukealonso/MiniMax-M2.7-NVFP4, modelopt_fp4)

No speculative decoder available yet — these are non-speculative numbers.
When a NEXTN drafter ships, expect meaningful uplift at low concurrency.

Aggregate throughput, tok/s (ctx=0):

| C=1 | C=2 | C=4 | C=8 | C=16 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|---|---|
| **127.7** ± 0.0 | 175.3 ± 17.8 | 340.3 ± 0.1 | 471.6 ± 15.4 | 742.7 ± 0.0 | 1078.9 ± 15.4 | 1695.4 ± 20.3 | **2800.2** ± 20.2 |

Prefill (C=1, baseline TTFT subtracted):

| ctx | TTFT | tok/s |
|---|---|---|
| 8K | 0.50 s | 17,286 |
| 16K | 0.99 s | 16,926 |
| 32K | 2.09 s | 15,861 |
| 64K | 4.94 s | 13,319 |
| 128K | 13.25 s | 9,908 |

Raw JSON: [`sglang_m27_b12x_0.8.3_3x_mean.json`](sglang_m27_b12x_0.8.3_3x_mean.json)

### Qwen3.5-122B-A10B NVFP4 (txn545 modelopt_fp4) + NEXTN

SGLang speculative decoding with NEXTN (5 steps, 6 draft tokens, eagle-topk=1).

Aggregate throughput, tok/s (ctx=0):

| C=1 | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---|---|---|---|---|
| **194.0** ± 2.8 | 327.6 ± 1.4 | 522.2 ± 0.9 | 813.4 ± 1.5 | 994.9 ± 56.6 | **1411.5** ± 2.7 |

Higher concurrencies not measured — NEXTN cudagraph configured for max
batch 32 on this setup.

Raw JSON: [`sglang_qwen35-122b_b12x_nextn_0.8.3_3x_mean.json`](sglang_qwen35-122b_b12x_nextn_0.8.3_3x_mean.json)

## Key configuration notes

Our launch scripts use Luke Alonso's canonical b12x recipe. Two
deliberate choices worth calling out, both validated by A/B this session:

1. **No `--enable-pcie-oneshot-allreduce`.** On b12x 0.7.x this flag was
   a ~5 % win on 2-GPU PIX topology. On b12x 0.8.3 it's a net **-3 to
   -7 %** loss on both M2.7 and 122B (NEXTN) — the default NCCL ring
   allreduce has improved enough that oneshot's O(N²) direct writes are
   no longer competitive at decode batch sizes. Luke's public recipes
   already omit the flag.

2. **No `--enable-pcie-oneshot-allreduce-fusion`.** Fuses the allreduce
   with the following RMSnorm, but the fusion kernel is a pybind11 C++
   extension that `torch.compile`/Dynamo cannot trace. Causes piecewise
   cudagraph warmup crashes on M2.7 specifically, so safest to omit on
   all M2.x models.

## KV cache budget caveat

On 2-GPU with `--mem-fraction-static 0.85`:

| Model | KV pool (total tokens) |
|---|---|
| M2.7 (bf16 KV) | ~83 K |
| 122B (fp8 KV) | larger, not the bottleneck |

Cells where `concurrency × (ctx + max_tokens) > KV pool` get pre-skipped
by the benchmark's auto-detection. This is why the M2.7 sweep is
sparser at long context × high concurrency. To force every cell
(including thrashing ones that queue-starve), pass
`--max-total-tokens 2147483647` explicitly — the documented
`--max-total-tokens 0` actually means "auto-detect from server", not
"no limit", which surprised us while writing these benchmarks.

## Methodology notes and known caveats

- **FlashInfer autotuner variance.** Fresh JIT caches pick different
  kernels on first run. 3x averaging with full server restart between
  iterations is necessary to get reproducible numbers. Single runs can
  differ by ±15 % on the same config.
- **SGLang radix prefix cache contaminates sequential ctx sweeps.** A
  16 K context test that runs right after an 8 K test may measure
  near-instant decode because the prefix is cached. Our ctx=16K numbers
  in M2.7 above reflect this partial contamination — treat ctx=0 as the
  clean baseline and assume real-world 16 K performance sits between
  the two rows.
- **Environment drift is real.** A 122B measurement published earlier
  this month at 198 tok/s C=1 / 1,523 C=32 (same hardware, same flags,
  same b12x 0.7.2 image) could not be reproduced the night of
  2026-04-12 — warm-cache rerun landed at 181 / 1,421 on 0.7.2, close
  to 194 / 1,411 on 0.8.3. Something drifted (driver, kernel,
  autotuner state, thermal) over the weeks between runs. Reported
  numbers are the *currently reproducible* baseline; historical
  higher values appear to have been environment-favorable peaks.

## Hardware / firmware tuning in effect

- Kernel boot params: `pci=noacs,realloc iommu=pt mitigations=off pcie_aspm=off`
- `/etc/modprobe.d/uvm.conf`: `options nvidia_uvm uvm_disable_hmm=1`
- **No `ForceP2P` modprobe override** — PIX topology with the PLX enables
  BAR1 P2P by default; adding `ForceP2P=0x11` (which WRX90 / direct-attach
  rigs need) would break auto-crossover detection here.
- GPUs are put in persistence mode and set to 600 W TDP after every boot.
- CPU `performance` governor pinned via `/etc/tmpfiles.d/cpu-governor.conf`
  (+5 % at C=1 vs `schedutil`).

See the top-level `hardware/` directory of this repo for PCIe topology
and P2P bandwidth/latency measurements.
