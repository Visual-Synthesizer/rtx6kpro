# Qwen3.8-Flash-Next: independent validation on 1× RTX PRO 6000 Workstation @600 W

Status: **independent community reproduction** of the TP1 engine comparison and the
mratsim turbo path, on Workstation-edition silicon at 600 W with stock clocks. Measured
2026-09-04 → 2026-09-10. Contributed from a B650D4U + c-payne PM50100 (PIX) host,
driver 610.57.04, CUDA 13.3, ECC enabled, Gen5 links verified before every run.

## Turbo path (mratsim/sglang-qwen38fn-sm120-turbo) — reproduced

The [TP1 engine comparison](tp1-engine-comparison.md) lists the turbo numbers as
research-only, not independently reproduced. We reproduce them on different silicon,
power class and prompts. Checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`, image built
from the fork at r22, `--ple-offload-embedding`, NEXTN-3, FP8 KV, stock memory clocks.

| Measurement (single request, tok/s) | This host (WS 600 W, stock) | mratsim reported (360 W, +6000 mem OC) |
|---|---:|---:|
| llm-inference-bench `lavd` | 310.1 | ~355 |
| llm-inference-bench `estonia` | 267.7 | 206–234 ("hard" band) |
| generic hard synthetic (30 s cells) | 191–195 | — |

Quality on the same runs: estonia 30/30 and 30/30 (100 %), lavd 9/10 + 3/3.
Aggregate 528 tok/s at C=4 (the profile's `max-running` cap). KV budget observed
896,320 tokens with the PLE table in host RAM.

## vLLM Compose recipe — reproduced within ~1 %

Same tool and settings as the qualified comparison (llm-inference-bench, ctx 0, 30 s
windows). Image `localinferencelab/vllm:jovian-judgement-community-20260910-r34`,
checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, recipe defaults.

| Aggregate tok/s | C1 | C2 | C4 | C8 | C16 |
|---|---:|---:|---:|---:|---:|
| This host (WS 600 W) | 174.5 | 307.0 | 461.7 | 631.9 | 942.4 |
| Qualified comparison | 172.8 | 304.3 | 485.4 | 632.4 | 944.5 |

## `VLLM_PLE_TABLE_MEMORY=mmap` — no penalty confirmed

Same recipe, PLE table mmap'd from NVMe instead of pinned host RAM:
C1 178.6 (+2 %), C4 495.5 (+7 %) — within run variance, no regression. This
supports disk-backed engram tables as a viable path for larger Engram-family
checkpoints on RAM-constrained hosts.

## Engine roles on a 2-GPU host

SGLang turbo wins single-stream decode (191–310 by workload) and prefill (20–26k
tok/s vs ~15k); the vLLM recipe wins multi-stream ceiling (942 vs 528 aggregate).
On this host we serve interactive sessions on the turbo path and group workloads
on the vLLM recipe.

Raw run directories (JSON + logs) available on request; harness:
github.com/Visual-Synthesizer/rtx6kpro fork, `benchmarks/inference-throughput/`.

## Companion result: Qwen3.8-27B quantization A/B (same host)

`nvidia/Qwen3.8-27B-NVFP4` vs `huginnfork/Qwen3.8-27B-FP8`, identical serving
(vLLM 0.28.1 nightly, TP2, MTP3, FP8 KV, FlashInfer attention, 262,144 max len):

| | C1 ctx0 | C32 ctx0 (agg) | C2 @16k | Estonia | lavd |
|---|---:|---:|---:|---:|---:|
| nvidia NVFP4 | **135.1** | **2,405** | **286.4** | 100 % (30/30) | 100 % (10/10) |
| huginnfork FP8 | 100.0 | 1,921 | 194.5 | 100 % (30/30) | 100 % (10/10) |

NVFP4 is 22–47 % faster in every cell with identical MTP acceptance (0.40 vs 0.41)
and task parity on both accuracy profiles. Teacher-forced top-20 KLD vs the FP8
reference (33k wikitext tokens, eager, no spec decode): median 0.0061, mean 0.124,
P95 0.65 — i.e. **not lossless**: median-token agreement is backend-equivalence tier,
but a real tail of divergence sits in low-confidence tokens where task benchmarks
don't look. The Qwen3.6-era NVFP4 quality collapse does not reproduce; for
maximum-fidelity use prefer FP8, for throughput NVFP4 is a sound trade.
