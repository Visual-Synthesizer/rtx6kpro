# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for
Local Inference Lab.</em></p>

This page specifies the qualified GLM-5.3-Flash deployment for four NVIDIA RTX
PRO 6000 Blackwell Workstation Edition GPUs. The runtime serves the
`local-inference-lab/GLM-5.3-Flash-NVFP4` target checkpoint without
speculation, with three-token Multi-Token Prediction (MTP), or with the
`local-inference-lab/GLM-5.3-Flash-DFlash2` draft checkpoint.

The commands use Hugging Face repository names and named Docker volumes. They
do not require checkpoint paths or source-code bind mounts.

## Status

| Capability | Status |
|---|---|
| Tensor parallelism of four with one decode-context rank | **qualified** for no speculation, MTP depth 3, and DFlash2 depth 7 |
| Tensor parallelism of four with four decode-context ranks | **qualified** for the same three serving modes, including complete-KV prefill |
| Two decode-context ranks | **implemented**; not independently performance-qualified for this artifact |
| Tensor parallelism of eight | **implemented**; not independently hardware-qualified for this artifact |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4`; Hugging Face `main` unless `MODEL_REVISION` is set |
| QAD research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity, verifier-backed behavior, and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified execution; inconclusive one-point decision** for the topology-matched BF16, published-NVFP4, and QAD checkpoints in the [VBF report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | FP8 by default; packed NVFP4 is selectable |
| GPU prefix cache | **qualified** with independently sized target and recurrent allocations |
| Native DRAM offload | **qualified** and opt-in with `CACHE_MODE=native` |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache` |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; execution-time compute-share fairness assigns 40% of contended model execution to prefill; scheduling interval 1 |
| Root filesystem | Two layers, within standard Docker overlay2 limits |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qualification date | 2026-09-04 |

The [BF16-to-NVFP4 distribution-fidelity report](../kld/glm-5.3-flash-bf16-nvfp4.md)
and [QAD step 1,750 comparison](../kld/glm-5.3-flash-qad-step1750.md)
are research-only. They measure a reproducible FlashInfer CUTLASS path rather
than the B12X serving path specified here.

The [AA-LCR result](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) qualifies the BF16,
published NVFP4, and QAD configurations on 100 long-context questions with
three independent generations each. The published checkpoint scores 74.00%,
QAD scores 73.00%, and BF16 scores 71.67%; paired evidence does not distinguish
the three complete configurations. The accompanying
[reproduction specification](glm-5.3-flash/aa-lcr-reproduction.md) fixes the
dataset, prompt, sampling, runtime, equality checker, and receipt validation.

The [Verifier-Backed Behavioral Fidelity report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md)
uses 224 deterministic tasks with executable answer keys and no language-model
judge. BF16 scores 93.11%, published NVFP4 scores 91.63%, and QAD step 1,750
scores 92.35% on the primary fractional metric. All paired one-point decisions
are inconclusive; the report separates qualified execution provenance from the
statistical power required to claim behavioral equivalence or improvement.

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260904-r22
voipmonitor/vllm@sha256:284784e685aa0377f1cf63a312a364fc884b02beb98949d6886624edbddb3806
```

The embedded source-lock SHA-256 is
`57789f330528d65c80f4ffac208beb63617b6c6ab1e077056b2a0fe992e997d8`.
The Docker digest fixes the runtime. Model repository names follow Hugging Face
`main` unless an optional revision variable is supplied.

## Runtime backends

| Operation | Selected implementation |
|---|---|
| Target sparse attention and C4 index selection | B12X |
| Target gated-delta-network prefill | FlashKDA by default; B12X KDA is explicitly selectable |
| Target gated-delta-network decode | B12X when eligible, with Triton fallback |
| Target routed experts | B12X NVFP4 W4A4 |
| Target linear layers | B12X |
| Tensor-parallel all-reduce | B12X PCIe one-shot/two-shot first; PyNCCL outside the qualified B12X ranges |
| MTP attention | B12X |
| MTP experts | Marlin |
| DFlash2 MXFP8 linear and fused key/value projections | B12X |
| DFlash2 local attention | Graph-safe split-KV FlashAttention |
| Sampling | FlashInfer |
| External cache | LMCache DRAM L1 and native-filesystem L2 when selected |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths.

## Qualified performance

The measurements used four RTX PRO 6000 Blackwell Workstation Edition GPUs
with PCIe Gen5 x16 links, tensor parallelism of four, FP8 target KV cache, a
4,096-token scheduler budget, full and piecewise decode graphs, exactly 16 NCCL
channels, and a 2 MiB NCCL buffer. Each prefill value is a 30-second 32K-token
measurement after a 30-second warmup. C1 and C8 are 30-second sustained
context-zero decode cells. Other serving workloads were active on the host, so
the table qualifies regression safety and gives conservative absolute decode
rates; it is not an isolated peak-throughput sweep.

| DCP | Serving mode | KV capacity | 32K prefill tok/s | C1 output tok/s | C1 steps/s | C1 accepted/step | C8 output tok/s | C8 steps/s | C8 accepted/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | No speculation | 4,810,752 | 15,318 | 177.6 | — | — | 803.9 | — | — |
| 1 | MTP, depth 3 | 3,840,000 | 14,986 | 293.1 | 119.3 | 2.46 | 1,032.7 | 409.1 | 2.52 |
| 1 | DFlash2, depth 7 | 4,034,560 | 15,101 | 237.7 | 97.2 | 2.45 | 759.0 | 321.8 | 2.36 |
| 4 | No speculation | 21,929,984 | 13,719 | 168.9 | — | — | 761.5 | — | — |
| 4 | MTP, depth 3 | 17,858,560 | 13,315 | 267.1 | 105.6 | 2.53 | 921.3 | 373.3 | 2.47 |
| 4 | DFlash2, depth 7 | 18,604,032 | 13,469 | 230.1 | 91.8 | 2.51 | 683.4 | 290.9 | 2.35 |

Speculative output throughput varies with accepted length. Target steps per
second isolates target-model execution speed.

### Complete-KV prefill

With four decode-context ranks, every rank must select sparse-attention
candidates from the complete target KV sequence rather than its rank-local
quarter. The qualified implementation gathers the complete target KV view and
keeps recurrent-state cache ownership independent. A feature-isolation test in
the `20260903-r20` runtime measured 12,999 prompt tok/s with complete-KV
selection versus 9,858 prompt tok/s with rank-local selection, a 31.86%
increase. The R22 runtime measured 13,469 prompt tok/s for its
DFlash2 DCP4 qualification cell.

### Memory-weighted cache allocation

GLM-5.3 target attention and recurrent state have different bytes-per-request
costs. The runtime partitions compatible layers by their actual cost instead
of forcing equal layer counts into each shared pool. The deterministic search
is capped at eight cache groups, preserving bounded scheduler and connector
overhead.

For TP4/DCP4 without speculation, the allocator increased usable KV capacity
from 20,873,216 tokens in the `20260903-r20` artifact to 21,929,984 tokens:
1,056,768 additional tokens, or 5.06%. The DCP1 DFlash2 A/B remained within run
noise: 32K prefill changed from 15,151 to 15,138 tok/s, C1 verifier throughput
from 101.6 to 103.3 steps/s, and C8 verifier throughput from 332.1 to 327.3
steps/s.

### Compute-share fairness

The scheduler measures model-execution time separately for prefill and decode
and targets 40% prefill share only while both classes are contending. A mixed
C8 DFlash2 decode and cold 32K-prefill run assigned 42.97% of measured
contended execution time to prefill. Forward passes are indivisible, so short
runs oscillate around the configured target.

Set `FAIRNESS_ENGINE=none` to disable fairness. Compute-share fairness requires
`PREFILL_SCHEDULE_INTERVAL=1`; the launcher rejects incompatible values rather
than silently changing scheduling behavior.

### R21 compatibility and FlashKDA correction

R22 changes the FlashKDA source pin and compiled extension; vLLM Python, B12X,
LMCache, scheduling, cache geometry, and decode kernels are unchanged. Across
no speculation, MTP depth 3, and DFlash2 depth 7 under DCP1 and DCP4, the six
32K prefill cells changed by -0.24% to +1.37% relative to the R21
qualification.

A direct no-speculation decode control under the same host load measured R21
at 178.2 tok/s for C1 and 814.6 tok/s for C8, versus 177.6 and 803.9 tok/s for
R22. The differences are -0.34% and -1.31%, which establishes no attributable
decode regression.

The deterministic 16,384-token near-collinear BF16-key reproducer found 55,264
non-finite output elements and 16,000 non-finite recurrent-state elements in
the R21 extension. R22 produced zero non-finite output or state elements for
identical inputs. The R22 server also completed 24 long concurrent TP4/DCP4
MTP3 requests without a wrong or runaway result.

### Gated-delta-network prefill backend

B12X KDA remains qualified as an explicit backend. FlashKDA is the default.
The stable FlashKDA extension measured 102.98 microseconds at the packed TP4/C4
kernel shape versus 105.03 microseconds for the numerically unstable extension,
a 1.95% kernel-time reduction.

## Start the server

Select a serving mode and run the common command. The image already contains
the qualified B12X, full-and-piecewise CUDA graph, FlashInfer sampler,
16-channel NCCL, 2 MiB NCCL buffer, and one-thread OpenMP defaults. No
`CUDAGRAPH_MODE` override is required.

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260904-r22
GPU_DEVICES=0,1,2,3
PORT=8000
docker pull "$IMAGE"
```

Ordinary serving without speculative tokens:

```bash
NAME=jovian-judgement-nospec
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
```

Three-token MTP:

```bash
NAME=jovian-judgement-mtp3
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
```

DFlash2 with its trained seven-draft-token configuration:

```bash
NAME=jovian-judgement-dflash2
MODE_ARGS=(
  -e SPECULATOR=dflash2
  -e DFLASH_DEPTH=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2
)
```

Common GPU-cache command:

```bash
docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e CACHE_MODE=vram \
  -e KV_CACHE_QUANT=fp8_ds_mla \
  -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e PORT="$PORT" \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=1 \
  -e FAIRNESS_ENGINE=compute_share \
  -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.93 \
  "${MODE_ARGS[@]}" \
  "$IMAGE"
```

For two or four decode-context ranks, replace `DCP=1` with `DCP=2` or `DCP=4`.
The launcher enables complete-KV gathering automatically when DCP is greater
than one. This applies to no speculation, MTP, and DFlash2.

To test the qualified B12X gated-delta-network prefill backend, add:

```bash
-e GLM53_KDA_PREFILL_BACKEND=b12x
```

The default `GLM53_KDA_PREFILL_BACKEND=flashkda` is faster or equal in the
qualified configurations.

### Cache page geometry

The launcher owns cache page geometry; normal deployments should not pass a
page-size argument. GPU-only and native-offload modes use 2,048-token target
and recurrent pages. LMCache derives per-rank pages from its 4,096-token object
size and the selected DCP value. A DFlash2 sliding window smaller than one
engine page is transferred as one complete engine page so no live cache bytes
are omitted.

The 2,048-token GPU page increases usable KV capacity without a measurable
prefill or decode regression. It does not change the public vLLM attention
block size, which remains 256 tokens.

## Native DRAM offload

Use the common server command with these cache settings:

```bash
-e CACHE_MODE=native
-e NATIVE_KV_OFFLOADING_SIZE_GB=64
```

The launcher enables the shareable cuMem allocator required by the native
offload backend.

## LMCache DRAM and filesystem storage

LMCache uses a sidecar process in the same container. It stores complete
4,096-token objects in DRAM and optionally in a mounted filesystem. Use a
shared-memory allocation of at least 96 GiB; 128 GiB is the qualified setting.

Replace the cache settings and shared-memory size in the common command with:

```bash
--shm-size 128g
-v jovian-judgement-lmcache-l2:/lmcache-l2
-e CACHE_MODE=lmcache
-e KV_CACHE_QUANT=fp8_ds_mla
-e LMCACHE_CHUNK_SIZE=4096
-e LMCACHE_TARGET_TOKEN_BUDGET=4096
-e LMCACHE_L1_SIZE_GB=64
-e LMCACHE_L2_ENABLED=1
-e LMCACHE_L2_ROOT=/lmcache-l2
```

`KV_CACHE_QUANT=nvfp4_ds_mla` selects the qualified packed-NVFP4 target cache
instead. The filesystem namespace includes the target and draft revisions,
cache format, parallelism, DCP gathering policy, speculation mode, and object
size, preventing incompatible cache objects from being reused.

Qualification covered cold compute, vLLM automatic prefix reuse, LMCache DRAM
restore, and filesystem restore for FP8 and NVFP4. Both formats restored 12,288
prompt tokens after restarting vLLM and the LMCache sidecar. FP8 DFlash2 also
restored 12,288 tokens from DRAM and filesystem tiers without a service
restart. External bytes were compared on every tensor-parallel rank across the
target attention, recurrent-state, and DFlash sliding-attention groups.

## Source and review contract

| Component | Qualified source |
|---|---|
| vLLM | [R22 source](https://github.com/local-inference-lab/vllm/tree/artifact/jovian-judgement-community-20260904-r22-source); commit `70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19`; tree `89481110674c08be1759a9222c525a0be14ad52a`; package tree `4fbb1c257ac59e5e68450655ad4061d2c8a05e5c` |
| FlashKDA | commit `3b225bf26bb8e218928a1fe14751cb48cf31d11b`; extension SHA-256 `16aece5ffb83c2dfb0355758bbbc9d6e0ea50a2cfc36ecee4936607d445aba0a` |
| B12X | commit `1e59a1fd09f782d302b1068b15c8a0bd66103894`; tree `f322c804eec1c58a63bd4fe6e7901a95a678a575`; package tree `aaa5f189acae0206d886553421f6e9044f4c458a` |
| LMCache | commit `aefe3ab701ab7a835532e701be89f5055b13ec0f`; tree `683ab2c165a9aa0e2d1a1ab757af4a8b193688c5`; package tree `976a97f22c0497f34db089dc5f02a713dd0b5888` |

The [vLLM merge checklist](https://github.com/local-inference-lab/vllm/issues/590)
lists each open pull request, dependency, resulting behavior, attribution, and
qualification result. The image embeds the same source contract at
`/opt/glm53-flash/source.lock`.
