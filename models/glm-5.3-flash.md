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
| QAD research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | FP8 by default; packed NVFP4 is selectable |
| GPU prefix cache | **qualified** with independently sized target and recurrent allocations |
| Native DRAM offload | **qualified** and opt-in with `CACHE_MODE=native` |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache` |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; concurrent-prefill interval 8 |
| Root filesystem | Two layers, within standard Docker overlay2 limits |
| Qualification date | 2026-09-03 |

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

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260903-r20
voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340
```

The embedded source-lock SHA-256 is
`9a6167d415d824e1707ba7df0dd5906e05c004f1ed2666f80f2f9e1ea9fde4be`.
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

The measurements used four stock-clock RTX PRO 6000 Blackwell Workstation
Edition GPUs with PCIe Gen5 x16 links, tensor parallelism of four, FP8 target
KV cache, a 4,096-token scheduler budget, full and piecewise decode graphs,
exactly 16 NCCL channels, and a 2 MiB NCCL buffer. Prefill values are medians of
three cold 32K-token requests. C1 and C8 decode values are coordinate-wise
medians of three 30-second context-zero samples. Each DCP4 C8 run was followed
by a short C1 request to validate the transition from an eight-request CUDA
graph to a one-request graph.

| DCP | Serving mode | 32K prefill tok/s | C1 output tok/s | C1 steps/s | C1 accepted/step | C8 output tok/s | C8 steps/s | C8 accepted/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | No speculation | 14,908 | 169.76 | — | — | 735.86 | — | — |
| 1 | MTP, depth 3 | 14,507 | 256.33 | 105.15 | 2.438 | 894.96 | 370.06 | 2.440 |
| 1 | DFlash2, depth 7 | 14,691 | 214.27 | 89.46 | 2.436 | 689.35 | 291.12 | 2.370 |
| 4 | No speculation | 13,019 | 151.46 | — | — | 666.02 | — | — |
| 4 | MTP, depth 3 | 12,838 | 234.70 | 93.67 | 2.513 | 843.97 | 341.79 | 2.471 |
| 4 | DFlash2, depth 7 | 13,028 | 207.92 | 81.12 | 2.563 | 645.85 | 263.11 | 2.456 |

Speculative output throughput varies with accepted length. Target steps per
second isolates target-model execution speed.

### Complete-KV prefill

With four decode-context ranks, every rank must select sparse-attention
candidates from the complete target KV sequence rather than its rank-local
quarter. The qualified implementation gathers the complete target KV view and
keeps recurrent-state cache ownership independent. On DFlash2 depth 7, the
implementation measured 12,999 prompt tok/s versus 9,858 prompt tok/s for
rank-local selection, a 31.86% increase. The packaged configuration measured
13,028 prompt tok/s in the three-run qualification above.

### Comparison with the 2026-09-02 community artifact

The public `jovian-judgement-community-20260902-r17` artifact used the same
stock GPU quartet and TP4/DCP1, but used 32 NCCL channels and 512-token split
cache pages. The comparison therefore measures complete artifacts rather than
one isolated source change.

| Workload | 20260902-r17 | 20260903-r20 | Change |
|---|---:|---:|---:|
| No-spec 32K prefill | 14,663 tok/s | 14,908 tok/s | +1.67% |
| MTP, depth 3, 32K prefill | 14,272 tok/s | 14,507 tok/s | +1.65% |
| DFlash2, depth 7, 32K prefill | 14,323 tok/s | 14,691 tok/s | +2.57% |
| No-spec C1 output | 163.43 tok/s | 169.76 tok/s | +3.87% |
| No-spec C8 output | 734.96 tok/s | 735.86 tok/s | +0.12% |
| MTP, depth 3, C1 target steps | 102.53 steps/s | 105.15 steps/s | +2.56% |
| MTP, depth 3, C8 target steps | 374.58 steps/s | 370.06 steps/s | -1.21% |
| DFlash2, depth 7, C1 target steps | 89.78 steps/s | 89.46 steps/s | -0.36% |
| DFlash2, depth 7, C8 target steps | 294.22 steps/s | 291.12 steps/s | -1.05% |

Speculative C8 output rates are 894.96 tok/s for MTP depth 3 and 689.35 tok/s
for DFlash2 depth 7. They are respectively 5.41% and 10.83% below the earlier
artifact because the measured accepted lengths fell from 2.538 to 2.440 and
from 2.613 to 2.370. Target-step throughput, which isolates runtime execution
from data-dependent proposal acceptance, changed by -1.21% and -1.05%.

### Gated-delta-network prefill backend

The B12X KDA comparison used the same source tree, hardware, clocks, scheduler,
and three 32K-token samples per cell.

| Decode-context configuration | B12X KDA | FlashKDA | B12X change |
|---|---:|---:|---:|
| One rank per request | 14,899 prompt tok/s | 14,900 prompt tok/s | -0.01% |
| Four ranks with complete-KV gathering | 13,106 prompt tok/s | 13,242 prompt tok/s | -1.03% |

B12X KDA is qualified as an explicit backend. FlashKDA remains the default
because it is tied with one decode-context rank and faster with four.

## Start the server

Select a serving mode and run the common command. The image already contains
the qualified B12X, full-and-piecewise CUDA graph, FlashInfer sampler,
16-channel NCCL, 2 MiB NCCL buffer, and one-thread OpenMP defaults. No
`CUDAGRAPH_MODE` override is required.

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260903-r20
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
  -e PREFILL_SCHEDULE_INTERVAL=8 \
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
restore, and filesystem restore after restarting both vLLM and the LMCache
sidecar. DFlash2 restored 12,288 prompt tokens from filesystem storage after a
full restart. External bytes were compared on every tensor-parallel rank across
the target attention, recurrent-state, and DFlash sliding-attention groups.

## Source and review contract

| Component | Qualified source |
|---|---|
| vLLM | commit `7015eb6949a93247df02fb6f9101d17c40bd83e8`; tree `456231387f6f5adc2d1a5241428f1226b29ea835`; package tree `83e4480f930da37a26f893871a55bcbf54493b3b` |
| B12X | commit `1e59a1fd09f782d302b1068b15c8a0bd66103894`; tree `f322c804eec1c58a63bd4fe6e7901a95a678a575`; package tree `aaa5f189acae0206d886553421f6e9044f4c458a` |
| LMCache | commit `aefe3ab701ab7a835532e701be89f5055b13ec0f`; tree `683ab2c165a9aa0e2d1a1ab757af4a8b193688c5`; package tree `976a97f22c0497f34db089dc5f02a713dd0b5888` |

The [vLLM merge checklist](https://github.com/local-inference-lab/vllm/issues/590)
lists each open pull request, dependency, resulting behavior, attribution, and
qualification result. The image embeds the same source contract at
`/opt/glm53-flash/source.lock`.
