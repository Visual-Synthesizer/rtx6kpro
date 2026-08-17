# GLM-5.2 on Infernal Invocation r17

**Status: qualified for projection-mixed EXL3 TP4/DCP1/MTP3.** The published
image loaded the pinned checkpoint without source overlays, captured FULL and
PIECEWISE CUDA graphs, returned coherent output, and passed a warmed
concurrency-one decode gate. The NVFP4 TP8 profile is source-qualified but was
not executed against this image.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllmc53cc73-b12xc0a44a1-fi1ac6942-cu133-torch213-20260817-r17` |
| Registry digest | `sha256:c5e96c5bcc5a073f7ce6b56173d88538de3a416900cff97c88b4bf7967fe1dc0` |
| vLLM integration tree | `c53cc73dd64992f013842edf53513f604457c402` |
| B12X integration tree | `c0a44a16f884222faab1ff52ed5db0875bf61971` |
| Runtime receipt | [`infernal-invocation-r17-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r17-remote-gpu.json) |
| Raw CC1 result | [`glm-exl3-tp4-cc1.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r17/glm-exl3-tp4-cc1.json) |
| Source merge contract | [`rtx6kpro` issue #73](https://github.com/local-inference-lab/rtx6kpro/issues/73) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

## Projection-Mixed EXL3 TP4

Download and start the deployment profile:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-exl3-infernal-invocation-r17.yml
docker compose -f docker-compose-glm52-exl3-infernal-invocation-r17.yml up -d
```

The profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`:

| Setting | Deployment default |
|---|---|
| Parallelism | TP4 / DCP1 / MTP3 |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` |
| CUDA graph cap | 32 rows |
| Context limit | 65,536 tokens |
| Routed experts | Checkpoint-native MCG Trellis K3/K4/K5 per projection |
| Dense and shared projections | Online `trellis-mcg-b6` |
| KV cache | NVFP4 DS-MLA |
| Loading | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` |
| Decode graphs | FULL |

The first load encodes eligible BF16 tensors as unpaired MCG Trellis K6 and
stores the derived payload below the `/cache` mount. Reuse the persistent cache
for the same checkpoint revision and quantization contract.

Projection-mixed means that each routed-expert gate, up, and down projection
retains its declared K3, K4, or K5 payload. B12X prepares the checkpoint-native
storage through its BTX representation and adopts the prepared owner without
copying or repacking tensors.

## Qualified Decode Gate

The serving qualification used a bounded scheduler envelope to isolate the
routed-expert and MTP runtime:

```bash
MAX_NUM_SEQS=1 \
GRAPH=4 \
MAX_MODEL_LEN=32768 \
docker compose -f docker-compose-glm52-exl3-infernal-invocation-r17.yml up -d
```

Hardware: four RTX PRO 6000 Blackwell Server Edition GPUs attached to separate
PCIe root ports. The model used 86.61 GiB per rank and completed loading in at
most 95.92 seconds after reusing 444 online K6 cache entries.

| CC1 metric | Result |
|---|---:|
| Aggregate decode | 94.06 tok/s |
| Active-user decode | 94.35 tok/s |
| Target steps | 33.42 steps/s |
| Effective acceptance length | 2.814 |
| Speculative acceptance | 60.48% |
| Request errors | 0 |

PIECEWISE and FULL prefill graphs and a FULL decode graph were captured. The
server returned a coherent chat completion. No traceback, CUDA runtime error,
GPU Xid event, or engine-fatal event was observed.

Infernal Invocation r16 measured 33.33 target steps/s under the same
checkpoint, hardware, and serving configuration. The 0.27% difference is
within run variation. The emitted-token difference follows speculative
acceptance variation and is not evidence of a kernel speedup.

## NVFP4 TP8 Profile

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-nvfp4-infernal-invocation-r17.yml
docker compose -f docker-compose-glm52-nvfp4-infernal-invocation-r17.yml up -d
```

The profile specifies `lukealonso/GLM-5.2-NVFP4`, TP8/DCP1/MTP3, serialized
NVFP4 routed experts, online MXFP8 dense projections, FP8 MLA KV, InstantTensor
`BUFFERED` loading, and FP8 ring transport. Its launch and source contracts
passed release tests. Full-checkpoint startup and throughput remain
**source-qualified**, not runtime-qualified, for r17.

## Required Source Pull Requests

The r17 image composes these open pull requests over the declared repository
bases:

| Responsibility | Pull request |
|---|---|
| Fused unpaired MCG K6 dense projection | [B12X #221](https://github.com/local-inference-lab/b12x/pull/221) |
| Projection-mixed K3/K4/K5 routed experts and BTX ownership | [B12X #223](https://github.com/local-inference-lab/b12x/pull/223) |
| Projection-mixed EXL3 loader, online K6 cache, and graph storage | [vLLM #300](https://github.com/local-inference-lab/vllm/pull/300) |

The reproducible source locks are stored in
[`patches/releases/infernal-invocation-r17`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/patches/releases/infernal-invocation-r17).

## Qualification Limits

- **Qualified:** projection-mixed EXL3 TP4/DCP1/MTP3 with
  `MAX_NUM_SEQS=1`, graph cap 4, and a 32,768-token model-length envelope.
- **Implemented and source-qualified:** the EXL3 deployment defaults with
  `MAX_NUM_SEQS=8`, graph cap 32, and a 65,536-token model-length envelope.
- **Implemented and source-qualified:** NVFP4 TP8/DCP1/MTP3.
- **Unsupported by the EXL3 loader:** SQG checkpoint payloads.
- DCP greater than one, larger scheduler capacities, longer contexts, TP6, and
  GLM-5.2 NVFP4 TP8 require independent r17 serving measurements.
- NVFP4 and EXL3 processes must not share a writable JIT directory.
