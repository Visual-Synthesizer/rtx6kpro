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
| QAD step-1,750 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity, verifier-backed behavior, and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| QAD step-2,500 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step2500`](../kld/glm-5.3-flash-qad-step2500.md); distribution fidelity is measured, and the [9,856-task VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md) qualifies practical equivalence on its primary semantic score; production serving remains unqualified |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified** practical equivalence for QAD step 2,500 versus published NVFP4 on the primary 9,856-task semantic score; **qualified execution with inconclusive one-point decisions** in the [TP8 BF16/published-NVFP4/QAD-step-1,750 report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | **qualified** FP8; packed NVFP4 is implemented but not qualified for R28 |
| MTP proposal vocabulary head | NVFP4 draft-only copy by default; the target verifier vocabulary head remains BF16 |
| GPU prefix cache | **qualified** request/SYSTEM boundaries in all six TP4 mode/DCP combinations; fine aligned retention is selectable |
| Native DRAM offload | **implemented** and opt-in with `CACHE_MODE=native`; not independently requalified for R28 |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; execution-time compute-share fairness assigns 40% of contended model execution to prefill; scheduling interval 1 |
| Root filesystem | Two layers, within standard Docker overlay2 limits |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qwen serving integration | **implemented** in the source; no Qwen model execution or performance qualification for R28 |
| Qualification date | 2026-09-08 |

The [BF16-to-NVFP4 distribution-fidelity report](../kld/glm-5.3-flash-bf16-nvfp4.md),
[QAD step 1,750 comparison](../kld/glm-5.3-flash-qad-step1750.md), and
[QAD step 2,500 progression comparison](../kld/glm-5.3-flash-qad-step2500.md)
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

The [QAD step-2,500 VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
pools 9,856 non-overlapping task pairs from a two-replica TP4 component and an
independent one-replica Max-Q TP4 component. Published NVFP4 scores 91.001%
and QAD step 2,500 scores 91.333%; the +0.332-point difference has a paired
95% interval from -0.317 to +0.982 points, entirely inside the predeclared
±1-point equivalence band. Exact-task accuracy and exploratory family results
remain mixed, so the qualified claim is limited to the primary VBF semantic
score.

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260908-r28
voipmonitor/vllm@sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669
```

The image has two filesystem layers: an immutable CUDA 13.3, PyTorch 2.13,
FlashInfer, NCCL and InstantTensor runtime, followed by the complete vLLM,
B12X, FlashKDA and LMCache installation. It has no dependency on a preceding
community image's writable filesystem. The embedded source lock identifies
complete committed sources, native artifacts and launcher inputs.

The [embedded source lock](glm-5.3-flash/validation/fp8-serving-r28.source.lock)
has SHA-256 `15a9a649559830822cb943ea0c3c6a644c8b69c9e54d7be3e265801851843932`.

### Changes from R27

- Extends exact request and leading SYSTEM/developer checkpoint reuse to
  DFlash2 and DCP4, alongside no-speculation and MTP3. Different user
  continuations can reuse their shared instruction prefix in all six modes.
- LMCache stores those semantic checkpoints as immutable, all-rank bundles.
  Worker-owned asynchronous SHM copies support RAM and filesystem restore,
  including restart of both services, without a sidecar CUDA context.
- Cancellation and eviction cannot recycle checkpoint pages while copies are
  using them. Full RAM pools reclaim eligible payloads; event setup and copy
  submission failures release task ownership correctly.
- Fine aligned retention can keep recurrent state every 256 tokens while
  attention pages remain 2048 tokens. Packed prefill exports avoid a separate
  target forward at every recurrent checkpoint, and speculative convolution
  history remains consistent with the accepted prefix.
- GLM DCP sparse-index compaction preserves selected-index order and initializes
  counts and invalid tails without separate fill kernels.
- Shared-expert output retains its allocation until consumer-stream reads finish.
  Disjoint MLA projection batches avoid the documented SM120/121 cuBLAS
  allocation-boundary fault while retaining CUDA graphs.
- Truncated tool-call arguments remain parseable; malformed tool history does
  not crash template rendering. Aborted external retrieves cannot complete
  twice or free unrelated active requests.
- Qwen source integration is included for independent testing. No Qwen model
  execution or performance qualification is claimed for this release.

## Runtime backends

| Operation | Implementation |
|---|---|
| Target sparse attention and C4 index selection | B12X |
| Target recurrent prefill | FlashKDA with packed checkpoint exports |
| Target recurrent decode | B12X when eligible, with the supported Triton path otherwise |
| Target routed experts | B12X NVFP4 W4A4 |
| Target dense projections | B12X |
| Tensor-parallel all-reduce | B12X PCIe one-shot/two-shot for supported sizes; PyNCCL for the remaining sizes |
| MTP attention / experts | B12X / Marlin |
| MTP vocabulary projection | Private NVFP4 draft copy; target vocabulary projection remains BF16 |
| DFlash2 weights and linear projections | Offline MXFP8 checkpoint and B12X |
| DFlash2 local attention | Graph-safe split-KV FlashAttention |
| Sampling | FlashInfer-compatible probability dispatch and standard rejection |
| External cache | LMCache through worker-owned asynchronous pinned SHM |

The target uses FP8 KV cache. DFlash's local attention uses its compatible
automatic cache dtype; it is not advertised as an MXFP8 KV format.
DeepGEMM and TileLang are installed dependencies but are not selected for the
qualified GLM target, MTP or DFlash2 hot paths. FlashKDA is the prefill default;
`GLM53_KDA_PREFILL_BACKEND=b12x` selects the retained B12X alternative. The
performance table below uses FlashKDA, not that alternative.

## Measured performance

Stock RTX PRO 6000 Blackwell Workstation Edition GPUs, TP4, FP8 target KV,
4096-token scheduler budget, OMP1, NCCL 16 channels/2 MiB buffers, and
`FULL_AND_PIECEWISE` CUDA graphs. Each R27/R28 ratio uses the same physical
quartet: GPUs 0–3 for no-speculation, 4–7 for MTP3, and 8–11 for DFlash2.
Do not compare absolute rates between quartets as an isolated mode speedup.
R27 was rerun for this comparison; its values are not copied from the R27
publication measurements taken under a different test session.

The benchmark uses a fresh 32K context bucket for prefill and context-zero
mathematics generation for C1/C8. Prefill is input tokens divided by API TTFT,
including first-output work, not an isolated GPU prefill timer. C8 is aggregate
throughput across eight concurrent requests. Three 30-second samples follow
warmup; MTP3/DCP4 and DFlash2/DCP1 retain six samples across both boot orders.

| Mode | DCP | 32K prefill, R27 → R28 tok/s | C1 output, R27 → R28 tok/s | C8 total output, R27 → R28 tok/s |
|---|---:|---:|---:|---:|
| No-spec | 1 | 14,773 → 14,692 (−0.55%) | 158.80 → 158.86 (+0.04%) | 704.42 → 701.11 (−0.47%) |
| MTP3 | 1 | 14,428 → 14,378 (−0.35%) | 249.73 → 247.55 (−0.87%) | 885.30 → 892.23 (+0.78%) |
| DFlash2 K7 | 1 | 14,318 → 14,252 (−0.46%) | 208.97 → 206.90 (−0.99%) | 688.13 → 681.52 (−0.96%) |
| No-spec | 4 | 13,180 → 13,531 (+2.66%) | 141.80 → 142.21 (+0.29%) | 647.73 → 639.71 (−1.24%) |
| MTP3 | 4 | 12,890 → 13,195 (+2.37%) | 233.43 → 225.25 (−3.50%) | 819.19 → 829.69 (+1.28%) |
| DFlash2 K7 | 4 | 12,814 → 13,102 (+2.25%) | 187.84 → 196.14 (+4.42%) | 627.93 → 629.43 (+0.24%) |

The MTP3/DCP4 short-duration C1 cell fails the 2% output-loss gate. A separate
24-seed comparison on the exact packaged R28 image completes 4,096 output
tokens per request, retaining all 96 fresh/repeated requests across both images:

| MTP3/DCP4 request | C1 output, R27 → R28 tok/s | Verifier, R27 → R28 steps/s |
|---|---:|---:|
| Fresh prompt | 228.35 → 232.97 (+2.02%) | 91.84 → 92.38 (+0.59%) |
| Repeated prompt | 229.43 → 231.64 (+0.96%) | 91.84 → 92.35 (+0.55%) |

R27 recomputes all 78 repeated prompt tokens; R28 restores all 78 and computes
zero. Within R28, repeated versus fresh output changes −0.57% and verifier
rate −0.04%. All declared median-rate gates pass. The paired checkpoint output
interval is −1.62% to +1.30%; this is descriptive, not a universal equivalence
proof. The larger control does not reproduce a persistent checkpoint loss,
but the failed short-duration result remains visible above.

The [qualification report](glm-5.3-flash/validation/fp8-serving-r28.md) records
artifact boundaries, complete-output summaries and numerical limitations.

| Mode | DCP | C1 verifier, R27 → R28 steps/s | C8 aggregate verifier, R27 → R28 steps/s |
|---|---:|---:|---:|
| MTP3 | 1 | 102.64 → 102.85 (+0.20%) | 365.80 → 365.93 (+0.04%) |
| MTP3 | 4 | 91.71 → 92.20 (+0.53%) | 334.02 → 337.57 (+1.06%) |
| DFlash2 K7 | 1 | 85.54 → 85.29 (−0.29%) | 285.26 → 283.23 (−0.71%) |
| DFlash2 K7 | 4 | 77.99 → 78.16 (+0.22%) | 256.14 → 258.77 (+1.03%) |

Output also depends on proposal acceptance. C8 verifier rates sum request
progress; they do not count physical batched CUDA graph launches. Sub-percent
changes are observations, not individually demonstrated speedups.

Separate DCP1 Sieve tests use temperature 1, top-p 0.95 and 4096 output tokens,
with LMCache enabled. Median cold/RAM output is **312.01/310.34 tok/s for MTP3**
and **386.14/381.20 tok/s for DFlash2**. They retain 24 and 48 distinct seeds
respectively. These are different workloads from the mathematics table and
are not an additional R27/R28 comparison. No +6000 VRAM result is inferred
from these stock-clock measurements.

## Start the server

Use model names and named volumes; source-code mounts are not needed.
The defaults already select full-and-piecewise graphs, the B12X paths,
FlashInfer sampling, NCCL 16 channels/2 MiB and OMP1.

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260908-r28
GPU_DEVICES=0,1,2,3
PORT=8000
docker pull "$IMAGE"
```

Choose one mode. No speculation:

```bash
NAME=jovian-judgement-nospec
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
```

MTP3:

```bash
NAME=jovian-judgement-mtp3
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
```

DFlash2 with seven draft tokens:

```bash
NAME=jovian-judgement-dflash2
MODE_ARGS=(-e SPECULATOR=dflash2 -e DFLASH_DEPTH=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2)
```

Run the common command after assigning the chosen mode's variables:

```bash
docker run -d --name "$NAME" --init \
  --gpus "\"device=${GPU_DEVICES}\"" --network host --ipc host \
  -v jovian-judgement-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
  -e TP=4 -e DCP=1 -e PORT="$PORT" \
  -e MAX_MODEL_LEN=1048576 -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=1 -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.93 \
  "${MODE_ARGS[@]}" "$IMAGE"
```

For DCP4, change `-e DCP=1` to `-e DCP=4`. Full-CKV prefill is selected
automatically; it is not DFlash-only. DCP2 and TP8 are implemented but are not
independently qualified by the TP4/DCP1/DCP4 measurements on this page.

Model names resolve Hugging Face `main` at startup. Optional `MODEL_REVISION`
and `DFLASH_MODEL_REVISION` variables provide reproducible model selection.
The runtime authenticates the resolved revisions before external checkpoint
reuse; changing a model or source identity produces safe cache misses.

### Scheduler and cache geometry

`PREFILL_COMPUTE_SHARE=0.4` targets 40% of measured execution time for prefill
only while prefill and decode contend. `PREFILL_SCHEDULE_INTERVAL=1` is required.
Set `FAIRNESS_ENGINE=none` to disable this policy. MTP uses a private NVFP4
proposal vocabulary head by default, costing 85.08 MiB per TP4 rank. The target
vocabulary head remains BF16; `VLLM_GLM53_MTP_DRAFT_HEAD=bf16` selects the
unquantized draft head for independent comparison.

The launcher owns page geometry. GPU-local serving keeps 2048-token attention
pages; fine recurrent retention is selected separately below. LMCache derives
per-rank pages from 4096-token storage objects and the DCP width. The public
vLLM attention block argument remains 256. Weighted allocation groups layers
by their actual cache cost, with a bounded number of groups, to reduce padding.
Normal deployments do not need to override these layout settings.

### Recurrent checkpoint policy

The default `auto` policy selects request-boundary retention for the qualified
GLM no-spec/MTP3/DFlash2 configurations, including DCP4. It preserves exact
request endpoints and leading SYSTEM/developer instruction endpoints. It does
not promise a hit at every arbitrary byte or token inside a changed user turn.

For arbitrary shared token prefixes, GPU-local aligned retention is selectable.
Add the following environment setting before the image name and arguments
after the image name:

```bash
# Before "$IMAGE":
-e GLM53_MAMBA_BLOCK_SIZE=256

# After "$IMAGE":
--recurrent-checkpoint-policy aligned --prefix-match-unit 256
```

Attention pages remain 2048 tokens. Packed exports retain interior recurrent
states without forcing a target forward per checkpoint. This trades additional
retained recurrent state for finer prefix reuse. Request-boundary retention
stores fewer states for ordinary shared-instruction and turn-boundary reuse.
The fine-aligned MTP3/DCP4 interval comparison measured −0.20% prefill,
+0.32% C1 verifier and +0.35% C8 verifier rate; that comparison precedes the
disjoint-MLA projection change. It is not a fresh speed claim for every mode.

## LMCache RAM and filesystem storage

LMCache is opt-in. In the common command, replace `-e CACHE_MODE=vram` with:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_TRANSFER_MODE=engine_driven \
-e LMCACHE_L1_SIZE_GB=64 \
-e LMCACHE_L2_ENABLED=1 \
-v jovian-judgement-lmcache-l2:/lmcache-l2
```

The host shared-memory filesystem must have at least 96 GiB available for the
default 64 GiB RAM pool and transfer buffers. With `--ipc host`, the host's
`/dev/shm` capacity applies; `--shm-size` does not enlarge it. The sidecar runs
inside the container and has no CUDA context. Existing vLLM workers perform
GPU gather/scatter through asynchronous pinned SHM. Do not manually remove a
shared-memory pool while either service is using it.

The launcher chooses geometry compatible with `LMCACHE_CHUNK_SIZE=4096` and
`LMCACHE_TARGET_TOKEN_BUDGET=4096`. Semantic target, recurrent and draft state
are published only as complete all-rank bundles. Payload locks protect active
copies, and incompatible source/model/layout identities safely miss. RAM
pressure evicts unlocked payloads; filesystem storage remains reusable after
both services restart. Incomplete or incompatible semantic generations are
recomputed, never partially imported. Version-1 semantic payload files do not
match the version-2 storage keys used here.

Set `LMCACHE_L2_ENABLED=0` for RAM-only operation. If multiple instances share
the host network, give each distinct API and LMCache HTTP/MP/metrics ports.
Do not share a writable cache directory across independent sidecars.

All six FP8 mode/DCP combinations pass million-token cold/RAM/restart checks:

| Mode | DCP | 1M cold, seconds | RAM restore, seconds | Restore after service restart, seconds |
|---|---:|---:|---:|---:|
| No-spec | 1 | 93.899 | 0.918 | 1.768 |
| No-spec | 4 | 93.905 | 0.694 | 1.147 |
| MTP3 | 1 | 97.102 | 0.900 | 3.157 |
| MTP3 | 4 | 97.739 | 0.739 | 0.977 |
| DFlash2 | 1 | 95.957 | 0.996 | 1.618 |
| DFlash2 | 4 | 96.374 | 0.722 | 0.987 |

Each restore attributes all one million prompt tokens to external storage and
zero to local compute. Times include API/first-output work. The OS page cache
was not flushed; restart results are not cold-device storage benchmarks.
The exact packaged image also passes 54K lookup answers across all cache tiers,
shared-SYSTEM reuse, C4 all-rank bytes and C8 cancellation/live-read eviction.
These tests qualify storage correctness, not universal bitwise generation
equivalence across different floating-point prefill partitions.

Native DRAM offload remains implemented through `CACHE_MODE=native`; it is not
independently requalified for R28. The qualified external path above is LMCache.
Packed NVFP4 target KV and Qwen execution are outside this release's qualification.

For independent native-offload testing, use `-e CACHE_MODE=native` and
`-e NATIVE_KV_OFFLOADING_SIZE_GB=64` in the common command. The launcher selects
the required shareable allocator. These settings do not enable LMCache.

## Source and review contract

The [portable source-locked build recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
includes the native FlashKDA build, source-bundle verification and CPU build
tests. It lists the exact source revisions; no chain of preceding community
images is needed. Runtime ABI dependencies are supplied by its pinned base.

Complete Git mirrors preserve authorship and integration resolutions:
[vLLM](https://github.com/voipmonitor/vllm/tree/integration/glm-fp8-checkpoint-serving-20260908),
[B12X](https://github.com/voipmonitor/b12x/tree/integration/glm-fp8-checkpoint-serving-20260908),
[LMCache](https://github.com/local-inference-lab/LMCache/tree/integration/glm-fp8-checkpoint-serving-20260908).
The [open merge checklist](https://github.com/local-inference-lab/vllm/issues/651)
describes each PR and integration caveat. Source locks, not tag-name inference,
identify the measured packages. The timing matrix and exact packaged storage
checks have their source boundaries recorded in the validation evidence.

Git histories retain the original contributions and author attribution,
including Luke Alonso, MadeBy561, Giancarlo Delfin, Derek Yates, Thien Tran,
logprobz, Apple FCU Fleet, Martin Vit, Codex and the other recorded authors.
