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
| QAD step-2,500 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step2500`](../kld/glm-5.3-flash-qad-step2500.md); distribution fidelity is measured, and the [matched TP4×2 VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md) records qualified executions with inconclusive behavioral comparisons; production serving remains unqualified |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified executions; inconclusive one-point decisions** in the [TP8 BF16/published-NVFP4/QAD-step-1,750 report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) and the [TP4×2 published-NVFP4/QAD-step-1,750/QAD-step-2,500 report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | FP8 by default; packed NVFP4 is selectable |
| MTP proposal vocabulary head | NVFP4 draft-only copy by default; the target verifier vocabulary head remains BF16 |
| GPU prefix cache | **qualified** with independently sized target and recurrent allocations |
| Native DRAM offload | **qualified** and opt-in with `CACHE_MODE=native` |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; execution-time compute-share fairness assigns 40% of contended model execution to prefill; scheduling interval 1 |
| Root filesystem | Two layers, within standard Docker overlay2 limits |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qualification date | 2026-09-05 |

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
uses a matched two-replica TP4 profile for published NVFP4, QAD step 1,750, and
QAD step 2,500. Their semantic scores are 92.31%, 91.87%, and 89.72%
respectively. QAD step 2,500 has the lowest point estimate, but its paired
intervals against both comparators are inconclusive. The controls were
commissioned after the step-2,500 result was inspected, so behavioral claims
remain research-only.

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260905-r26
voipmonitor/vllm@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68
```

The embedded source-lock SHA-256 is
`bced2847d40e650a145d7060d614bcf77de7df5d8a1d2db1a64c7bcaa338217a`.
The Docker digest fixes the runtime. Model repository names follow Hugging Face
`main` unless an optional revision variable is supplied.

The image installs source-locked vLLM, B12X, FlashKDA, and LMCache package
trees over one flattened CUDA 13.3, PyTorch 2.13, FlashInfer, NCCL 2.31.2, and
InstantTensor runtime layer. The second layer contains every model-serving
package, native LMCache extension, launcher, and source lock.

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
| External cache | LMCache DRAM L1 and native-filesystem L2 through asynchronous engine-driven pinned shared memory when selected |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths.

## Qualified performance

The measurements used physical GPUs 4 through 7 on one RTX PRO 6000 Blackwell
Workstation Edition host at stock clocks, PCIe Gen5 x16 links, tensor
parallelism of four, FP8 target KV cache, a 4,096-token scheduler budget, full
and piecewise decode graphs, exactly 16 NCCL channels, and a 2 MiB NCCL buffer.
Each 32K prefill cell measures client time to first token after a warmup request.
C1 and C8 are sustained context-zero decode cells. The complete matrix used
30-second cells; the published-manifest regression gate below used 60-second C1
cells after explicit warmup.

| DCP | Serving mode | 32K prefill tok/s | C1 output tok/s | C1 steps/s | C1 accepted/step | C8 output tok/s | C8 steps/s | C8 accepted/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | No speculation | 14,917 | 169.9 | — | — | 734.6 | — | — |
| 1 | MTP, depth 3, NVFP4 proposal head | 14,584 | 258.4 | 109.31 | 2.36 | 899.3 | 373.84 | 2.41 |
| 1 | DFlash2, depth 7 | 14,705 | 213.5 | 90.54 | 2.36 | 692.1 | 292.73 | 2.36 |
| 4 | No speculation | 13,355 | 151.9 | — | — | 667.8 | — | — |
| 4 | MTP, depth 3, NVFP4 proposal head | 13,003 | 249.0 | 97.57 | 2.55 | 844.5 | 344.89 | 2.45 |
| 4 | DFlash2, depth 7 | 13,210 | 190.9 | 81.86 | 2.33 | 636.9* | 259.94* | 2.45* |

Speculative output throughput varies with accepted length. Target steps per
second isolates target-model execution speed. The starred DFlash2 DCP4 C8 cell
served eight simultaneous requests but the benchmark marked it
capacity-limited; the value is reported rather than treated as a saturation
ceiling.

The published manifest was rebuilt from the final source mirrors after the
complete matrix. The changes after the matrix were an exact-shape safety guard
for the already selected DFlash fused kernel, test-fixture initialization, and
an RTX PRO 6000 Max-Q profile correction; none changes the measured workstation
hot paths. The final two-layer manifest passed these independent regression
gates at stock clocks:

| Mode | DCP | 32K prefill | C1 output | C1 target execution | Emitted tokens per step |
|---|---:|---:|---:|---:|---:|
| No speculation | 1 | 14,937 tok/s | 170.1 tok/s | 170.1 steps/s | 1.00 |
| MTP depth 3 | 1 | 14,509 tok/s | 273.6 tok/s | 109.0 steps/s | 2.51 |
| DFlash2 depth 7, packed-NVFP4 KV | 4 | 12,783 tok/s | 199.2 tok/s | 81.3 steps/s | 2.45 |

The packaged vLLM, B12X, and LMCache trees also passed 383 focused tests; 33
unsupported-device cases were skipped.

The NVFP4 MTP proposal head uses 85.08 MiB per tensor-parallel rank. A matched
BF16 proposal-head run measured approximately 99.5 verifier steps/s; the image
default measured 109.31 steps/s, approximately 9.9% higher. The target model,
target vocabulary head, and verifier remain BF16, so standard rejection
preserves the target distribution.

### Complete-KV prefill

With four decode-context ranks, every rank must select sparse-attention
candidates from the complete target KV sequence rather than its rank-local
quarter. The qualified implementation gathers the complete target KV view and
keeps recurrent-state cache ownership independent. A matched feature-isolation
test measured 12,999 prompt tok/s with complete-KV selection versus 9,858
prompt tok/s with rank-local selection, a 31.86% increase.

### Memory-weighted cache allocation

GLM-5.3 target attention and recurrent state have different bytes-per-request
costs. The runtime partitions compatible layers by their actual cost instead
of forcing equal layer counts into each shared pool. The deterministic search
is capped at eight cache groups, preserving bounded scheduler and connector
overhead.

For TP4/DCP4 without speculation, the weighted allocation admits 21,929,984
tokens instead of 20,873,216 under equal-count grouping: 1,056,768 additional
tokens, or 5.06%. A matched DCP1 DFlash2 A/B found no attributable execution
regression: 32K prefill was 15,151 versus 15,138 tok/s, C1 verifier throughput
was 101.6 versus 103.3 steps/s, and C8 verifier throughput was 332.1 versus
327.3 steps/s.

### Compute-share fairness

The scheduler measures model-execution time separately for prefill and decode
and targets 40% prefill share only while both classes are contending. A mixed
C8 DFlash2 decode and cold 32K-prefill run assigned 42.97% of measured
contended execution time to prefill. Forward passes are indivisible, so short
runs oscillate around the configured target.

Set `FAIRNESS_ENGINE=none` to disable fairness. Compute-share fairness requires
`PREFILL_SCHEDULE_INTERVAL=1`; the launcher rejects incompatible values rather
than silently changing scheduling behavior.

### FlashKDA numerical stability

The deterministic 16,384-token near-collinear BF16-key reproducer produces
zero non-finite output or recurrent-state elements with the packaged FlashKDA
extension. The extension replaced by this artifact produced 55,264 non-finite
output elements and 16,000 non-finite recurrent-state elements for identical
inputs. The qualified server also completed 24 long concurrent TP4/DCP4 MTP3
requests without a wrong or runaway result.

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
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260905-r26
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

LMCache uses a CPU-only sidecar process in the same container. GPU gather and
scatter run in the existing vLLM workers; the sidecar receives an empty
`CUDA_VISIBLE_DEVICES` and creates no additional CUDA context. Asynchronous
engine-driven transfer through pinned shared memory is selected automatically.
The cache stores complete 4,096-token objects in DRAM and optionally in a
mounted filesystem. Use a private shared-memory allocation of at least 96 GiB;
128 GiB is the qualified setting.

Replace the cache settings, shared-memory size, and GPU-memory-utilization line
in the common command with:

```bash
--shm-size 128g
-v jovian-judgement-lmcache-l2:/lmcache-l2
-e CACHE_MODE=lmcache
-e KV_CACHE_QUANT=fp8_ds_mla
-e LMCACHE_CHUNK_SIZE=4096
-e LMCACHE_TARGET_TOKEN_BUDGET=4096
-e LMCACHE_L1_SIZE_GB=64
-e LMCACHE_SHM_NAME=jovian-judgement-cache
-e LMCACHE_L2_ENABLED=1
-e LMCACHE_L2_ROOT=/lmcache-l2
-e GPU_MEMORY_UTILIZATION=0.93
```

`KV_CACHE_QUANT=nvfp4_ds_mla` selects the qualified packed-NVFP4 target cache
instead. The filesystem namespace includes the target and draft revisions,
cache format, parallelism, DCP gathering policy, speculation mode, and object
size, preventing incompatible cache objects from being reused.

Qualification covered cold compute, vLLM automatic prefix reuse, LMCache DRAM
restore, and filesystem restore for FP8 and NVFP4 cache formats. External bytes
were compared on every tensor-parallel rank across target attention, recurrent
state, and DFlash sliding attention.

For DFlash2 with packed-NVFP4 KV cache, a one-million-token cold request took
95.244 seconds with LMCache enabled and 93.717 seconds with LMCache disabled,
an overhead of 1.527 seconds or 1.63%. RAM L1 restored 999,424 tokens and
recomputed 576 in 1.150 seconds at 42.778 GiB/s. After both vLLM and LMCache
were restarted, native-filesystem L2 restored the same prefix in 1.261 seconds
at 39.023 GiB/s. Both restores produced the same greedy output as cold compute.

Four concurrent 36,036-token requests also passed a transfer-boundary check.
Every byte matched for four tensor-parallel ranks and nine hybrid cache groups;
source and destination GPU blocks were disjoint, and all 131,072 complete
prompt tokens were attributed to external KV transfer.

The engine-driven store path creates its immutable paged-cache pointer table
once at worker registration and reuses exclusive pinned-host and CUDA block-ID
staging slots. The CPU-only sidecar therefore receives completed shared-memory
payloads without pickle serialization. On restart, LMCache removes only the
configured stale named pool before measuring tmpfs capacity; a pool still
mapped by another process remains charged and cannot be counted as free space.

The matched TP4/DCP4 DFlash2 execution checks used packed-NVFP4 KV cache. The
LMCache geometry uses a 1,024-token per-rank target page so a 4,096-token global
object divides exactly across four DCP ranks.

| Measurement | GPU-only | LMCache | Change |
|---|---:|---:|---:|
| 32K prefill | 13,210 tok/s | 12,428 tok/s | -5.92% |
| C1 target execution | 81.86 steps/s | 81.6 steps/s | -0.32% |
| C8 target execution | 259.94 steps/s | 263.3 steps/s | +1.29% |

The shorter 32K prefill cell exposes the smaller LMCache-aligned page geometry;
the one-million-token cold comparison above isolates asynchronous store
overhead under identical geometry.

## Build from source

The [build directory](glm-5.3-flash/build/) contains the Dockerfile, launchers,
LMCache runtime requirements, and
[source lock](glm-5.3-flash/build/glm53-jovian-judgement-community-20260905-r26.source.lock)
used for the published image. The source lock identifies complete integration
trees rather than a set of uncommitted overlays.

From the root of a clone of this repository, check out the three locked source
trees and pass them to Docker BuildKit as named build contexts:

```bash
mkdir -p build-sources

git clone https://github.com/voipmonitor/vllm.git build-sources/vllm
git -C build-sources/vllm checkout 7f53b30481b4110f293521d09f7af18f9e8f9d9e

git clone https://github.com/voipmonitor/b12x.git build-sources/b12x
git -C build-sources/b12x checkout 60dbc57098a3abff60cbf6048bcf8784c117feaf

git clone https://github.com/local-inference-lab/LMCache.git build-sources/LMCache
git -C build-sources/LMCache checkout 63919a2c6c310f9b34de7049b9b28b77fab13ca0

DOCKER_BUILDKIT=1 docker build \
  --build-context vllm_source=build-sources/vllm \
  --build-context b12x_source=build-sources/b12x \
  --build-context lmcache_source=build-sources/LMCache \
  --file models/glm-5.3-flash/build/Dockerfile \
  --tag local/glm-5.3-flash-jovian-judgement:r26 \
  models/glm-5.3-flash/build
```

The build fails if a source package tree, launcher, FlashKDA extension, or
source-lock hash differs from the qualified contract. The resulting runtime
has two root-filesystem layers: one flattened CUDA/PyTorch foundation and one
source-locked serving layer.

## Source and review contract

| Component | Qualified source |
|---|---|
| vLLM | [Source-locked GLM integration](https://github.com/voipmonitor/vllm/tree/integration/glm53-r26-current-head-20260905); commit `7f53b30481b4110f293521d09f7af18f9e8f9d9e`; tree `c861b31da5cf527d96b5772a6709450b23c93a6f`; package tree `3d98f6ebfaabf06309c9a5d3a7fbd469747aa5e1` |
| FlashKDA | commit `3b225bf26bb8e218928a1fe14751cb48cf31d11b`; extension SHA-256 `16aece5ffb83c2dfb0355758bbbc9d6e0ea50a2cfc36ecee4936607d445aba0a` |
| B12X | [Source-locked GLM integration](https://github.com/voipmonitor/b12x/tree/integration/glm53-r26-current-head-20260905); commit `60dbc57098a3abff60cbf6048bcf8784c117feaf`; tree `c20b6aab67ed791cc226ee91de23f7c45509f436`; package tree `00248b09689830e55d8fbce8ee630600b63ef663` |
| LMCache | [PR 49](https://github.com/local-inference-lab/LMCache/pull/49) preserves bounded native-filesystem keys, [PR 50](https://github.com/local-inference-lab/LMCache/pull/50) reuses registered paged-transfer metadata, and [PR 51](https://github.com/local-inference-lab/LMCache/pull/51) makes named-SHM restart capacity authoritative. PR 51 head `63919a2c6c310f9b34de7049b9b28b77fab13ca0` has the qualified tree `008ac3e09ae5917aa0849147480d7bd5b9f8b37a` and package tree `fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca`. |

The image embeds the complete source contract at
`/opt/glm53-flash/source.lock`. Git commit attribution is preserved for Luke
Alonso, MadeBy561, Thien Tran, logprobz, Apple FCU Fleet, Martin Vit, and the
authors recorded by the source branch histories.

The human-readable merge order and per-change purpose are tracked in
[vLLM issue 651](https://github.com/local-inference-lab/vllm/issues/651).
