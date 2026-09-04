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
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
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
voipmonitor/vllm:jovian-judgement-community-20260904-r24
voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd
```

The embedded source-lock SHA-256 is
`5ef0481267b0e672c13ffb2c8ffe914d927a02eb652f5112e1c0fbbea6e84d49`.
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
| External cache | LMCache DRAM L1 and native-filesystem L2 through asynchronous engine-driven pinned shared memory when selected |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths.

## Qualified performance

The measurements used physical GPUs 4 through 7 on one RTX PRO 6000 Blackwell
Workstation Edition host at stock clocks, PCIe Gen5 x16 links, tensor
parallelism of four, FP8 target KV cache, a 4,096-token scheduler budget, full
and piecewise decode graphs, exactly 16 NCCL channels, and a 2 MiB NCCL buffer.
The measured vLLM tree, B12X tree, and model launcher are byte-identical to the
R24 artifact. Each prefill value is a 30-second 32K-token measurement after a
30-second warmup. C1 and C8 are 30-second sustained context-zero decode cells.
Other serving workloads were active on the host, so the table gives
conservative rather than isolated peak throughput.

| DCP | Serving mode | 32K prefill tok/s | C1 output tok/s | C1 steps/s | C1 accepted/step | C8 output tok/s | C8 steps/s | C8 accepted/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | No speculation | 14,911 | 169.9 | — | — | 737.8 | — | — |
| 1 | MTP, depth 3 | 14,485 | 247.8 | 99.0 | 2.50 | 903.2 | 364.7 | 2.48 |
| 1 | DFlash2, depth 7 | 14,648 | 221.1 | 89.8 | 2.46 | 689.6 | 289.4 | 2.38 |
| 4 | No speculation | 13,368 | 151.8 | — | — | 667.5 | — | — |
| 4 | MTP, depth 3 | 12,912 | 227.0 | 88.8 | 2.56 | 830.9 | 331.2 | 2.51 |
| 4 | DFlash2, depth 7 | 13,181 | 206.7 | 81.8 | 2.53 | 630.8 | 267.7 | 2.36 |

Speculative output throughput varies with accepted length. Target steps per
second isolates target-model execution speed.

The published two-layer R24 image additionally completed a packaging smoke on
the same four stock-clock GPUs. DCP4 DFlash2 with packed-NVFP4 cache and the
2,048-token GPU page measured 12,769 prompt tok/s and 80.6 verifier steps/s;
the observed 192.1 output tok/s corresponds to an accepted length of 2.38.

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
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260904-r24
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
-e LMCACHE_L2_ENABLED=1
-e LMCACHE_L2_ROOT=/lmcache-l2
-e GPU_MEMORY_UTILIZATION=0.95
```

`KV_CACHE_QUANT=nvfp4_ds_mla` selects the qualified packed-NVFP4 target cache
instead. The filesystem namespace includes the target and draft revisions,
cache format, parallelism, DCP gathering policy, speculation mode, and object
size, preventing incompatible cache objects from being reused.

Qualification covered cold compute, vLLM automatic prefix reuse, LMCache DRAM
restore, and filesystem restore for FP8 and NVFP4. External bytes were compared
on every tensor-parallel rank across target attention, recurrent state, and
DFlash sliding attention. A one-million-token request restored 999,424 tokens
and recomputed the 576-token suffix in 1.14 to 1.54 seconds, depending on
serving mode and cache format. Full-container restart tests also restored the
expected chunk-aligned prefix from filesystem storage.

The matched TP4/DCP4 performance test used packed-NVFP4 KV cache, 1,024-token
per-rank LMCache pages, stock clocks, and the same vLLM and B12X package trees
on both arms. Decode comparisons use target steps per second for speculative
modes so stochastic acceptance does not masquerade as an execution change.

| Serving mode | GPU-only 32K prefill tok/s | LMCache 32K prefill tok/s | Prefill change | GPU-only C1 | LMCache C1 | C1 change |
|---|---:|---:|---:|---:|---:|---:|
| No speculation | 12,647 | 12,569 | -0.62% | 150.94 tok/s | 150.79 tok/s | -0.10% |
| MTP, depth 3 | 12,303 | 12,185 | -0.96% | 93.3 steps/s | 93.2 steps/s | -0.11% |
| DFlash2, depth 7 | 12,476 | 12,397 | -0.63% | 80.88 steps/s | 81.45 steps/s | +0.70% |

The measured cold-prefill overhead remains below 1%. DFlash2 C1 output was
202.03 tok/s without LMCache and 204.11 tok/s with LMCache; the +1.03%
difference is consistent with its small accepted-length variation rather than
a cache execution cost.

## Source and review contract

| Component | Qualified source |
|---|---|
| vLLM | [R24 integration source](https://github.com/local-inference-lab/vllm/tree/integration/glm53-r23-lmcache-parser-20260904); commit `d49385468458cf97dff0fc8d9c8863f8082abf4f`; tree `e2c687bb823dbe1b37c3d9f9742a0ae54419fdb0`; package tree `17acb470467c1a6d4b318a3c4a0960794fb4da6a` |
| FlashKDA | commit `3b225bf26bb8e218928a1fe14751cb48cf31d11b`; extension SHA-256 `16aece5ffb83c2dfb0355758bbbc9d6e0ea50a2cfc36ecee4936607d445aba0a` |
| B12X | [R24 integration source](https://github.com/local-inference-lab/b12x/tree/integration/glm53-r23-lmcache-parser-20260904); commit `e3d0ae067f607538e3709ac3c30c7042276c6f88`; tree `d93cd222b027ed1df7f7df221007196994c80354`; package tree `fc977aa2b732935cd0f70c365d7f767b449d21da` |
| LMCache | [PR 43](https://github.com/local-inference-lab/LMCache/pull/43) preserves complete DFlash pages; [PR 45](https://github.com/local-inference-lab/LMCache/pull/45) adds stride-correct asynchronous engine-driven hybrid stores; commit `415c5d60bd7b57e85f20c34a6f5a3e51f6018136`; tree `311de681786928048d9975db07bc81c70141668d`; package tree `6685246954c3e83d95dc1c1deff2ec82b5d430cc` |

The [vLLM merge checklist](https://github.com/local-inference-lab/vllm/issues/590)
lists each open pull request, dependency, resulting behavior, attribution, and
qualification result. The image embeds the same source contract at
`/opt/glm53-flash/source.lock`.
