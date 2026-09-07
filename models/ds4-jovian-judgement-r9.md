# DeepSeek-V4-Flash Jovian Judgement r9

This serving specification covers DeepSeek-V4-Flash text and Vision inference
on two 96 GiB NVIDIA RTX PRO 6000 Blackwell GPUs. The runtime combines B12X
sparse attention and routed experts, DeepGEMM FP8 dense projections, fixed
probabilistic DSpark, FP8 compressed KV storage, and InstantTensor loading.

**Status: qualified for the TP2 checks listed below.** The delayed r8
production crash reports do not have a locally reproduced cause. Operator
corrections described below must not be interpreted as proof that those
production failures are resolved.

## Start The Prebuilt Image

Text checkpoint, GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r9.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r9.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r9.yml up -d
```

Vision checkpoint:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r9.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r9.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r9.yml up -d
```

Choose one profile for a GPU pair. The Compose files download a prebuilt
image; they contain no `build` section. `GPUS=0,1` and `PORT=8000` are defaults.

GPU KV caching is enabled. Host-memory reuse through LMCache is opt-in:

```bash
LMCACHE_MODE=ram LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r9.yml up -d
```

The LMCache process owns a CPU shared-memory pool. vLLM workers execute the
GPU transfers; the standalone cache process does not create a CUDA context.
The 24 GiB setting is host RAM, not a GPU reservation. Native vLLM filesystem
KV offload is unsupported in these serving profiles.

## Artifact Identity

```text
voipmonitor/vllm:jovian-judgement-vllmf66599d-b12x15b6813-fi803c466-cu133-torch213-20260907-r9
```

The source review checklist is [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95).
The [r8-to-r9 changelog](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r9/CHANGELOG.md)
separates release changes from the serving contract on this page.
OCI labels and the committed composition locks identify every source base,
pull-request head, integration tree, and patch digest. vLLM and B12X are
composed from their canonical branches plus the listed PRs, not runtime
source mounts.

Runtime dependencies: CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, FlashInfer
0.6.18+cu133, CUTLASS DSL 4.6.2, XGrammar 0.2.5, and InstantTensor 0.1.9.

Image ID: `sha256:b0b8af509dc8c3990a10c738ea0a2240a22e52af41af6c272e3945512b236139`.
Registry digest: `sha256:5bea088597980b299a1df8a6f3fc6d2d22c723276088ea8583b456f27043c0cd`.
Build recipe commit: `fe3253b07000387d543cbe6d39773aab6e144949`.

## Serving Profiles

| Setting | Text | Vision |
|---|---|---|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| Revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | `6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| Parallelism | TP2 / DCP1 | TP2 / DCP1 |
| DSpark | Fixed probabilistic K5 | Fixed probabilistic K3 |
| Backend | `b12x-a8-dglin` | `b12x-a8-dglin` |
| Loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |
| `MAX_NUM_SEQS` | 8 | 4 |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | 4096 |
| `MAX_MODEL_LEN` | 1048576 | 1048576 |
| CUDA graph cap | 48 | 16 |
| GPU memory utilization, GPU KV only | 0.975 | 0.975 |
| GPU memory utilization, engine-driven LMCache | 0.970 | 0.970 |

Target decode and DSpark use FULL CUDA graphs. Prefill uses PIECEWISE graphs;
the combined mode is `FULL_AND_PIECEWISE`. Vision contains three draft layers,
so speculative depth greater than three is unsupported.

## Measured Results

Hardware: two 96 GiB RTX PRO 6000 Blackwell Workstation GPUs, 600 W limits,
GPU IDs 0 and 1, behind the same PCIe switch. These measurements do not qualify
direct-root PCIe 4 systems. Both images use the serving profiles above.
The control image is [Jovian Judgement r8](ds4-jovian-judgement-r8.md), with
vLLM tree `d267ca7` and B12X tree `e58515a`.

| Measurement | r8 reference | r9 | Conditions |
|---|---:|---:|---|
| Text K5 CC1 decode, tok/s | 197.42 | 207.45 | Temperature 1; one 30-second sustained window |
| Text K5 target steps/s | 72.90 | 75.71 | Same decode window |
| Text K5 strict draft acceptance | 34.17% | 34.80% | Accepted draft tokens / proposed draft tokens |
| Vision K3 CC1 decode, tok/s | 170.18 | 174.58 | Temperature 1; one 30-second sustained window |
| Vision K3 target steps/s | 81.96 | 82.27 | Same decode window |
| Vision K3 strict draft acceptance | 35.88% | 37.40% | Accepted draft tokens / proposed draft tokens |
| Text K5 cold prefill, tok/s | 13,836 | 14,093 | Four samples; 32k target, 32,100/32,101 actual prompt tokens |
| Text K5 GPU KV pool, tokens | 1,185,524 | 1,186,087 | Utilization 0.975; MNS8; MNB4096; graph cap 48 |
| Vision K3 GPU KV pool, tokens | 1,206,495 | 1,208,466 | Utilization 0.975; MNS4; MNB4096; graph cap 16 |
| Text K5 GPU KV pool with engine-driven LMCache, tokens | 1,113,883 | 1,114,446 | Utilization 0.970; 24 GiB host pool; otherwise text profile |

These are bounded regression checks with llm-inference-bench 0.6.2, not
repeated statistical performance trials. Sampling and DSpark acceptance vary;
the token-rate difference is not an isolated kernel speedup. The benchmark's
generic KV-capacity estimator does not describe DS4's hybrid cache. Pool sizes
above come from vLLM startup logs. A GPU KV pool is shared across requests;
the per-request context limit remains 1,048,576 tokens.

| Correctness check | Result | Evidence boundary |
|---|---|---|
| Unused KV pages containing NaN | r8 fails 8/8 cases; r9 passes | Independently reproduced operator defect |
| Compressed attention suite | r9 passes 20/20 | Decode, prefill, graph replay, and wide page offsets |
| Worker accounting and auxiliary lifetimes | r9 passes 20/20 | Fourteen accounting cases and six stream cases |
| LMCache asynchronous-copy lifetime suite | r9 passes 9/9 | Includes exceptions after partial copy submission |
| Vision exact prefix resume | 6/6 continuations pass | Prompt 270,731; cached 267,008; output 4,096 tokens |
| Vision variable prefix resumes | 96/96 continuations pass | 24 waves of four; appended lengths 257-8,192; exact cache-hit checks |
| Vision 810k prefill plus image request | Both pass | Ten 2048px images submitted during prefill; at least 1,025 MiB/rank free |
| Text K5 810k prefill, GPU KV only | Pass, 122.20 s | Exact 810,000 prompt tokens; at least 903 MiB/rank free |
| Text K5 810k prefill with engine-driven LMCache | Pass, 121.19 s | At least 1,201 MiB/rank free; host stores enabled |
| Text K5 LMCache restore after GPU eviction pressure | Pass | Zero local prefix hits; 4,095 tokens actually supplied by external KV transfer |

Synthetic prefix-resume checks exercise scheduler shapes and allocation
lifetimes. They do not measure answer quality or reproduce a multi-hour
conversation history. The Vision reference also passes the exact and variable
resume workloads, so these results cannot establish a production crash fix.
Long-prefill wall times include shape compilation when encountered and are
memory-admission evidence, not an isolated comparison of cache-transfer cost.
The [public validation directory](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/validation/jovian-judgement-ds4-r9)
contains receipts and reproduction commands.

The host-tier check stores an 810k prefix, submits an independent 400k
prefill to exceed the GPU pool, then replays a stored 4,096-token request.
Both ranks retrieve a complete block from LMCache; vLLM recomputes the final
token. The response completes in 1.60 s including first-use indexer compilation.
Each rank logs 0.006 s for retrieval itself; these are different timing spans.
RAM-tier functionality is qualified for text K5. Vision host restoration and
filesystem persistence were not retested in this image.

## Correctness Contracts

### Masked KV Rows

An attention candidate excluded by its length or index must contribute zero
regardless of the bytes in an unused cache page. A zero probability alone
does not enforce this when the value contains NaN, because zero multiplied
by NaN is still NaN. B12X clears masked shared-memory value rows before MMA.
Its prefill dispatcher also preserves the requested computation mode so that
compressed positional components are read from the correct cache.

These changes are in B12X commit
[`cfce3d64`](https://github.com/local-inference-lab/b12x/commit/cfce3d64587dd4cff2e35acbed21dde29f98c701).
The operator tests cover 16/32 query heads, decode/prefill, CUDA graph replay,
distinct sliding-window/indexed cache contents, and live pages beyond a 2 GiB offset.

### Auxiliary Output Lifetimes

A CUDA event join orders producer and consumer operations but does not, by
itself, extend the allocator lifetime of a tensor returned from another
stream. [vLLM #695](https://github.com/local-inference-lab/vllm/pull/695)
registers returned tensor allocations with the consuming stream outside
CUDA graph capture. It handles nested lists, tuples, and dictionaries without
adding a device synchronization or disabling overlap.

The reported attention branch with `compress_ratio=128` returns no tensor
from its auxiliary compressor callback. This correction therefore does not
establish the cause of the corresponding production crash.

### Memory Accounting

Admission profiles a complete scheduler-token prefill with actual attention
and every hybrid KV group. Vision encoder outputs remain resident during the
architecture-specific profile. The measured peak is reserved before GPU KV
allocation; no fixed additional reserve is introduced.

[vLLM #694](https://github.com/local-inference-lab/vllm/pull/694) keeps estimated
graph memory separate from transient activation memory. The recommendation
emitted after graph capture counts measured graph allocations exactly once.
Initial KV admission is unchanged. This is an accounting correction, not a
demonstrated fix for delayed physical-memory exhaustion.

### Engine-Driven Transfers

[LMCache #55](https://github.com/local-inference-lab/LMCache/pull/55) holds
locally pinned host buffers until queued transfers complete, including when
a transfer raises after partial submission. Caller-owned pinned buffers and
operations that enqueue no copy retain their existing behavior.

## Image Prefix Caching

GPU KV reuse requires a complete reusable block. The block size is 256 tokens.
A repeated 219-token image request cannot demonstrate GPU prefix reuse;
zero hits are expected. The `prefix_cache_queries_total` Prometheus counter
counts queried tokens, not API requests.

On both r8 and r9, repeated 4,049-token image-bearing prompts returned
3,840 cached tokens. Image preprocessing/encoder caching, GPU KV prefix
caching, and LMCache host-tier reuse are separate mechanisms. A cache report
must identify which mechanism its counters measure.

## Source Composition

| Component | Canonical base | Applied PRs | Integration tree |
|---|---|---|---|
| vLLM | `dev/jovian-judgement@2a979314dc97b03173a0a76fc15664ec924db32b` | #628, #630, #634, #553, #671, #679, #694, #695 | `f66599d9a90d57172fd26ca5b9116f381b582b94` |
| B12X | `master@06b4de7c723e6f166d65abf5909c5b7d0f8acc68` | #301 | `15b6813011bd47e466b39f9b474b3bca0c48c8e8` |
| LMCache | `dev@7ed4675404a31f4ffafd98975899dc83832ba965` | #49, #50, #51, #55, #56 | `d85748de9bf985dabc00c044396a3b8de97f4ac1` |

B12X #246 is included through master and is not applied twice. Source PRs
remain subject to maintainer review; publishing this image does not merge them.

## Qualification Limits

The r8 reference completed 96 varied continuations in 24 concurrent waves,
each with exactly 267,008 cached tokens, without a service failure. This did
not reproduce the reported multi-hour workload. A successful bounded test
does not close a delayed crash report.

For a continuing failure, provide the image digest, checkpoint revision,
effective launch settings, both workers' complete logs, and a redacted request
sequence preserving prompt lengths and cache reuse. The first allocator or
worker error matters more than the final asynchronous-error stack. Remove
credentials and private conversation content before sharing artifacts.

Temperature-zero byte-repeatability and the separate `)Skip` report are
outside this release's investigation scope. TP greater than two, DCP greater
than one, GPUs below 96 GiB, and task-level quality comparisons are not
qualified by the TP2 release checks.
