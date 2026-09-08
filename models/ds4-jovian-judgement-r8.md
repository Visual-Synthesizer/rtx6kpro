# DeepSeek-V4-Flash Jovian Judgement r8

This serving specification covers `deepseek-ai/DeepSeek-V4-Flash-0731` and
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on two NVIDIA SM120 GPUs. The
runtime uses vLLM, B12X, fixed probabilistic DeepSeek DSpark speculative
decoding, FP8 compressed multi-head latent attention (MLA) KV storage,
InstantTensor model loading, and optional engine-driven LMCache host storage.

**Status: qualified.** The exact TP2 image completed scheduler-quantum and
810,000-token prefills for text, Vision, and engine-driven LMCache profiles.
The Vision gate overlapped the long text prefill with ten 2048x2048 image
encodes. Every service remained healthy and no allocation failed.

## TL;DR

Start the text checkpoint on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r8.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r8.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r8.yml up -d
```

Start the Vision checkpoint:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r8.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r8.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r8.yml up -d
```

Enable a 24 GiB in-memory LMCache tier for the text checkpoint:

```bash
LMCACHE_MODE=ram LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r8.yml up -d
```

GPU KV storage remains active when LMCache is enabled. LMCache is an
additional host-memory reuse tier and is disabled unless `LMCACHE_MODE` is
set. Both Compose files reference a prebuilt image and contain no `build`
section.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllmd267ca7-b12xe58515a-fi803c466-cu133-torch213-20260906-r8` |
| Registry digest | `sha256:7f4de2bb4faf58d9b05c508037de57415dc4e5b7d53a24588db0ed0bcb7f8968` |
| Image ID | `sha256:7e533ec88bcd04979f05c737768f75de0fb16fa80391c1b5d104565092f85579` |
| Image size | 34,642,440,427 bytes |
| Docker source used by the image | `local-inference-lab/blackwell-llm-docker@f68d9412248e1c7ec38966a2475e9790cca409d6` |
| Validation receipt | [`validation/jovian-judgement-ds4-r8.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r8.json) |
| Source merge contract | [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, B12X 1.3.0, FlashInfer 0.6.18+cu133, LMCache 0.5.2+jj.ds4.r6, XGrammar 0.2.5, InstantTensor 0.1.9 |

OCI labels record every source base, pull-request head, integration tree,
patch digest, and dependency revision. The runtime contains compiled installed
packages and no source overlay.

## Memory Admission Contract

GPU KV admission reserves the measured maximum non-KV memory needed by every
DeepSeek V4 execution shape reachable under `MAX_NUM_BATCHED_TOKENS`.

The standard vLLM profile distributes its token budget across requests and
omits attention. DeepSeek V4 can instead execute the complete token budget as
one prefill request. That shape creates the largest query projection and
sparse-indexer transient. Vision serving can hold multimodal encoder outputs
at the same time.

The DeepSeek V4 model runners in vLLM integration tree
`d267ca78d0d07e3993093b023844619af41a5e11` therefore perform an additional
architecture-gated profile with these invariants:

- the complete scheduler token budget belongs to one prefill request;
- attention executes against a minimal temporary KV cache;
- the single-request profile allocates one temporary KV block rather than one
  block per configured request;
- every hybrid KV group participates in the modular model runner;
- Vision encoder outputs remain resident during the attention profile;
- generic-profile outputs are released because the two profiles represent
  mutually exclusive scheduler steps;
- temporary KV, attention, and graph state is released after success or
  failure.

CUDA peak-memory accounting observes the larger reachable transient peak and
assigns all remaining memory to GPU KV storage. The implementation does not
use a fixed memory reserve, lower the standard `0.975` GPU utilization, retain
a permanent query workspace, or change attention arithmetic.

## Serving Profiles

| Setting | Text | Vision |
|---|---:|---:|
| Checkpoint | `DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | `DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| Tensor / decode-context parallelism | TP2 / DCP1 | TP2 / DCP1 |
| Speculative decoding | fixed probabilistic DSpark K5 | fixed probabilistic DSpark K3 |
| `BACKEND` | `b12x-a8-dglin` | `b12x-a8-dglin` |
| Attention | B12X sparse MLA | B12X sparse MLA |
| Routed experts | B12X W4A8 | B12X W4A8 |
| Dense projections | DGLIN FP8 | DGLIN FP8 |
| KV format | FP8 compressed MLA | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |
| `MAX_NUM_SEQS` | `8` | `4` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` |
| `MAX_MODEL_LEN` | `1048576` | `1048576` |
| CUDA graph cap | `48` | `16` |
| CUDA graph mode | `FULL_AND_PIECEWISE` | `FULL_AND_PIECEWISE` |
| GPU memory utilization | `0.975` | `0.975` |

The Vision checkpoint contains three draft layers, so K3 is its deepest
supported DSpark mode. The target consumes and verifies image embeddings. The
drafter proposes from text-only inputs because it has no external multimodal
embedding interface; target verification remains authoritative.

## Runtime Evidence

The qualification used two 96 GiB NVIDIA RTX PRO 6000 Blackwell Workstation
Edition GPUs connected through one PCIe switch. Each profile used the exact
registry image listed above.

| Profile | GPU utilization | GPU KV tokens | GPU KV/rank | 1M concurrency | Minimum free/rank |
|---|---:|---:|---:|---:|---:|
| Text, K5 | 0.975 | 1,185,524 | 7.86 GiB | 1.13x | 1,043 / 1,023 MiB |
| Vision, K3 | 0.975 | 1,206,495 | 8.00 GiB | 1.15x | 1,329 / 1,329 MiB |
| Text K5 + engine-driven LMCache | 0.970 | 1,113,883 | 7.39 GiB | 1.06x | 1,599 / 1,457 MiB |

The capacity values include the measured execution reserve; they are not
capacities obtained by omitting a reachable transient. No fixed padding is
subtracted. A smaller or larger scheduler token budget is profiled from its
configured `MAX_NUM_BATCHED_TOKENS` value.

| Execution gate | Result |
|---|---|
| Text scheduler quantum | Exact 4,096 prompt tokens; HTTP 200 in 0.876 s |
| Text long prefill | Exact 810,000 prompt tokens; HTTP 200 in 127.901 s |
| Vision scheduler quantum | Exact 4,096 prompt tokens; HTTP 200 in 0.881 s |
| Vision overlap | 810,000-token prefill plus ten 2048x2048 images; both HTTP 200 in 139.97 s or less |
| LMCache scheduler quantum | Exact 4,096 prompt tokens; HTTP 200 in 0.304 s |
| LMCache long prefill | Exact 810,000 prompt tokens; HTTP 200 in 121.817 s |
| LMCache restore after GPU eviction | 4,095 usable prompt tokens reported by vLLM; 4,096 lookup-hit tokens reported by LMCache |

Text captured 9/9 PIECEWISE target graphs, 7/7 FULL target graphs, and 6/6
FULL DSpark graphs. Vision captured 5/5 PIECEWISE target graphs, 3/3 FULL
target graphs, and 3/3 FULL DSpark graphs. No qualified request used an eager
decode fallback.

## Loader Contract

DeepSeek V4 text and Vision serving use InstantTensor 0.1.9 with the `BUFFERED`
backend by default:

```text
--load-format instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

An installation that requires FastSafeTensors can select it explicitly with
`LOAD_FORMAT=fastsafetensors`. The separate Spark launcher uses
FastSafeTensors by default because its loading contract depends on that
loader.

## LMCache Contract

`LMCACHE_MODE=ram` creates a named shared-memory pool owned by the standalone
LMCache process. Engine-driven transfer means vLLM workers perform device-to-
host and host-to-device copies; the standalone process creates no CUDA
context. `LMCACHE_L1_GB` controls host capacity, not GPU KV capacity.

The one-million-token TP2 engine-driven profile uses
`GPU_MEMORY_UTILIZATION=0.970`. Its asynchronous transfer lifetime is measured
separately from model execution. Direct LMCache transfer remains an explicit
compatibility mode and uses a smaller qualified GPU memory envelope because
the cache process creates CUDA resources after vLLM memory profiling.

The byte-exact engine-driven transfer contract covers every DeepSeek V4 hybrid
KV group. A 144,028-token cold request and 143,360-token restore produced
16/16 exact rank/object-group comparisons under the r6 integration tree. The
r8 image retains that exact LMCache integration tree.

## Source Composition

### vLLM

Base: `dev/jovian-judgement@7d66922a7bf0c9c7efe9a35a87df128f6b24e762`  
Integration tree: `d267ca78d0d07e3993093b023844619af41a5e11`

| Pull request | Resulting behavior |
|---|---|
| [#628](https://github.com/local-inference-lab/vllm/pull/628) | Registers speculative verifier-row counts before B12X graph warmup. |
| [#630](https://github.com/local-inference-lab/vllm/pull/630) | Clears custom all-reduce selectors when NCCL is selected explicitly. |
| [#634](https://github.com/local-inference-lab/vllm/pull/634) | Provides DeepSeek V4 Vision, incremental loading, image-aware sparse attention, and serving profiles. |
| [#553](https://github.com/local-inference-lab/vllm/pull/553) | Allows expandable allocator segments with engine-driven LMCache. |
| [#671](https://github.com/local-inference-lab/vllm/pull/671) | Makes the DeepSeek V4 padded-query output caller-owned and reserves it before GPU KV admission. |
| [#679](https://github.com/local-inference-lab/vllm/pull/679) | Profiles the scheduler-reachable DeepSeek V4 attention peak before GPU KV admission. |

### B12X

Base: `master@a1bbd02781c7505754e7aa58a959c1a77891c690`  
Integration tree: `e58515a63b7b5d15bbc523258e1e338f49698ce3`

| Pull request | Resulting behavior |
|---|---|
| [#246](https://github.com/local-inference-lab/b12x/pull/246) | Provides generation-safe TP2 graph peer-push and PIECEWISE shape binding. |
| [#301](https://github.com/local-inference-lab/b12x/pull/301) | Supports FP8 DeepSeek V4 dual-cache prefill with sparse top-k 512. |

The B12X base includes merged pull request
[#306](https://github.com/local-inference-lab/b12x/pull/306), which supports
the Vision checkpoint's `rms_norm_eps=1e-20` mHC contract.

### LMCache

Base: `dev@7ed4675404a31f4ffafd98975899dc83832ba965`  
Integration tree: `86ee2a3bb5675cd3a25b09ad3e2f20dad4720f58`

| Pull request | Resulting behavior |
|---|---|
| [#49](https://github.com/local-inference-lab/LMCache/pull/49) | Preserves oversized filesystem keys across native restarts. |
| [#50](https://github.com/local-inference-lab/LMCache/pull/50) | Reuses registered CUDA metadata for engine-driven paged transfer. |
| [#51](https://github.com/local-inference-lab/LMCache/pull/51) | Bounds ownership and cleanup of named shared-memory pools. |
| [#55](https://github.com/local-inference-lab/LMCache/pull/55) | Synchronizes asynchronous-copy source lifetimes; direct backport of upstream #4830. |
| [#56](https://github.com/local-inference-lab/LMCache/pull/56) | Reports `engine_driven_shm_pool` as the active transport identity. |

All listed source changes remain pull-request dependencies until their target
repositories merge them. The image composes the exact heads recorded in its
OCI labels and validation receipt.

## Correctness And Regression Coverage

| Suite | Result |
|---|---:|
| Legacy model runner DeepSeek memory tests | 8 passed |
| Modular model runner complete test file | 15 passed |
| vLLM formatting and static checks | passed |
| Docker source-composition contract | passed |
| Docker Compose profile contract | passed |
| Runtime package and import contract | passed |
| TP2 text memory-pressure execution | passed |
| TP2 Vision memory-pressure execution | passed |
| TP2 engine-driven LMCache memory pressure and restore | passed |

## Qualification Limits

- **Implemented:** TP2/DCP1 text fixed K5, Vision fixed K3, B12X W4A8, FP8
  compressed MLA KV, InstantTensor loading, and engine-driven LMCache.
- **Unsupported:** native vLLM filesystem KV offload and Vision speculative
  depth greater than three.
- **Not qualified:** TP1, TP greater than two, DCP greater than one, GPUs with
  less than 96 GiB, abrupt-host-failure persistence, comparative decode
  throughput, and task-level model quality.
