# DeepSeek-V4-Flash Jovian Judgement r6

This serving specification covers `deepseek-ai/DeepSeek-V4-Flash-0731` on two
NVIDIA SM120 GPUs. The runtime uses vLLM, B12X, fixed probabilistic DeepSeek
DSpark K5 speculative decoding, FP8 compressed multi-head latent attention
(MLA) KV storage, and optional engine-driven LMCache host storage.

**Status: qualified.** The live qualification covers TP2/DCP1 text serving on
two 96 GiB RTX PRO 6000 Blackwell GPUs connected through one PCIe switch.
DeepSeek V4 Vision source support is present, but the r6 registry identity has
not received a separate live Vision qualification.

## TL;DR

Download the committed Compose profile, pull the prebuilt image, and start the
text checkpoint on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r6.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r6.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r6.yml up -d
```

Enable the qualified 24 GiB in-memory LMCache tier:

```bash
LMCACHE_MODE=ram LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r6.yml up -d
```

The Compose profile references a prebuilt image and contains no `build`
section. GPU KV storage remains active in both commands. LMCache is an
additional host-memory reuse tier and is disabled unless `LMCACHE_MODE` is set.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllma67b59a-b12xaa76f04-fi803c466-cu133-torch213-20260905-r6` |
| Registry digest | `sha256:8222ac5d319c0f4dae04e6a2abd379745c6a2433b1c5c7b454c9fed076d84b08` |
| Image ID | `sha256:7e684820db5b51e0097c185e34cbc2320374c3b08dc8ededad93e5beee206086` |
| Image size | 34,639,500,367 bytes |
| Docker source used by the image | `local-inference-lab/blackwell-llm-docker@ae44f25029014e417e2aa4dd8178f0340bf0a720` |
| Validation receipt | [r6 engine-driven LMCache qualification](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r6-engine-driven-lmcache.json) |
| Source merge contract | [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, B12X 1.3.0, FlashInfer 0.6.18+cu133, LMCache 0.5.2+jj.ds4.r6, XGrammar 0.2.5 |

OCI labels record each source base, pull-request head, integration tree, patch
digest, and dependency revision. The runtime contains compiled installed
packages and no source overlay.

## Serving Profile

| Setting | Qualified value |
|---|---|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Tensor / decode-context parallelism | TP2 / DCP1 |
| Speculative decoding | fixed probabilistic DSpark K5 |
| `BACKEND` | `b12x-a8-dglin` |
| Attention | B12X sparse MLA |
| Routed experts | B12X W4A8 |
| Dense projections | DGLIN FP8 |
| All-reduce | B12X automatic policy |
| KV format | FP8 compressed MLA |
| Weight loader | FastSafeTensors |
| `MAX_NUM_SEQS` | `8` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| `MAX_MODEL_LEN` | `1048576` |
| CUDA graph cap | `48` |
| CUDA graph mode | `FULL_AND_PIECEWISE` |
| GPU memory utilization | `0.970` |

The launcher captures every scheduler-reachable target and DSpark row count.
The qualified process captured 9/9 PIECEWISE target shapes, 7/7 FULL target
shapes, and 6/6 FULL DSpark shapes. Decode does not depend on an eager fallback
for the configured serving envelope.

## GPU And Host KV Capacity

At the qualified scheduler and graph settings, vLLM allocated 1,199,317 GPU KV
tokens, or 7.95 GiB per tensor-parallel rank. The reported maximum concurrency
for a 1,048,576-token request was 1.14x.

`LMCACHE_MODE=ram` creates a named shared-memory pool owned by the standalone
LMCache process. Engine-driven transfer means the vLLM workers perform the
device-to-host and host-to-device copies; the standalone process creates no
CUDA context. The 24 GiB profile creates one object such as:

```text
/dev/shm/lmcache_l1_pool_lmcache-8000
```

`LMCACHE_L1_GB` controls host capacity, not GPU KV capacity. The Docker shared
memory limit must exceed the requested L1 size; the committed Compose profile
uses 32 GiB by default.

## Hybrid-KV Integrity

DeepSeek V4 registers eight heterogeneous KV object groups per tensor-parallel
rank. A valid external-cache restore must preserve each group's physical shape,
stride, dtype, and chunk lifetime.

The exact r6 image processed a 144,028-token cold request, reset vLLM's GPU
prefix cache, and restored the same prefix through engine-driven LMCache:

| Measurement | Result |
|---|---:|
| Cold cached prompt tokens | 0 |
| Cold elapsed time | 12.159 s |
| Restored prompt tokens | 143,360 |
| Restore elapsed time | 3.149 s |
| Rank/object-group comparisons | 16/16 byte-exact |
| Generated output | Identical single token |
| Service health after restore | Healthy |

The comparison hashes every retained source and destination chunk. Matching
generated output alone is not used as evidence of KV integrity.

## Decode Measurement

The sustained decode measurement used one request, zero user-context tokens,
a 30-second window, fixed probabilistic DSpark K5, and the serving profile
listed above.

| Metric | Result |
|---|---:|
| Output throughput | 201.1 tok/s |
| Engine step rate | 72.6 steps/s |
| Mean inclusive acceptance length | 2.771 tokens |
| Strict draft acceptance | 35.42% |
| Time to first token | 0.234 s |
| Errors | 0 |

DSpark output throughput varies with workload predictability because accepted
draft length changes with the generated token trajectory. Engine step rate is
the backend execution metric for comparisons that need to remove that source
of variation.

## Source Composition

### vLLM

Base: `dev/jovian-judgement@b7e3d033676d5db46fb7d6cdd40d760365a1e239`  
Integration tree: `a67b59a4099457fbcdadce4476c88504fafaf083`

| Pull request | Resulting behavior |
|---|---|
| [#628](https://github.com/local-inference-lab/vllm/pull/628) | Registers speculative verifier-row counts before B12X graph warmup. |
| [#630](https://github.com/local-inference-lab/vllm/pull/630) | Clears custom all-reduce selectors when NCCL is selected explicitly. |
| [#634](https://github.com/local-inference-lab/vllm/pull/634) | Provides DeepSeek V4 Vision, incremental loading, serving profiles, and unbounded `MAX_MODEL_LEN=-1` handling. |
| [#553](https://github.com/local-inference-lab/vllm/pull/553) | Allows expandable allocator segments with engine-driven LMCache. |
| [#671](https://github.com/local-inference-lab/vllm/pull/671) | Makes padded-query output caller-owned and accounts for it before GPU KV admission. |

### B12X

Base: `master@d27805aef99ae0ad092f79fc458aa1fae1a580e3`  
Integration tree: `aa76f044cbe43c191d33c0c9232e42193b16a544`

| Pull request | Resulting behavior |
|---|---|
| [#246](https://github.com/local-inference-lab/b12x/pull/246) | Provides generation-safe TP2 graph peer-push and PIECEWISE shape binding. |
| [#301](https://github.com/local-inference-lab/b12x/pull/301) | Supports FP8 DeepSeek V4 dual-cache prefill with sparse top-k 512. |
| [#306](https://github.com/local-inference-lab/b12x/pull/306) | Supports the Vision checkpoint's `rms_norm_eps=1e-20` contract. |

B12X #302 is not part of the image and is not required by this serving
contract.

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
| LMCache focused tests | 189 passed |
| NVIDIA/AMD DSpark context-insert tests | 2 passed |
| vLLM Ruff check and format | passed |
| DS4 r6 launcher and Compose contract | passed |
| GLM-5.2 LMCache helper compatibility | passed |
| JSON and repository diff validation | passed |

## Qualification Limits

- **Qualified:** TP2/DCP1 text serving, fixed probabilistic DSpark K5, B12X
  W4A8, FP8 compressed MLA KV, FastSafeTensors, target and draft CUDA graphs,
  engine-driven shared-memory transfer, 144k cold/restore integrity, and CC1.
- **Implemented:** GPU-only KV storage, LMCache RAM storage, LMCache filesystem
  storage, explicit direct LMCache transfer, and DeepSeek V4 Vision support.
- **Unsupported:** native vLLM filesystem KV offload under this serving
  contract and Vision speculative depth above K3.
- **Not qualified by this receipt:** TP other than two, DCP greater than one,
  GPUs with less than 96 GiB, r6 Vision execution, abrupt-host-failure
  filesystem durability, and task-level model quality.
