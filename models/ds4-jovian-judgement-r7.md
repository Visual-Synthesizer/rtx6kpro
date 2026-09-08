# DeepSeek-V4-Flash Jovian Judgement r7

This serving specification covers `deepseek-ai/DeepSeek-V4-Flash-0731` on two
NVIDIA SM120 GPUs. The runtime uses vLLM, B12X, fixed probabilistic DeepSeek
DSpark K5 speculative decoding, FP8 compressed multi-head latent attention
(MLA) KV storage, InstantTensor model loading, and optional engine-driven
LMCache host storage.

**Status: qualified.** The live qualification covers TP2/DCP1 text serving on
two 96 GiB RTX PRO 6000 Blackwell GPUs. DeepSeek V4 Vision source support is
implemented but was not executed against the r7 registry identity.

## TL;DR

Download the committed Compose profile, pull the prebuilt image, and start the
text checkpoint on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r7.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r7.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r7.yml up -d
```

Enable a 24 GiB in-memory LMCache tier:

```bash
LMCACHE_MODE=ram LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r7.yml up -d
```

The Compose profile references a prebuilt image and contains no `build`
section. GPU KV storage remains active in both commands. LMCache is an
additional host-memory reuse tier and is disabled unless `LMCACHE_MODE` is set.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllmc0f75cf-b12xaa76f04-fi803c466-cu133-torch213-20260906-r7` |
| Registry digest | `sha256:8a34b0be7be5315256f07181c2e3f684a4f086337bbc0363a1066dcd06b666a4` |
| Image ID | `sha256:e51c9b0af6364e0d369ddd09b976bafe9adcc21041c469053ad4be5e48f2b7cc` |
| Image size | 34,640,909,635 bytes |
| Docker source used by the image | `local-inference-lab/blackwell-llm-docker@95f8751c6fcdcb62adc8d71678d7ae479dadec6b` |
| Validation receipt | [r7 InstantTensor qualification](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r7-instanttensor.json) |
| Source merge contract | [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, B12X 1.3.0, FlashInfer 0.6.18+cu133, LMCache 0.5.2+jj.ds4.r6, XGrammar 0.2.5, InstantTensor 0.1.9 |

OCI labels record each source base, pull-request head, integration tree, patch
digest, and dependency revision. The runtime contains compiled installed
packages and no source overlay.

## Loader Contract

DeepSeek V4 text and Vision serving use `InstantTensor 0.1.9` with the
`BUFFERED` backend by default. The launcher passes:

```text
--load-format instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

The buffered loader owns the CUDA tensors used after loading and does not
retain FastSafeTensors mappings in the model execution path. An installation
that requires FastSafeTensors can select it explicitly:

```bash
LOAD_FORMAT=fastsafetensors \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r7.yml up -d
```

The separate Spark launcher uses FastSafeTensors by default because its model
loading contract depends on that loader. Changing the DeepSeek V4 text and
Vision default does not change Spark serving.

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
| Weight loader | InstantTensor 0.1.9, `BUFFERED` |
| `MAX_NUM_SEQS` | `8` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| `MAX_MODEL_LEN` | `1048576` |
| CUDA graph cap | `48` |
| CUDA graph mode | `FULL_AND_PIECEWISE` |
| GPU memory utilization | `0.975` |

The launcher captures every scheduler-reachable target and DSpark row count.
The qualified process captured 9/9 PIECEWISE target shapes, 7/7 FULL target
shapes, and 6/6 FULL DSpark shapes. Decode does not depend on an eager fallback
for the configured serving envelope.

## Runtime Evidence

The qualification used GPUs 14 and 15 on a 16-GPU RTX PRO 6000 Blackwell
system. No `LOAD_FORMAT` or `INSTANTTENSOR_BACKEND` override was supplied.

| Measurement | Result |
|---|---:|
| Target weight load | 155G in 68.89 s |
| DSpark weight load | 155G in 68.63 s |
| Complete model load | 145.797 s |
| Model memory | 80.67 GiB/rank |
| GPU KV capacity | 1,247,312 tokens |
| GPU KV memory | 8.27 GiB/rank |
| One-million-token concurrency | 1.19x |
| API response | `Model generation works.` |

The target and draft both reported `Loading safetensors using InstantTensor
loader` and completed without a loader, graph-capture, CUDA, or engine error.

One zero-context request was measured for 20 seconds after a five-second
warmup. This measurement is a runtime health gate on GPUs 14 and 15, not a
cross-revision performance comparison:

| Metric | Result |
|---|---:|
| Output throughput | 154.2 tok/s |
| Engine step rate | 59.2 steps/s |
| Mean inclusive acceptance length | 2.605 tokens |
| Strict draft acceptance | 32.09% |
| Time to first token | 0.223 s |
| Errors | 0 |

Loader and decode throughput depend on storage locality, PCIe topology, GPU
power state, and generated-token acceptance. The r7 qualification used a
different GPU pair and host I/O path from the r6 performance receipt, so these
numbers do not establish a loader speed delta.

## LMCache Contract

`LMCACHE_MODE=ram` creates a named shared-memory pool owned by the standalone
LMCache process. Engine-driven transfer means the vLLM workers perform the
device-to-host and host-to-device copies; the standalone process creates no
CUDA context. `LMCACHE_L1_GB` controls host capacity, not GPU KV capacity.

The r7 image retains the exact B12X and LMCache integration trees qualified by
the [r6 engine-driven LMCache receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r6-engine-driven-lmcache.json).
That receipt covers a 144,028-token cold request, a 143,360-token restore after
GPU prefix-cache reset, and 16/16 byte-exact rank/object-group comparisons. The
r7 loader change does not execute in the KV transfer path. The 144k transfer
test was not repeated against the r7 image identity.

## Source Composition

### vLLM

Base: `dev/jovian-judgement@b7e3d033676d5db46fb7d6cdd40d760365a1e239`  
Integration tree: `c0f75cf4ea4eb158522b58571c681bd914187e52`

| Pull request | Resulting behavior |
|---|---|
| [#628](https://github.com/local-inference-lab/vllm/pull/628) | Registers speculative verifier-row counts before B12X graph warmup. |
| [#630](https://github.com/local-inference-lab/vllm/pull/630) | Clears custom all-reduce selectors when NCCL is selected explicitly. |
| [#634](https://github.com/local-inference-lab/vllm/pull/634) | Provides DeepSeek V4 Vision, incremental loading, serving profiles, unbounded `MAX_MODEL_LEN=-1` handling, and the InstantTensor default for DeepSeek V4 text and Vision. |
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
| vLLM DeepSeek V4 launcher tests | passed |
| vLLM launcher syntax | passed |
| Docker DS4 r7 release contract | passed |
| Docker Compose release tests | 11 passed |
| Explicit FastSafeTensors override | passed |
| Spark loader contract | FastSafeTensors retained |
| JSON and repository diff validation | passed |

## Qualification Limits

- **Qualified:** TP2/DCP1 text serving, fixed probabilistic DSpark K5, B12X
  W4A8, FP8 compressed MLA KV, InstantTensor with the BUFFERED backend, target
  and draft CUDA graphs, API generation, and a zero-context CC1 health gate.
- **Implemented:** GPU-only KV storage, LMCache RAM storage, LMCache filesystem
  storage, explicit direct LMCache transfer, FastSafeTensors override, and
  DeepSeek V4 Vision support.
- **Unsupported:** native vLLM filesystem KV offload under this serving
  contract and Vision speculative depth above K3.
- **Not qualified by the r7 receipt:** TP other than two, DCP greater than one,
  GPUs with less than 96 GiB, Vision execution, repeated 144k LMCache transfer,
  abrupt-host-failure filesystem durability, and task-level model quality.
