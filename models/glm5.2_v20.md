# GLM-5.2 v20: Gilded Gnosis DCP Release

v20 is the tested successor to [v19](glm5.2_v19.md). It keeps the same GLM-5.2,
NF3, MXFP4, DCP, MTP, and InstantTensor launch contract while updating the
canonical GG/SparkInfer stack, fixing two release blockers, and adding the
measured DCP prefill topology:

- DCP outputs now preserve a cuBLAS-safe physical head-major layout without a
  hot-path clone or tail-padding reservation;
- virtual TP6 accepts partial pitched DCP workspaces and correctly plans the
  N128-padded W4A8-MX scratch extent;
- exact owner top-k merge, partial indexer replication, and a bounded
  one-layer CKV prefetch improve DCP prefill without lossy transport.

Historical comparison data remains on [v18](glm5.2_v18.md), while the DCP
optimization background remains on [v19](glm5.2_v19.md). This page is
self-contained for building, starting, operating, and validating v20; older
pages are provenance, not required setup instructions.

Canonical source merging and the required post-merge rebuild are tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).
The image below is the exact measured release candidate. The listed source
PRs remain independently reviewable and were not merged while producing it.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725
Docker manifest: sha256:e7a8a8549c10b5d16899e0fb45ff7eeca09dd7c1d1a83eee13fb03930d8eb80a
Local image ID: sha256:727ac3af71b729be93fd01c9fe60472c27f619c8ef0e3c67b4f627023933027c
```

This supersedes the earlier `vllm83a1f7f` candidate. Its hand-resolved stacked
integration let the partial-indexer change restore a `DCP>1` gate over the
DCP1 query-split path. The final source tree is a conflict-free merge of the
public PR heads and restores DCP1 query split without a private patch.

`si` identifies SparkInfer, the renamed B12X project. Legacy B12X environment
variable names remain accepted for compatibility.

Pinned source stack:

| Component | Ref / commit |
|---|---|
| Canonical GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58` |
| vLLM release source | `voipmonitor/vllm build/gilded-gnosis-v20-dcp-final2-20260725` @ `551719766029e78824a30d97ae6ac63917405b5f` |
| SparkInfer base | `local-inference-lab/sparkinfer master` @ `c39b8062ba450c030e669d898a026d10980c9470` |
| SparkInfer release source | `local-inference-lab/sparkinfer build/sparkinfer-v20-dcp-final-20260725` @ `be0edcaae6f5d284bb29a82325aba7a0ead6960f` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| CUTLASS C++ / DSL | `e6233cbac5d7c7a865c19c91cd684ceece19513c` / `4.6.0` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA / cuDNN | `2.12.0+cu132` / `13.2.1` / `9.22.0.52` |
| Launcher source | `local-inference-lab/blackwell-llm-docker` @ `48c8add4907775babeac03da68ee47224c23475c` |
| Build recipe | `local-inference-lab/blackwell-llm-docker` @ `b620596be1b8955fe3c4bf2854b65d6bff38aaaf` |

The image contains no `VLLM_PATCH_URL`, `VLLM_PATCH_FILE`, source bind mount,
or private source overlay. Image labels expose every source pin and a cache
fingerprint derived from the vLLM and SparkInfer commits.

## Build It Exactly

The canonical build entry point is
[`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b620596be1b8955fe3c4bf2854b65d6bff38aaaf/build-gilded-gnosis-v20-final-cu132.sh).
It builds with the exact commits above, validates runtime symbols and source
contracts, verifies the image labels, and only then allows an optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout b620596be1b8955fe3c4bf2854b65d6bff38aaaf
PUSH_IMAGE=1 ./build-gilded-gnosis-v20-final-cu132.sh
```

The build deliberately excludes the separate weight-lifetime experiments in
vLLM PR #154, vLLM PR #157, and SparkInfer PR #62. It also excludes the
experimental sparse-CKV decode stack in vLLM PRs #159-#161 and SparkInfer PRs
#64-#65.

## Source Changes

The cuBLAS/Xid correction is already in the pinned GG base through
[vLLM PR #147](https://github.com/local-inference-lab/vllm/pull/147) and
[SparkInfer PR #54](https://github.com/local-inference-lab/sparkinfer/pull/54).
The release source adds these independently reviewable deltas:

| Project | Review | Purpose |
|---|---|---|
| vLLM | [#145](https://github.com/local-inference-lab/vllm/pull/145) | Calibrated NVFP4 MLA KV outer-scale wiring. |
| vLLM | [#172](https://github.com/local-inference-lab/vllm/pull/172) | Profile persistent kernel resources before allocating KV cache. |
| vLLM | [#175](https://github.com/local-inference-lab/vllm/pull/175) | Split sparse prefill queries and reduce gathered result traffic. |
| vLLM | [#177](https://github.com/local-inference-lab/vllm/pull/177) | Preallocate a bounded, memory-profiled CKV prefetch workspace. |
| vLLM | [#178](https://github.com/local-inference-lab/vllm/pull/178) | Merge exact FP32 sparse top-k candidates by query-row owner. |
| vLLM | [#179](https://github.com/local-inference-lab/vllm/pull/179) | Add partial replicated-indexer topology and mixed target/draft grouping. |
| SparkInfer | [#76](https://github.com/local-inference-lab/sparkinfer/pull/76) | Account persistent PCIe DMA output storage during KV profiling and release it on close. |

The release build itself does not merge canonical branches. Its exact
integration branches contain only the GG/SparkInfer bases and the reviews
listed above. There is no runtime patch file or source bind mount.

### Canonical Merge Status

[Issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33) is the
authoritative ordered merge checklist. As of 2026-07-25, every runtime PR above
is non-draft, mergeable against `dev/gilded-gnosis` or `sparkinfer/master`, and
has no unresolved review thread. The `pre-commit` jobs for #172 and #175 are
queued without a runner; their pre-run checks passed and no check failed.

PR #145 is intentionally present in the image but is not requested for merge
yet. The exact SparkInfer candidate-owner transport in
[SparkInfer #79](https://github.com/local-inference-lab/sparkinfer/pull/79)
was measured separately and is not in this release image or its default path.
The broader DCP design and rejected experiments are recorded in
[research issue #35](https://github.com/local-inference-lab/rtx6kpro/issues/35).
The subsequent remote selected-record and query-sharding POC is archived with
its exact source branches, tests, and measurements in
[research issue #36](https://github.com/local-inference-lab/rtx6kpro/issues/36).
Those paths were correct but slower than the retained local-CKV design; they
do not change this image, its defaults, or the canonical merge checklist.

## Start The Server

The helper is inside the image, so users do not need to download a launch
script. Docker with NVIDIA Container Toolkit, host IPC, and at least four
Blackwell GPUs is required. Pull the immutable image first:

```bash
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725
```

Save the following as `compose.yml`. Bare environment entries pass a host
variable only when it is set; otherwise the helper chooses the correct default
for `MODEL_FAMILY`. The two explicit memory entries raise the recommended
standard TP8 service to a 262k context and a validated 0.96 memory budget. The
TP6 recipe below overrides the memory budget with its separately validated
limit.

```yaml
services:
  glm52:
    image: voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725
    entrypoint: ["/usr/local/bin/serve-gilded-gnosis.sh"]
    network_mode: host
    ipc: host
    privileged: true
    init: true
    shm_size: 32gb
    gpus: all
    ulimits:
      memlock: -1
      stack: 67108864
      nofile:
        soft: 1048576
        hard: 1048576
    environment:
      - MODEL_FAMILY=${MODEL_FAMILY:-glm52}
      - MODEL
      - MODEL_REVISION
      - SERVED_MODEL_NAME
      - GPUS
      - PORT
      - TP
      - DCP
      - DCP_BACKEND
      - DCP_A2A_MAX_TOKENS
      - DCP_A2A_LARGE_BACKEND
      - DCP_QUERY_SPLIT
      - DCP_CKV_GATHER
      - DCP_TOPK_OWNER_MERGE
      - DCP_INDEXER_SHARDS
      - DCP_CKV_PREFETCH_DEPTH
      - DCP_CKV_PREFETCH_WORKSPACE_MIB
      - DCP_PREFILL_WORKSPACE
      - MTP
      - MAX_NUM_SEQS
      - GRAPH
      - MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
      - MAX_BATCHED_TOKENS
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.96}
      - MOE_MODE
      - MOE_BACKEND
      - LINEAR_BACKEND
      - QUANTIZATION
      - ONLINE_QUANT
      - QUANTIZATION_CONFIG_JSON
      - KV_CACHE_DTYPE
      - F8_DMA
      - B12X_PCIE_DMA
      - NF3_GRID188
      - LOAD_FORMAT
      - INSTANTTENSOR_BACKEND
      - PYTORCH_CUDA_ALLOC_CONF
      - DRY_RUN
    volumes:
      - ${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface
      - ${MODEL_ROOT:-/root/models}:/root/models:ro
      - ${JIT_CACHE:-./cache/glm52-v20}:/cache
      - ${CONTAINER_TMP:-./cache/glm52-v20/tmp}:/container-tmp
```

The image helper and Compose contract both use `MAX_MODEL_LEN=262144` and
`GPU_MEMORY_UTILIZATION=0.96` for standard TP8. Virtual TP6 remains separately
validated at `128000` and `0.95`.

### Start, Inspect, And Stop

The standard model preset is Luke NVFP4, TP8/DCP1, native A4, MTP off. The
highest-accuracy standard launch changes only `MOE_MODE` to A16:

```bash
MOE_MODE=a16 docker compose up -d
docker compose logs -f glm52
```

Wait for the health endpoint before sending traffic:

```bash
curl -fsS http://127.0.0.1:${PORT:-8000}/health
curl -fsS http://127.0.0.1:${PORT:-8000}/v1/models | jq .
```

The first start compiles kernels. Reuse the same `JIT_CACHE` for the same image
and configuration family; do not benchmark while this or another model is
still loading. Stop the service without deleting either model or JIT cache:

```bash
docker compose down
```

Inspect the fully expanded environment and `vllm serve` command without loading
weights:

```bash
DRY_RUN=1 MOE_MODE=a16 docker compose run --rm --no-deps glm52
```

### Common Launch Recipes

These commands use the same Compose file. Variables not shown remain owned by
the image helper.

```bash
# Luke NVFP4, highest-accuracy routed-expert mode, no speculation.
MOE_MODE=a16 MTP=0 TP=8 DCP=1 docker compose up -d

# Luke NVFP4, native A4 expert activations with three-token MTP.
MOE_MODE=a4 MTP=3 TP=8 DCP=1 docker compose up -d

# Luke NVFP4 with eligible BF16 dense linears converted online to MXFP8.
MOE_MODE=a16 ONLINE_QUANT=mxfp8 MTP=0 TP=8 DCP=1 \
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
  docker compose up -d

# AMD MXFP4 experts, forced A8 path, native BF16 dense linears.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  ONLINE_QUANT=none MTP=0 TP=8 DCP=1 docker compose up -d

# The same AMD checkpoint with online MXFP8 dense linears.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  ONLINE_QUANT=mxfp8 MTP=0 TP=8 DCP=1 \
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
  docker compose up -d

# Virtual TP6/DCP3 validation profile for the AMD checkpoint.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  TP=6 DCP=3 MTP=3 MAX_NUM_SEQS=16 GRAPH=64 \
  MAX_MODEL_LEN=128000 MAX_BATCHED_TOKENS=4096 \
  GPU_MEMORY_UTILIZATION=0.95 docker compose up -d

# TP8/DCP4 full-CKV prefill profile for Luke A16, MTP off.
MOE_MODE=a16 TP=8 DCP=4 MTP=0 MAX_NUM_SEQS=32 GRAPH=128 \
  MAX_BATCHED_TOKENS=8192 docker compose up -d

# NF3 hybrid. MODEL_FAMILY selects its TP4/A16/NVFP4-KV defaults.
MODEL_FAMILY=glm52-hybrid DCP=4 MTP=3 docker compose up -d
```

For a local checkpoint, `MODEL` must use its in-container path below
`/root/models`. For another Hugging Face repository, set both `MODEL` and its
immutable `MODEL_REVISION`; the standard preset otherwise pins Luke's tested
revision `8a1f4a13204acf2b7ac840375efaed64c231c522`.

### Stable Controls

| Variable | Default and meaning |
|---|---|
| `MODEL_FAMILY` | `glm52`; use `glm52-hybrid` for the TP4 NF3 preset. The unified image also accepts `ds4`. |
| `MODEL` | Luke NVFP4 for `glm52`; the madeby561 NF3 checkpoint for `glm52-hybrid`; local paths are supported. |
| `MODEL_REVISION` | Immutable tested Hugging Face revision. Set the correct revision when changing a remote `MODEL`. |
| `SERVED_MODEL_NAME` | API model name; defaults to the selected checkpoint preset. |
| `GPUS` | Physical GPU list. Standard default is `0,1,2,3,4,5,6,7`; NF3 default is `0,1,2,3`. |
| `PORT` | `8000`. Host networking exposes it directly. |
| `TP` | Standard `8`, virtual-sharded `6`, or NF3 `4`. |
| `DCP` | Decode context parallel size. `1` disables DCP communication; validated values are topology-dependent. |
| `MTP` | `0` disables speculation. `3` is the principal validated speculative mode; the helper accepts an integer token count. |
| `MAX_NUM_SEQS` | Standard `64`; scheduler concurrency and the input to automatic graph sizing. |
| `GRAPH` | When unset, standard GLM uses `4 * MAX_NUM_SEQS`; the NF3 preset uses `64`. |
| `MAX_MODEL_LEN` | Recommended standard and NF3 default: `262144`. TP6 remains `128000`. Raise only within the KV capacity reported at startup. |
| `MAX_BATCHED_TOKENS` | Standard `8192`; NF3 `2048`. The validated virtual-TP6 profile uses `4096`. |
| `GPU_MEMORY_UTILIZATION` | Recommended TP8 and NF3 default: `0.96`; TP6 at most `0.95`. TP8 `0.98` boots but is unsafe for long-prefill runtime allocations. |
| `MOE_MODE` | `a4`, `a16`, or `force-a8-experimental`. |
| `ONLINE_QUANT` | `none`, `mxfp8`, `fp8`, `nf3-mxfp8`, or `custom`. |
| `QUANTIZATION_CONFIG_JSON` | Explicit online quantization policy; overrides the helper preset. |
| `KV_CACHE_DTYPE` | Standard `fp8`; NF3 uses `nvfp4_ds_mla`. |
| `F8_DMA` | `0`, `ag`, or `ring`; optional FP8 DCP transport experiment. It does not accelerate decode. |

Advanced A/B controls are `DCP_QUERY_SPLIT`, `DCP_CKV_GATHER`,
`DCP_TOPK_OWNER_MERGE`, `DCP_INDEXER_SHARDS`, `DCP_CKV_PREFETCH_DEPTH`,
`DCP_CKV_PREFETCH_WORKSPACE_MIB`, and `DCP_PREFILL_WORKSPACE`. Keep them on
`auto` or their defaults for published results. `B12X_PCIE_DMA=1`,
`DCP_A2A_MAX_TOKENS=64` (`16` for NF3), and
`DCP_A2A_LARGE_BACKEND=ag_rs` remain transport defaults. Backend overrides
such as `MOE_BACKEND` and `LINEAR_BACKEND` are diagnostic controls, not
separate release modes.

The 262k/0.96 standard memory pair was validated on the exact v20 image with
Luke A16, MTP3, seq=64, graph=256, batch=8,192, FP8 KV, and no online quant.
Each topology processed a 240,041-token prompt followed by 512 decode tokens:

| Topology | GMU | KV tokens | Max concurrency at 262,144 | 240k + decode |
|---|---:|---:|---:|---|
| TP8 / DCP1 | `0.96` | 603,456 | 2.30x | pass; server remained healthy |
| TP8 / DCP4 | `0.96` | 2,285,824 | 8.72x | pass; query-split/full-CKV active |

Do not raise the generic default to `0.98`. That value booted TP8/DCP1 and
reported 641,088 KV tokens, but the same 240k request OOMed when an unprofiled
Inductor buffer requested another 64 MiB with only 66.38 MiB physically free.
This is why successful startup and reported KV capacity alone are insufficient
for selecting the serving memory budget.

### Checkpoint And Quantization Modes

| Checkpoint | `QUANTIZATION` | `MOE_MODE` | Supported tested online mode |
|---|---|---|---|
| `lukealonso/GLM-5.2-NVFP4` | `modelopt_fp4` | `a4` or `a16` | `none` or `mxfp8` |
| `festr2/GLM-5.2-BF16-AMDMXFP4experts` | `mxfp4` | `force-a8-experimental` | `none`, `mxfp8`, or `fp8` |
| `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` | `nvfp4_nf3_hybrid` | `a16` | `nf3-mxfp8` |

For Luke NVFP4, A4 and A16 select the routed-expert activation path; they do
not rewrite the NVFP4 checkpoint weights. A16 uses BF16 expert activations and
is the highest-accuracy tested mode. Force-A8 selects MXFP4 expert W4A8 and
applies to the AMD checkpoint, not Luke NVFP4. Generic online MXFP8 converts
eligible BF16 dense linears and does not rewrite existing NVFP4/MXFP4 routed
expert tensors.

With `MTP>0`, the helper creates a same-checkpoint MTP draft using the same MoE
backend and probabilistic draft sampling. The target and draft share the
virtual 66-head layout at TP6. Acceptance must be read from the server log for
the exact measurement window; the client acceptance field is not the release
source of truth.

### DCP Dispatch

`auto` is a launcher decision, not a value passed into vLLM. When neither the
helper control nor its low-level runtime variable is set, the launcher resolves
the measured `TP:DCP` policy below before starting vLLM:

```text
DCP_QUERY_SPLIT  -> VLLM_DCP_QUERY_SPLIT
DCP_CKV_GATHER   -> VLLM_B12X_MLA_CKV_GATHER
DCP_TOPK_OWNER_MERGE -> VLLM_DCP_TOPK_OWNER_MERGE
DCP_INDEXER_SHARDS   -> VLLM_DCP_INDEXER_SHARDS
DCP_CKV_PREFETCH_DEPTH -> VLLM_B12X_MLA_CKV_PREFETCH_DEPTH
```

An explicit helper value bypasses the decision independently for that feature.
`DCP_INDEXER_SHARDS` and `DCP_CKV_PREFETCH_DEPTH` also accept non-negative
integers. The automatic mapping is:

| TP / DCP | Query split | Full CKV | Owner merge | Indexer shards | Prefetch depth |
|---|---:|---:|---:|---:|---:|
| TP8 / DCP1 | on | off | off | `0` | `0` |
| TP8 / DCP2 | on | on | on | `0` | `1` |
| TP8 / DCP4 | on | on | on | `2` | `1` |
| TP8 / DCP8 | on | on | on | `4` | `1` |
| TP4 / DCP1 | on | off | off | `0` | `0` |
| TP4 / DCP2, DCP4 | on | on | on | `0` | `1` |
| virtual TP6 / DCP1 | off | off | off | `0` | `0` |
| virtual TP6 / DCP2, DCP3, DCP6 | off | off | on | `0` | `0` |

`DCP_INDEXER_SHARDS=0` means the ordinary fully sharded indexer. At TP8/DCP4,
`2` creates a measured partial `2x2` topology; at TP8/DCP8, `4` creates `2x4`.
The CKV cache remains sharded by the full DCP size. The query-split flag at
DCP1 does not create inter-rank DCP traffic.

For example, this is sufficient to enable both optimizations; writing either
flag manually is unnecessary:

```bash
TP=8 DCP=4 docker compose up -d
```

To inspect the decision without loading weights:

```bash
DRY_RUN=1 TP=8 DCP=4 docker compose run --rm --no-deps glm52
# VLLM_DCP_QUERY_SPLIT=1
# VLLM_B12X_MLA_CKV_GATHER=1
# VLLM_DCP_TOPK_OWNER_MERGE=1
# VLLM_DCP_INDEXER_SHARDS=2
# VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1
```

At runtime, full-CKV use is confirmed by
`Using transient full-CKV gather for B12X sparse MLA prefill`. Query split
creates a `query_split` process group. Owner merge keeps candidate scores FP32
and final indices exact; the release policy does not use lossy peer-DMA score
transport.

Virtual TP6 pads 64 attention heads to 66, leaving 11 local heads per rank.
The aligned full-CKV kernel is not its default: the measured experimental
11-to-16 padding made TP6/DCP3 64k prefill slower. v20 instead validates and
compacts the exact pitched partial workspace returned by that topology.

For short DCP messages, the helper uses the SparkInfer PCIe A2A pool. Messages
above `DCP_A2A_MAX_TOKENS=64` use `ag_rs`. `F8_DMA=ag` or `ring` changes the
experimental PCIe payload representation; it is irrelevant to DCP1 and does
not change decode arithmetic.

### Helper-Owned Serving Contract

The embedded helper, not the Compose file, owns these release defaults:

- InstantTensor `BUFFERED`, page-cache-aware model loading;
- local-inference NCCL 2.30.4 through both `LD_PRELOAD` and
  `VLLM_NCCL_SO_PATH`;
- B12X sparse MLA, B12X MoE, B12X PCIe all-reduce, and hybrid DCP transport;
- AOT/mega-AOT, FlashInfer sampler and autotune, async scheduling, chunked
  prefill, and prefix caching;
- attention-inclusive memory profiling, CUDA-graph memory estimation, and the
  v2 model runner;
- FP8 KV for standard GLM and NVFP4 MLA KV for the NF3 preset;
- `--enable-prompt-tokens-details`, `--enable-force-include-usage`, and
  `--enable-request-id-headers`;
- GLM tool/reasoning parsers and `reasoning_effort=high`;
- the exact 78-character sparse-indexer pattern:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

Do not manually duplicate these flags in Compose. The startup gate must contain
both loader/runtime lines before benchmarking:

```text
Loading safetensors using InstantTensor loader
vLLM is using nccl==2.30.4
```

## Accuracy Reference

Lower KLD is better. The unchanged checkpoint modes retain the corrected
five-run reference campaign used by v18/v19; v20 did not rerun every unchanged
cell. The A16 online-MXFP8 release smoke and the larger ignore-pattern study
below independently remain in the same range.

| Case | Corrected-reference KLD mean +/- sample SD | Role |
|---|---:|---|
| Luke NVFP4 A4 original | `0.10228 +/- 0.00634` | Native A4 activation path. |
| Luke NVFP4 A4 online MXFP8 | `0.10800 +/- 0.00697` | Faster BF16 dense linears, with an accuracy cost. |
| Luke NVFP4 A16 original | **`0.05994 +/- 0.00129`** | Highest-accuracy tested standard mode. |
| Luke NVFP4 A16 online MXFP8 | `0.06587 +/- 0.00253` | A16 accuracy/speed balance. |
| AMD MXFP4 experts A8 original | `0.08160 +/- 0.00432` | Native BF16 dense linears. |
| AMD MXFP4 experts A8 online MXFP8 | `0.08030 +/- 0.00309` | Faster dense linears; same measured distribution. |

These values compare each served checkpoint against the same corrected BF16
reference logits. They are not directly comparable to old June logits or a
different prompt/window policy.

## Online MXFP8 Attention Precision

A 2026-07-22 factorial KLD test measured which BF16 GLM-5.2 attention
projections should be excluded from online MXFP8 conversion. Each run used the
same Luke NVFP4 snapshot, corrected BF16 reference logits, TP8/DCP1, A16,
MTP off, FP8 KV, and 2,047 teacher-forced positions. Lower KLD is better.

| MXFP8 ignore set | Runs | Mean KLD | SD between runs | VRAM delta / GPU |
|---|---:|---:|---:|---:|
| none | 10 | `0.066006794` | `0.002060655` | baseline |
| `kv_b_proj` only | 20 | `0.065398317` | `0.002308562` | about `+0.13 GiB` |
| `q_a_proj` + `kv_a_proj_with_mqa` | 20 | **`0.064174724`** | `0.001603532` | about `+1.09 GiB` |
| all three | 10 | `0.065975578` | `0.001666660` | about `+1.22 GiB` |

The old `kv_b_proj`-only exclusion has no detectable benefit versus quantizing
all eligible linears (`p=0.83`). Keeping all three projections in BF16 is also
indistinguishable from ignoring none: the mean changes by only `-0.0000312`
(`-0.05%`, `p` approximately `0.97`) while consuming about `1.22 GiB/GPU`.
Therefore the current helper source no longer excludes `kv_b_proj` by default.
The corrected launcher is
[`serve-glm52-v16.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/2993b2c02f4f00d562451105de740130d90da4a0/launchers/serve-glm52-v16.sh).

Keeping only the fused q/kv-a pair in BF16 is an optional quality experiment.
Its aggregate mean was 1.87% lower than the old `kv_b_proj`-only preset;
bootstrap P(improvement) was 97.69%, while the Welch test remained borderline
at `p=0.0599`. It costs about `1.09 GiB/GPU`, so it is not the memory-efficient
default.

The default online MXFP8 config in the updated helper source is:

```json
{"linear":{"weight":"mxfp8"}}
```

To retain the fused q/kv-a projection in BF16, set an explicit override:

```bash
ONLINE_QUANT=mxfp8 \
QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"},"ignore":["re:.*[.]q_a_proj$","re:.*[.]kv_a_proj_with_mqa$"]}' \
docker compose up
```

Both q/kv-a patterns must be supplied together because GLM-5.2 maps their
checkpoint shards into the runtime `fused_qkv_a_proj` module. Ignoring only one
creates an invalid mixed-precision fused module. Additional ignore patterns can
be appended to the same JSON array. For example, the historical `kv_b_proj`
override is `"re:.*kv_b_proj"`, although the KLD result above does not justify
using it.

The release image embeds this no-ignore default. An explicit value remains
useful when auditing a deployment or comparing alternate ignore sets:

```bash
ONLINE_QUANT=mxfp8 \
QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
docker compose up
```

Reference-logit provenance, Hugging Face artifacts, metric definitions, and
the exact corrected-reference workflow are documented on the standalone
[GLM-5.2 KLD evaluation page](../benchmarks/glm52-kld-evaluation.md). Do not
mix these results with the superseded June GLM logits.

## Validation Method

The reproducible v20 wrapper is
[`scripts/bench-glm52-v20-validation.sh`](../scripts/bench-glm52-v20-validation.sh).
It pins both the immutable tag and Docker image ID, sets the corrected no-ignore
online-MXFP8 policy, and delegates execution to the maintained v18/v19 runner.
The complete `all` campaign contains 40 configurations:

- seven TP8/DCP1/MTP0 checkpoint and online-quant cases;
- six TP8/DCP1/MTP3 cases;
- seven cases each at TP8/DCP2, DCP4, and DCP8 with MTP off;
- native and online-MXFP8 AMD cases at TP6/DCP3 and DCP6 with MTP3;
- NF3 TP4/DCP4 with MTP off and MTP3.

The runner uses two topology- and CPU-isolated slots on the 16-GPU host. TP8
uses all 16 GPUs, TP6 uses 12, and TP4 uses 8. It starts both containers and
waits for both health checks and required loader logs, then waits another 30
seconds before starting either client. The two clients run serially. No result
is accepted while another model is loading.

The following are fixed historical-comparison benchmark profiles, not the
recommended serving memory defaults. Keeping their model length and GMU fixed
avoids attributing a changed memory shape to a runtime performance change.

| Validation profile | TP | DCP | MTP | Max seqs | Graph | Batched tokens | Max model len | GMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard DCP1 | 8 | 1 | 0 or 3 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| Standard fast DCP | 8 | 2, 4, 8 | 0 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| Virtual TP6 | 6 | 3, 6 | 3 | 16 | 64 | 4,096 | 128,000 | 0.950 |
| NF3 hybrid | 4 | 4 | 0 or 3 | 8 | 64 | 3,072 | 131,072 | 0.960 |

Every configuration performs:

1. image-ID and source-mount rejection checks;
2. a short greedy coding response check that rejects empty or obviously
   corrupted output;
3. a 30-second context-zero CC1 decode run with up to 2,048 output tokens;
4. one discarded standalone 64k prefill warmup;
5. three standalone 64k prefill measurements, reported as the median;
6. mode-specific log assertions for A16/A8, online MXFP8/FP8, full-CKV DCP,
   borrowed TP6 workspace, and NF3 Grid188 execution;
7. server logs, container inspection, thermal snapshots, client JSON, and a
   per-case `summary.json` plus completion marker.

Published serving comparisons use `F8_DMA=0`. DMA `ag` and `ring` are separate
transport experiments and do not belong in the main decode table. Acceptance
statistics come from the exact post-decode server-log window. Prefill token
targeting must be recorded as either `estimate` for historical comparison or
`exact` for an exact 65,536-token prompt; never combine the two silently.

## Release Gate

All performance rows are MTP0 and were measured only after every paired model
finished loading. The final SparkInfer review commit changes only explicit
`out=` allocation and `close()` lifetime handling; after rebuilding, the pushed
image passed TP8/DCP2 boot, inference, and log checks on GPU 0-7.

### TP8 Luke NVFP4 A16

| DCP | Indexer topology | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---|---:|---:|---:|---:|
| 1 | sharded | 87.98 aggregate / 88.69 active | 6,149.9 exact | - | 559,616 |
| 2 | sharded | 73.2 | 5,866 | 5,197.2 | 1,040,128 |
| 4 | partial 2x2 | 72.7 | 5,834 | 5,106.1 | 1,984,256 |
| 8 | partial 2x4 | 67.6 | 5,741 | 5,025.5 | 3,964,928 |

DCP4 is the throughput profile. DCP8 retains 99.8% more KV capacity than DCP4
while remaining within 1.6% at 64k and 1.6% at 400k. DCP1 remains at the
established `~87-88 tok/s` decode baseline.

### Virtual TP6 AMD MXFP4 A8, online MXFP8 dense

| DCP | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---:|---:|---:|---:|
| 1 | 83.5 | - | - | 312,000 |
| 2 | 68.5 | 4,249 | 3,697.8 | 562,944 |
| 3 | 66.14 | 3,614 | 3,359.3 | 842,753 |
| 6 | 51.03 | 2,379 | 2,343.2 | 1,661,337 |

The DCP1 result matches the historical online-MXFP8 control (`83.43 tok/s`).
DCP3 improved 64k and 400k prefill by 43.5% and 39.7% over the older path;
DCP2 and DCP6 remain at their prior performance envelope.

### TP4 NF3 hybrid A16

| DCP | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---:|---:|---:|---:|
| 2 | 57.2 | 4,040 | 3,184.1 | - |
| 4 | 57.3 | 3,992 | 3,340.9 | 934,912 |

These are MTP0 numbers. The older `~104 tok/s` NF3 rows are MTP3 and must not
be compared directly to this table.

The release-gate corrected-reference KLD smoke for Luke A16 online MXFP8 was
`0.0662177` over 2,047 positions. It used the historical `kv_b_proj`-only
helper preset and is consistent with the new 20-run mean of `0.065398317`.
See [Online MXFP8 Attention Precision](#online-mxfp8-attention-precision) for
the later four-way campaign and the current recommendation.

## Xid 31 / cuBLAS Layout Fix

The old failure occurred when `_v_up_proj` consumed a strided BMM view backed
by a tightly sized DCP allocation. A guarded VMM reproduction proved that
cuBLAS can read through the next 64 KiB boundary for this shape. Normal
PyTorch allocator segments tolerate that read-ahead; a tight IPC allocation
can expose an unmapped page and produce `Xid 31 FAULT_PDE`.

The final fix does not clone every DCP output and does not reserve tail padding.
Instead, DCP producers write the same logical BHD tensor into physical
head-major storage. The transpose consumed by cuBLAS is therefore contiguous
and safe. This preserves DCP1 speed and avoids reducing KV capacity merely to
provide speculative read padding.

The exact reported production configuration was reproduced with TP8/DCP2,
MTP3, seq=16, graph=64, 8,192 batched tokens, A4, online MXFP8, ring DMA, and
FP8 KV. `expandable_segments` was deliberately disabled with:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

Two consecutive transitions passed:

| Run | Prompt tokens | Decode tokens | Server healthy | Kernel Xid/PDE |
|---|---:|---:|---:|---:|
| 1 | 300,068 | 512 | yes | none |
| 2 | 320,063 | 512 | yes | none |

The clean 2026-07-25 release tree was gated again with
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`, so expandable
segments were still disabled. A short request followed by a 301,244-token
prompt and 32 decode tokens completed; server logs and `dmesg` contained no
`Xid`, `FAULT_PDE`, or illegal-memory error. References to the "Xid gate" in
the validation logs name this regression test; they do not report a new Xid.

Reproduce the client side after the server is healthy:

```bash
python3 scripts/validate-glm52-xid31-long-prefill.py \
  --port 8000 \
  --model GLM-5.2-v20-xid31 \
  --target-tokens 300000 \
  --max-tokens 512 \
  --output xid31-run1.json
```

The helper still defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
for ordinary serving. Native KV-offload deployments that cannot use expandable
segments may override it; correctness no longer depends on allocator slack.

## TP6 Corrections

Virtual TP6 pads 64 attention heads to 66 and exposes 11 heads per rank. Two
independent contracts needed correction:

1. A partial final DCP prefill chunk retains the pitch of the full borrowed
   workspace. vLLM now accepts only that exact shape/stride and compacts it
   before projection.
2. The logical W4A8 expert width is 352, while the repacked dynamic kernel
   executes the zero-padded N128 extent of 384. SparkInfer now uses 384 for
   tile/task geometry and scratch sizing while preserving logical N=352 in the
   public execution plan.

At `MAX_NUM_SEQS=16`, graph=64, batch=4096, and GMU=0.95, TP6/DCP3 exposes
`700,449` KV tokens. This is lower than older unsafe estimates because v20 also
accounts for MRV2 graph and sparse-DCP transient memory before allocating KV.

## Reproduce The Campaign

The release wrapper is
[`scripts/bench-glm52-v20-validation.sh`](../scripts/bench-glm52-v20-validation.sh).
It pins both the image tag and local image ID and delegates to the established
resumable v18/v19 runner. Install the benchmark client at
`/root/llm-inference-bench/llm_decode_bench.py`, or set `BENCH` to its path.

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
cd rtx6kpro
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725

# Complete 40-case historical-compatible campaign. Existing completed cases
# under RESULT_ROOT are skipped only when both summary.json and complete exist.
RESULT_ROOT=/root/bench-results/glm52-v20-full-estimate \
  TOKEN_TARGETING=estimate \
  scripts/bench-glm52-v20-validation.sh all

# Exact-token TP8 DCP2/DCP4/DCP8 prefill campaign in a separate result root.
RESULT_ROOT=/root/bench-results/glm52-v20-dcp-fast-exact \
  TOKEN_TARGETING=exact \
  scripts/bench-glm52-v20-validation.sh dcp-fast
```

Individual resumable groups are also available:

```bash
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh dcp1-mtp0
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh dcp1-mtp3
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh tp6-mtp3
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh nf3

# One or more explicit cases use: "case TP DCP MTP".
TOKEN_TARGETING=exact scripts/bench-glm52-v20-validation.sh configs \
  "nvfp4-a16-orig 8 4 0" \
  "mxfp4-a8-orig 6 3 3"
```

Default checkpoint locations are the tested Luke snapshot under the Hugging
Face cache, `/root/models/GLM-5.2-BF16-AMDMXFP4experts`, and
`/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. Override `NVFP4_MODEL`,
`MXFP4_MODEL`, or `NF3_MODEL` when the same immutable checkpoints live
elsewhere. The runner defaults to GPU slots `0-7` and `8-15`, ports 8190/8191,
and CPU sets `0-31,64-95` and `32-63,96-127`; topology, ports, and CPU sets are
all explicit environment overrides.

Useful operational controls:

| Variable | Effect |
|---|---|
| `RESULT_ROOT` | Stable resumable output root. Use a different root for `estimate` and `exact`. |
| `FORCE_RERUN=1` | Ignore completion markers and rerun selected cases. |
| `KEEP_SERVERS=1` | Leave the last measured server pair running for manual inspection. |
| `SETTLE_SECONDS` | Delay after all paired servers become healthy; release default is 30. |
| `PREFILL_REPEATS` | Measured 64k repeats after warmup; release default is 3. |
| `CACHE_A`, `CACHE_B` | Persistent, isolated JIT caches for the two slots. |
| `CUDA_ALLOC_CONF` | Allocator setting passed as `PYTORCH_CUDA_ALLOC_CONF`. |

Each result root ends with aggregate `summary.json` and `summary.tsv`. Raw
case directories retain the exact command inputs, image/container inspection,
server logs, decode and prefill JSON, correctness response, thermal data, and
backend markers needed to audit an outlier.

The final validation artifacts on the release host are under:

```text
/root/bench-results/glm52-v20-final-tp6-20260725
/root/bench-results/glm52-v20-final-xid-transition-20260725
/root/bench-results/glm52-v20-final2-clean-20260725
/root/bench-results/glm52-v20-final2-xid-transition-20260725
/root/bench-results/glm52-gg-dcp-topology-matrix-20260725
/root/bench-results/glm52-gg-dcp8-indexer-2x4-20260725
```
