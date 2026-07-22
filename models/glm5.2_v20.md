# GLM-5.2 v20: Gilded Gnosis Safety And TP6

v20 is the tested successor to [v19](glm5.2_v19.md). It keeps the same GLM-5.2,
NF3, MXFP4, DCP, MTP, and InstantTensor launch contract while updating the
canonical GG/SparkInfer stack and fixing two release blockers:

- DCP outputs now preserve a cuBLAS-safe physical head-major layout without a
  hot-path clone or tail-padding reservation;
- virtual TP6 accepts partial pitched DCP workspaces and correctly plans the
  N128-padded W4A8-MX scratch extent.

Historical comparison data remains on [v18](glm5.2_v18.md), while the DCP
optimization background remains on [v19](glm5.2_v19.md). This page is
self-contained for building, starting, operating, and validating v20; older
pages are provenance, not required setup instructions.

Canonical source merging and the required post-merge rebuild are tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).
The image below remains the exact measured artifact until that checklist is
complete; merging a PR does not silently change the documented image.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm2167295-si6a92bcc-fi801d57a-cu132-20260721
Docker manifest: sha256:0eb4b6710200e22162ede859f49a5ef4f5ff7deadcb1ee02246d1a17d2325877
Local image ID: sha256:710e15143a31e97ded855612237bd211ad8f73f6f7a06f300eea02a1326beea4
```

`si` identifies SparkInfer, the renamed B12X project. Legacy B12X environment
variable names remain accepted for compatibility.

Pinned source stack:

| Component | Ref / commit |
|---|---|
| Canonical GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `b07bef75b6ef964b00f6278432d3d5973fee06de` |
| vLLM release source | `voipmonitor/vllm build/gilded-gnosis-v20-final-candidate-20260721` @ `2167295cd3e133d38ab22a67a42b0004db65d0a6` |
| SparkInfer base | `local-inference-lab/sparkinfer master` @ `c0a464f5e4173e26822e3c6e10059b6f3be0f7eb` |
| SparkInfer release source | `local-inference-lab/sparkinfer build/sparkinfer-v20-final-candidate-20260721` @ `6a92bcc0f2bf03b13dd03dbc7ce97e26133c580e` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| CUTLASS C++ / DSL | `e6233cbac5d7c7a865c19c91cd684ceece19513c` / `4.6.0` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA / cuDNN | `2.12.0+cu132` / `13.2.1` / `9.22.0.52` |
| Build repository | `local-inference-lab/blackwell-llm-docker` @ `60b65fd0b2dfd82b43a8559e1bbec84ef1742906` |

The image contains no `VLLM_PATCH_URL`, `VLLM_PATCH_FILE`, source bind mount,
or private source overlay. Image labels expose every source pin and a cache
fingerprint derived from the vLLM and SparkInfer commits.

## Build It Exactly

The canonical build entry point is
[`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/60b65fd0b2dfd82b43a8559e1bbec84ef1742906/build-gilded-gnosis-v20-final-cu132.sh).
It builds with the exact commits above, validates runtime symbols and source
contracts, verifies the image labels, and only then allows an optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 60b65fd0b2dfd82b43a8559e1bbec84ef1742906
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
| vLLM | [#149](https://github.com/local-inference-lab/vllm/pull/149) | Isolate target/draft graph resources and account for MRV2 graph memory. |
| vLLM | [#150](https://github.com/local-inference-lab/vllm/pull/150) | Allow shared-expert overlap only before resident grids make it unsafe. |
| vLLM | [#153](https://github.com/local-inference-lab/vllm/pull/153) | Defer KV-offload stores until request metadata is ready. |
| vLLM | [#155](https://github.com/local-inference-lab/vllm/pull/155) | Cap same-model draft positions at the model context limit. |
| vLLM | [#156](https://github.com/local-inference-lab/vllm/pull/156) | Preallocate absorbed MLA projections before dequant scratch. |
| vLLM | [#162](https://github.com/local-inference-lab/vllm/pull/162) | Accept the exact partial pitched TP6 DCP workspace layout. |
| SparkInfer | [#57](https://github.com/local-inference-lab/sparkinfer/pull/57) | Package PCIe CUDA sources needed by runtime compilation. |
| SparkInfer | [#59](https://github.com/local-inference-lab/sparkinfer/pull/59) | Isolate CUDA-graph PCIe channel lifecycle. |
| SparkInfer | [#60](https://github.com/local-inference-lab/sparkinfer/pull/60) | Expose MoE plans that safely permit shared-expert overlap. |
| SparkInfer | [#63](https://github.com/local-inference-lab/sparkinfer/pull/63) | Separate identity and dynamic MLA latent-scale compile keys. |
| SparkInfer | [#48](https://github.com/local-inference-lab/sparkinfer/pull/48) | Plan W4A8 scratch from the physical N128-padded TP shard. |

The release build itself does not merge PRs. The exact release branches remain
the integration points for reproducing the tested artifact regardless of each
review's later merge state.

### Canonical Merge Status

[Issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33) is the
authoritative ordered checklist for landing this stack in canonical GG and
SparkInfer. As of 2026-07-22, every open runtime PR listed above is non-draft,
mergeable against its canonical base, and has no unresolved current review
thread. SparkInfer must land #59, #60, and #48 before the paired vLLM changes
#149, #150, and #162 are used together in a rebuilt image.

The review cleanup after the image was pushed is intentionally explicit:

- vLLM #149 gained two runtime corrections in `c1b446a121`: projected AG_RS
  fallback memory is included in profiling, and a lazily created breakable
  CUDA graph inherits the disposable profiling pool;
- SparkInfer #48 gained only a tighter scratch-allocation regression assertion
  in `73de085078`;
- vLLM #156 was rebased onto current GG as `b7710b5ad1` without changing its
  intended materialized-MLA fallback behavior;
- [SparkInfer #68](https://github.com/local-inference-lab/sparkinfer/pull/68)
  fixes Python 3.10 compatibility in the packaging test and does not affect the
  runtime image.

Consequently, the existing image is the reproducible performance artifact, but
the canonical-head image must be rebuilt and pass the release gate after the
merge checklist completes. The sparse-CKV PRs remain available for future work
but are not v20 release dependencies or launcher defaults.

## Start The Server

The helper is inside the image, so users do not need to download a launch
script. Docker with NVIDIA Container Toolkit, host IPC, and at least four
Blackwell GPUs is required. Pull the immutable image first:

```bash
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm2167295-si6a92bcc-fi801d57a-cu132-20260721
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
    image: voipmonitor/vllm:gilded-gnosis-v20-vllm2167295-si6a92bcc-fi801d57a-cu132-20260721
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

The immutable image was measured with the older internal fallback of
`MAX_MODEL_LEN=131072` and `GPU_MEMORY_UTILIZATION=0.90`. The Compose contract
above intentionally overrides those two fallback values. A direct `docker run`
must likewise pass `MAX_MODEL_LEN=262144` and `GPU_MEMORY_UTILIZATION=0.96`;
future builds use these topology-aware values directly through
[`serve-glm52-v16.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/02411928c62260300d79c45c0f280851db2219b6/launchers/serve-glm52-v16.sh).

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

Advanced A/B controls are `DCP_QUERY_SPLIT`, `DCP_CKV_GATHER`, and
`DCP_PREFILL_WORKSPACE` (`auto`, `0`, or `1`), plus `B12X_PCIE_DMA` (`1` by
default), `DCP_A2A_MAX_TOKENS` (`64` for standard GLM and `16` for NF3), and
`DCP_A2A_LARGE_BACKEND` (`ag_rs`). Keep them on `auto` or their defaults for
published results. Backend overrides such as `MOE_BACKEND` and
`LINEAR_BACKEND` are diagnostic controls, not separate release modes.

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
helper control nor its low-level runtime variable is set, the launcher starts
with `auto`, resolves the `TP:DCP` pair below, and exports concrete `0` or `1`
values before starting vLLM:

```text
DCP_QUERY_SPLIT  -> VLLM_DCP_QUERY_SPLIT
DCP_CKV_GATHER   -> VLLM_B12X_MLA_CKV_GATHER
```

An explicit helper value of `0` or `1` bypasses this decision independently
for each feature. The automatic mapping is:

| TP / DCP | Query split | Full-CKV gather | Prefill path |
|---|---:|---:|---|
| TP8 / DCP1 | off | off | Ordinary non-collective DCP1 path. |
| TP8 / DCP2, DCP4, DCP8 | on | on | Local query heads plus transient full-CKV gather. |
| TP4 / DCP1 | off | off | Ordinary non-collective DCP1 path. |
| TP4 / DCP2, DCP4 | on | on | Local query heads plus transient full-CKV gather. |
| virtual TP6 / DCP1 | off | off | Ordinary DCP1 path. |
| virtual TP6 / DCP2, DCP3, DCP6 | off | off | Project-before-merge borrowed workspace. |

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
```

At runtime, full-CKV use is confirmed by
`Using transient full-CKV gather for B12X sparse MLA prefill`. Query split
creates a `query_split` process group. Both features optimize prefill only.

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

The pushed v20 image is immutable and predates this measurement. Its embedded
helper still selects the historical `kv_b_proj`-only config when
`ONLINE_QUANT=mxfp8` is used without `QUANTIZATION_CONFIG_JSON`. To obtain the
new no-ignore behavior with that exact image, pass this explicitly:

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

All values below were measured from the exact pushed image. Models finished
loading before any client started.

| Case | Topology | MTP | Decode CC1 | Prefill 64k | Acceptance | KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| Luke NVFP4 A16 original | TP8 / DCP1 | 0 | 87.18 | - | - | - |
| Luke NVFP4 A16 original | TP8 / DCP4 | 0 | - | 5,343 | - | - |
| AMD MXFP4 experts A8 original | TP6 / DCP3 | 0 | 57.27 | - | - | 700,449 |
| NF3 hybrid A16 | TP4 / DCP4 | 3 | 104.52 | 3,532 | 0.610 | - |

Matched interpretation:

- DCP1 decode matches the established `~87 tok/s` baseline.
- TP8/DCP4 prefill matches the optimized v19 runs (`5,338-5,361 tok/s`).
- TP6/DCP3 matches the same-GPU v20 candidate (`57.3 tok/s`) and the current
  v19 control (`56.2 tok/s`); the historical single number is not a stable
  MTP-independent baseline.
- NF3 `3,532 tok/s` matches optimized v19 (`3,545-3,552 tok/s`). The older v18
  value of `2,292 tok/s` predates the DCP prefill optimization. An earlier v20
  run at `3,624 tok/s` was a favorable run, not a separate v20 speedup.

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
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm2167295-si6a92bcc-fi801d57a-cu132-20260721

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
/root/bench-results/glm52-v20-final-validation-20260721
/root/kld/glm52_v20_validation_20260721
```
