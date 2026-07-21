# GLM-5.2 v20: Gilded Gnosis Safety And TP6

v20 is the tested successor to [v19](glm5.2_v19.md). It keeps the same GLM-5.2,
NF3, MXFP4, DCP, MTP, and InstantTensor launch contract while updating the
canonical GG/SparkInfer stack and fixing two release blockers:

- DCP outputs now preserve a cuBLAS-safe physical head-major layout without a
  hot-path clone or tail-padding reservation;
- virtual TP6 accepts partial pitched DCP workspaces and correctly plans the
  N128-padded W4A8-MX scratch extent.

The broad quant/DCP comparison tables remain on [v18](glm5.2_v18.md) and the
v19 DCP optimization background remains on [v19](glm5.2_v19.md). This page
records the exact v20 artifact, source composition, launch recipe, and matched
release-gate results.

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
vLLM PR #154, vLLM PR #157, and SparkInfer PR #62.

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

## Start The Server

The helper is inside the image, so users do not need to download a launch
script. This minimal Compose file exposes only deployment choices:

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
      MODEL_FAMILY: ${MODEL_FAMILY:-glm52}
      GPUS: ${GPUS:-0,1,2,3,4,5,6,7}
      PORT: ${PORT:-8000}
      MODEL: ${MODEL:-lukealonso/GLM-5.2-NVFP4}
      TP: ${TP:-8}
      DCP: ${DCP:-1}
      MTP: ${MTP:-0}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-131072}
      MAX_BATCHED_TOKENS: ${MAX_BATCHED_TOKENS:-8192}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      MOE_MODE: ${MOE_MODE:-a16}
      QUANTIZATION: ${QUANTIZATION:-modelopt_fp4}
      ONLINE_QUANT: ${ONLINE_QUANT:-none}
      F8_DMA: ${F8_DMA:-0}
    volumes:
      - ${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface
      - ${MODEL_ROOT:-/root/models}:/root/models:ro
      - ${JIT_CACHE:-./cache/glm52-v20}:/cache
      - ${CONTAINER_TMP:-./cache/glm52-v20/tmp}:/container-tmp
```

The helper supplies InstantTensor `BUFFERED`, local NCCL, FlashInfer autotune,
the exact 78-character sparse-indexer pattern, FP8 KV defaults, request usage
details, and a topology-aware CUDA graph cap. It enables query split plus
transient full-CKV gather for aligned TP4/TP8 `DCP>1` topologies and keeps the
borrowed-workspace path for virtual TP6.

Important user-facing modes:

| Checkpoint / mode | Required controls |
|---|---|
| Luke NVFP4 A4 | `MODEL_FAMILY=glm52 MOE_MODE=a4 ONLINE_QUANT=none` |
| Luke NVFP4 A16 | `MODEL_FAMILY=glm52 MOE_MODE=a16 ONLINE_QUANT=none` |
| Luke NVFP4 online MXFP8 | add `ONLINE_QUANT=mxfp8` |
| AMD MXFP4 experts A8 | `QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental` |
| AMD MXFP4 online MXFP8 | add `ONLINE_QUANT=mxfp8` |
| NF3 hybrid | `MODEL_FAMILY=glm52-hybrid`; its TP4/A16/NF3 defaults are built in |

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

The representative corrected-reference KLD smoke for Luke A16 online MXFP8
was `0.0662177` over 2,047 positions, consistent with the existing five-run
distribution on the [KLD page](../benchmarks/glm52-kld-evaluation.md). No new
full KLD campaign was required because v20 does not change the selected model
weight formats.

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

## Reproduce The Gate

The release wrapper is
[`scripts/bench-glm52-v20-validation.sh`](../scripts/bench-glm52-v20-validation.sh).
It pins both the image tag and local image ID and delegates to the established
resumable v18/v19 runner.

```bash
cd rtx6kpro

TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh dcp1-mtp0
TOKEN_TARGETING=exact scripts/bench-glm52-v20-validation.sh tp6-mtp3
```

The final validation artifacts on the release host are under:

```text
/root/bench-results/glm52-v20-final-validation-20260721
/root/kld/glm52_v20_validation_20260721
```

The runner records client JSON, server logs, correctness responses, thermal
snapshots, image identity, and backend-path markers. Do not benchmark one
instance while another model is still loading; v20 release measurements began
only after every concurrently started server reported healthy.
