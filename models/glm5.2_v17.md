# GLM-5.2 v17: TP4 NVFP4/NF3 Hybrid

This page documents the July 14, 2026 TP4 release for
`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. The release adds native loading
and execution of the mixed NVFP4/NF3 expert checkpoint, NVFP4 MLA KV cache,
the B12X NF3 tile-binding correctness fix, and optimized sparse-MLA DCP
prefill paths for the validated TP4, TP6, and TP8 topologies. The same image
remains the unified GLM-5.2 and DS4 base.

The published image is a clean source build. Runtime source or wheel overlays
are not required.

## Release Image

```text
voipmonitor/vllm:fathomless-firmament-v17-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-20260714
Docker manifest: sha256:a1ec6a43cbe4192abd5597123d9270cf16c6241ebfe74066dd7c2383bb41bb27
Local image ID: sha256:988415592c05e2d3dc12cbc8ab36af8b6557221849f095ec3d5442602a02e304
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| vLLM | `local-inference-lab/vllm codex/fathomless-firmament-v17-dcp-prefill-opt-20260714` @ `6ccc3ebbd17edb05ce11b095a5b14f25839774dd` |
| vLLM base | `dev/fathomless-firmament` plus the v16 unified stack |
| vLLM changes | hybrid format [#92](https://github.com/local-inference-lab/vllm/pull/92), NVFP4 KV [#82](https://github.com/local-inference-lab/vllm/pull/82), and generalized DCP prefill [#94](https://github.com/local-inference-lab/vllm/pull/94) |
| B12X | `voipmonitor/b12x codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714` @ `1377d5f22c98de0c17d9b3f35a5b56d7587992fa` |
| B12X changes | NF3/NVFP4 work from [lukealonso/b12x #31](https://github.com/lukealonso/b12x/pull/31) plus the [preplanned-tile fix](https://github.com/MadeBy561/b12x/pull/1) |
| FlashInfer | `voipmonitor/flashinfer codex/sm120-dspark-stack-20260711` @ `801d57a08958c13d375ddbb6be3be4808f48a708` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | local-inference `2.30.4`, CUDA 13.2 |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main` @ `6d3d0aad820107fba6cb9f8589f40c01bd83c108` |

The canonical build script is
[`build-fathomless-firmament-v17-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-fathomless-firmament-v17-cu132.sh).
It clones the exact vLLM, B12X, FlashInfer, InstantTensor, DeepGEMM, and
CUTLASS commits, builds the wheel, maps PyTorch and InstantTensor to the same
local NCCL 2.30.4 library, and validates the installed source paths.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 6d3d0aad820107fba6cb9f8589f40c01bd83c108
PUSH_IMAGE=1 ./build-fathomless-firmament-v17-cu132.sh
```

## Checkpoint Layout

The tested checkpoint revision is
`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at
`68babde27a97a4c980c2494e830dd424975cd5a3`.

- Routed experts are selected per layer by the checkpoint's
  `hybrid_bit_map`: the kept tier is NVFP4 and the remaining experts use the
  checkpoint's packed NF3 format.
- Both routed-expert tiers run through the B12X W4A16 path. `A16` therefore
  describes the expert activation path; it does not rewrite the checkpoint's
  NVFP4 or NF3 weights.
- Eligible BF16 non-expert linear weights are converted once at load time to
  MXFP8. Existing NVFP4/NF3 routed experts are not requantized.
- Shared experts remain on their checkpoint path unless an explicit
  `shared_experts` online-quantization rule is supplied. The v17 preset does
  not supply one.
- `kv_b_proj` is explicitly excluded from online MXFP8 conversion.
- MLA KV cache uses `nvfp4_ds_mla`.

The helper resolves this configuration as:

```text
QUANTIZATION=nvfp4_nf3_hybrid
MOE_MODE=a16
ONLINE_QUANT=mxfp8
KV_CACHE_DTYPE=nvfp4_ds_mla
LOAD_FORMAT=instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

## Start The Server

The helper is already inside the image at
`/usr/local/bin/serve-glm52-hybrid-v17.sh`. No host-side launch script is
needed. This is the exact tested TP4/DCP4, MTP-off profile:

```bash
docker run -d --name glm52-v17-hybrid \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e GPUS=0,1,2,3 -e PORT=8000 -e DCP=4 -e MTP=0 \
  -e MAX_NUM_SEQS=8 -e GRAPH=64 \
  -e MAX_MODEL_LEN=131072 -e MAX_BATCHED_TOKENS=3072 \
  -e GPU_MEMORY_UTILIZATION=0.96 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/models:/root/models:ro \
  -v /root/.cache/vllm-glm52-v17:/cache \
  -v /root/vllm/tmp/glm52-v17:/container-tmp \
  voipmonitor/vllm:fathomless-firmament-v17-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-20260714
```

To use a local checkpoint, add:

```bash
-e MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
```

The maintained minimal Compose file is
[`examples/docker-compose-glm52-hybrid-v17.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-glm52-hybrid-v17.yml).
It exposes only the deployment envelope; the quantization, backend, loader,
and exact 78-character `index_topk_pattern` stay in the image helper.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
GPUS=0,1,2,3 PORT=8000 DCP=4 MTP=0 \
MAX_NUM_SEQS=8 GRAPH=64 MAX_MODEL_LEN=131072 \
MAX_BATCHED_TOKENS=3072 GPU_MEMORY_UTILIZATION=0.96 \
  docker compose -f examples/docker-compose-glm52-hybrid-v17.yml up -d
```

User-facing controls:

| Environment | Tested value | Meaning |
|---|---:|---|
| `GPUS` | `0,1,2,3` | Four physical GPUs used by TP4 |
| `PORT` | `8000` | OpenAI-compatible API port |
| `DCP` | `1`, `2`, or `4` | Decode-context parallel size |
| `DCP_PREFILL_WORKSPACE` | `auto` | Enables validated workspace paths by default; use `0` for the baseline path |
| `MTP` | `0` | This page validates MTP off |
| `MAX_NUM_SEQS` | `8` | Scheduler concurrency ceiling |
| `GRAPH` | `64` | Maximum CUDA graph capture size |
| `MAX_MODEL_LEN` | `131072` | Tested maximum request length |
| `MAX_BATCHED_TOKENS` | `3072` | Prefill scheduler budget and DCP4 optimization profile |
| `GPU_MEMORY_UTILIZATION` | `0.96` | Per-GPU memory target |

## Clean-Image Performance

Profile: TP4, MTP off, A16, online MXFP8 non-expert linear weights, NVFP4
KV, `MAX_NUM_SEQS=8`, graph 64, and `MAX_BATCHED_TOKENS=3072`. Three model
instances were loaded first on GPU groups 0-3, 4-7, and 8-11. Every endpoint
was ready, followed by a 30-second settle period, before clients were run
serially. No model loaded while a benchmark was active.

| DCP | KV cache tokens | Decode CC1 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---:|---:|---:|---:|---:|
| 1 | 185,216 | 49.9 | 4,469 | 4,163 |
| 2 | 384,000 | 43.9 | 3,113 | 3,031 |
| 4 | 768,000 | 44.7 | 2,378 | 2,341 |

DCP2 decode was repeated and produced 43.9 tok/s both times. An earlier DCP2
run on GPU 0-3 measured 44.9 tok/s, while the clean release run used GPU 8-11.
PR #94's runtime optimization is strictly gated out for TP4/DCP1, TP4/DCP2,
and decode, so the one-token difference is treated as GPU-group/run variance.

### DCP4 Prefill Improvement

| Metric | Current stack before workspace reuse | Final clean v17 | Change |
|---|---:|---:|---:|
| Prefill 8k | 2,172 tok/s | 2,378 tok/s | +9.5% |
| Prefill 64k | 2,144 tok/s | 2,341 tok/s | +9.2% |
| Decode CC1 | 44.9 tok/s | 44.7 tok/s | -0.4%, noise |
| KV cache | 768,000 tokens | 768,000 tokens | unchanged |

The prefill path projects each sparse-MLA partial output from 512 to 256 before
the LSE-corrected reduce-scatter. For eligible eager TP4/DCP4 prefills it also
borrows existing B12X query and scratch workspaces for gather, projection, and
caller-owned reduce-scatter output. The gate requires B12X sparse MLA, a
validated TP/DCP topology, AG/RS, non-DBO eager execution, and at least 1,025
active rows. Other shapes retain the existing path.

The image helper enables this gate automatically for TP4/DCP4, TP6/DCP2/3/6,
and TP8/DCP2/4/8. `DCP_PREFILL_WORKSPACE=0` disables it for A/B testing;
`DCP_PREFILL_WORKSPACE=1` requests it explicitly but does not bypass
source-level topology, shape, capture, and backend safety checks.

### Generalized TP6/TP8 Results

These A/B runs validate the generalized PR #94 implementation. All rows use
MTP off, `F8_DMA=0`, InstantTensor `BUFFERED`, hybrid DCP (`a2a` for small
rows and `ag_rs` for large rows), exact 8,192/65,536-token prompts, and two
runs per side. All servers were loaded before benchmarking and clients ran
serially.

TP8 used `lukealonso/GLM-5.2-NVFP4`, A16, `MAX_BATCHED_TOKENS=8192`,
`MAX_NUM_SEQS=32`, and graph 128.

| Topology | Baseline 8k | Optimized 8k | Change | Baseline 64k | Optimized 64k | Change |
|---|---:|---:|---:|---:|---:|---:|
| TP8/DCP2 | 4,476 | 4,633 | +3.51% | 4,481.5 | 4,641.5 | +3.57% |
| TP8/DCP4 | 3,312 | 3,551.5 | +7.23% | 3,328.5 | 3,576 | +7.44% |
| TP8/DCP8 | 2,150.5 | 2,378.5 | +10.60% | 2,157.5 | 2,388 | +10.68% |

TP6 used `/root/models/GLM-5.2-BF16-AMDMXFP4experts`, forced A8,
`MAX_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=16`, and graph 64.

TP6 relies on FF's automatic B12X virtual-TP layout. Before vLLM's normal
divisibility check, it pads attention heads 64 -> 66, MoE intermediate width
2048 -> 2112, and vocabulary 129280 -> 129408; checkpoint tails are
zero-filled by the loader. There is no user-facing virtual-sharding flag. A
`64 heads must be divisible by TP 6` error means this B12X configuration step
did not run, normally because the wrong image/backend was used.

| Topology | Baseline 8k | Optimized 8k | Change | Baseline 64k | Optimized 64k | Change |
|---|---:|---:|---:|---:|---:|---:|
| TP6/DCP2 | 3,912 | 3,975.5 | +1.62% | 3,912.5 | 3,966.5 | +1.38% |
| TP6/DCP3 | 3,172.5 | 3,299 | +3.99% | 3,200.5 | 3,326.5 | +3.94% |
| TP6/DCP6 | 2,119 | 2,275 | +7.36% | 2,132.5 | 2,293 | +7.53% |

The fixed-half comparison is conservative because GPUs 0-7 were faster than
GPUs 8-15. Cross-over runs on both identical GPU groups measured the intrinsic
TP8/DCP2 gain at 4.56-4.80% and TP6/DCP2 at 2.92-3.31%. Logs confirmed the
borrowed-workspace path on every optimized topology. Decode and KV capacity
are unchanged because the optimization is confined to eager prefill.

## v1.3 Investigation

The workspace idea was adapted from the fast647 implementation in
[`davidsyoung/vllm-glm52` v1.3](https://github.com/davidsyoung/vllm-glm52/tree/v1.3).
An isolated A/B on that exact v1.3 stack measured:

| v1.3 mode | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---:|---:|
| Workspace disabled | 2,414 | 2,373 |
| Workspace enabled | 2,516 | 2,474 |
| Change | +4.2% | +4.2% |

The complete v1.3 overlay was not adopted. It includes an older vLLM base,
compact-KV/loading assumptions, and deployment tuning that do not match the
Fathomless Firmament stack. Current FF already contains the useful guarded
small-row B12X A2A decode transport and paged-indexer carry-fold work. v17
ports only the reusable sparse-prefill workspace concept and keeps explicit
fallbacks for decode, CUDA graphs, non-B12X backends, small prefills, and
unsupported shapes.

## Correctness Fixes

### NF3 preplanned tile binding

NF3 weights are packed for a specific tile-N geometry. The previous custom-op
boundary could silently rebuild a preplanned `(64,256,64,256)` launch as
`(128,128,64,256)`. The launch then interpreted the packed NF3 layout with the
wrong tile geometry and could produce garbled output. B12X now carries the
planned tile K/N values through the custom op and reuses that exact geometry
during compile and cache lookup.

### B12X DCP startup consensus

The first clean source image exposed a startup deadlock after every rank had
successfully created its B12X DCP pool. The four ranks then blocked in a Gloo
error-consensus reduction even though `init_error=None` everywhere. v17 keeps
that four-byte status reduction on the same NCCL DCP exchange group used by
the immediately preceding IPC-handle exchange. B12X A2A remains enabled; no
backend, stream, graph, or runtime feature is disabled.

Validation:

- 26 focused PR #94 tests passed in the final runtime environment.
- Ruff check and format check passed.
- DCP2 and DCP4 booted and warmed B12X PCIe DCP collective signatures.
- TP4/DCP4, TP6/DCP2/3/6, and TP8/DCP2/4/8 all served exact 8k/64k requests;
  every optimized log confirmed the borrowed-workspace path was active.
- A 30,017-token DCP4 generation produced coherent Python code with zero CJK
  characters; TTFT was 8.26 seconds.
- A deterministic long-prompt baseline/optimized comparison produced exactly
  the same output.
- The running containers had only model, Hugging Face, JIT-cache, and temporary
  mounts. There were no source or site-packages overlays.

## Reproduce The Benchmark

The checked-in script verifies the immutable image ID, starts all requested
servers through the image helper, rejects source overlays, confirms
InstantTensor and B12X DCP warmup from logs, waits for every endpoint, settles,
and then measures one endpoint at a time:

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
cd rtx6kpro
MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
RESULT_ROOT=/root/bench-results/glm52-v17-hybrid-reproduction \
  ./scripts/bench-glm52-v17-hybrid-tp4.sh
```

Defaults use GPU 0-3 for DCP4, 4-7 for DCP1, and 8-11 for DCP2, matching the
published clean-image run. To reproduce only DCP4 on four GPUs:

```bash
DCP_VALUES=4 GPU_DCP4=0,1,2,3 \
RESULT_ROOT=/root/bench-results/glm52-v17-hybrid-dcp4 \
  ./scripts/bench-glm52-v17-hybrid-tp4.sh
```

Raw local validation results from the release run are under:

```text
/root/bench-results/glm52-hybrid-v17-tp4-20260714/final-clean-source
/root/bench-results/pr94-generalization-20260714
```

This v17 campaign validates serving correctness and performance. It does not
introduce a new KLD reference campaign; use the corrected BF16-reference
procedure documented on the [v15 page](glm5.2_v15.md#kld-keypoint-rerun) when
comparing checkpoint quality.
