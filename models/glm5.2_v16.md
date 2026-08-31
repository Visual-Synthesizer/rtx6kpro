# GLM-5.2 v16: Unified Fathomless Firmament Image

This page documents the July 14, 2026 GLM-5.2 release for RTX 6000 Pro
Blackwell. v16 uses one image and one source stack for GLM-5.2 and
DeepSeek-V4-Flash/DSpark. It keeps the v15 GLM launch contract and adds the
B12X TP6 W4A8 correctness fix described below.

For the complete historical TP8 DCP1/2/4/8, MTP0/MTP3, FP8-DMA, coding, and
KLD tables, see [GLM-5.2 v15](glm5.2_v15.md). The release validation requested
after the TP6 fix was intentionally limited to TP6; already valid TP8 cells
were not rerun.

## Release Image

```text
voipmonitor/vllm:fathomless-firmament-v16-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714
Docker manifest: sha256:7a0ed4f956bc2f753fd8c67d32d4ee7358e71922794a471abdb9ae6513cabc54
Local image ID: sha256:d4d4739010a71c6f424c3f7a067e3fd0fdeea72b8e49040bd8e8f167b21418a7
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| vLLM | `local-inference-lab/vllm codex/fathomless-firmament-v16-unified-20260712` @ `8f86f425102cee08745462615d54115eee275f9f` |
| vLLM base | `dev/fathomless-firmament` |
| vLLM PRs | [#88](https://github.com/local-inference-lab/vllm/pull/88), [#90](https://github.com/local-inference-lab/vllm/pull/90), [#91](https://github.com/local-inference-lab/vllm/pull/91), and [#93](https://github.com/local-inference-lab/vllm/pull/93) |
| B12X | `voipmonitor/b12x codex/fathomless-firmament-v16-integration-20260714` @ `fe06f494719267fa3b399878b67caffb915dbdc4` |
| B12X PRs | [#28](https://github.com/lukealonso/b12x/pull/28) and [#32](https://github.com/lukealonso/b12x/pull/32) |
| FlashInfer | `voipmonitor/flashinfer codex/sm120-dspark-stack-20260711` @ `801d57a08958c13d375ddbb6be3be4808f48a708` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | local-inference `2.30.4`, CUDA 13.2 |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main` @ `d104659` |

The canonical build script is
[`build-fathomless-firmament-v16-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-fathomless-firmament-v16-cu132.sh).
It pins every source commit, installs both model helpers, checks both TP2 and
TP4 DSpark memory policies, verifies the cooperative B12X W4A8 launch, and can
push the immutable tag.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout d104659
PUSH_IMAGE=1 ./build-fathomless-firmament-v16-cu132.sh
```

No source overlay or patch mount is required at runtime.

## Start GLM-5.2

The image contains `/usr/local/bin/serve-glm52-v16.sh` and the shared
`/usr/local/bin/serve-fathomless-firmament.sh` dispatcher. The maintained
minimal Compose file is
[`examples/docker-compose-glm52-v16.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-glm52-v16.yml).

Luke NVFP4, native A4, MTP off:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
GPUS=0,1,2,3,4,5,6,7 TP=8 DCP=1 MTP=0 MOE_MODE=a4 ONLINE_QUANT=none \
  docker compose -f examples/docker-compose-glm52-v16.yml up -d
```

Luke NVFP4, forced A16, online MXFP8 dense weights:

```bash
GPUS=0,1,2,3,4,5,6,7 TP=8 DCP=1 MTP=0 MOE_MODE=a16 ONLINE_QUANT=mxfp8 \
  docker compose -f examples/docker-compose-glm52-v16.yml up -d
```

AMD MXFP4 experts, forced A8, online MXFP8 dense weights:

```bash
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental ONLINE_QUANT=mxfp8 \
GPUS=0,1,2,3,4,5 TP=6 DCP=3 MTP=0 \
  docker compose -f examples/docker-compose-glm52-v16.yml up -d
```

The helper validates that `GLM52_INDEX_TOPK_PATTERN` is exactly 78 characters
and uses this default:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

### User-facing Controls

| Environment | Default | Meaning |
|---|---|---|
| `MODEL` | `lukealonso/GLM-5.2-NVFP4` | Hugging Face model or mounted checkpoint path |
| `GPUS` | `0,1,2,3,4,5,6,7` | Visible physical GPUs |
| `TP` / `DCP` | `8` / `1` | Tensor and decode-context parallel sizes |
| `MTP` | `0` | MTP draft tokens; tested values are `0` and `3` |
| `MAX_NUM_SEQS` | `64` | Scheduler concurrency |
| `GRAPH` | `4 * MAX_NUM_SEQS` | Maximum CUDA graph capture size unless explicitly set |
| `MAX_MODEL_LEN` | `131072` | Maximum sequence length |
| `MAX_BATCHED_TOKENS` | `8192` | Scheduler token budget |
| `MOE_MODE` | `a4` | `a4`, `a16`, or `force-a8-experimental` |
| `ONLINE_QUANT` | `none` | `none`, `mxfp8`, `fp8`, or `custom` |
| `F8_DMA` | `0` | FP8 PCIe DMA mode: off, `ag`, or `ring` |
| `LOAD_FORMAT` | `instanttensor` | Model loader |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Page-cache-aware InstantTensor loading |

For Luke's NVFP4 checkpoint, A4 means the checkpoint's native NVFP4/A4 expert
path. A16 forces BF16 activations through the W4A16 expert path. A8 is not a
valid Luke-NVFP4 comparison mode; the experimental A8 force is used for the AMD
MXFP4-experts checkpoint.

`ONLINE_QUANT=mxfp8` converts eligible BF16 linear weights at load time with:

```json
{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}
```

`ONLINE_QUANT=fp8` uses static block FP8 and a wider ignore list for layers
that must retain their checkpoint precision. Existing NVFP4 or MXFP4 expert
weights are not requantized by either mode.

All published speed rows use `F8_DMA=0`. FP8 DMA changes prefill transport; it
does not provide a meaningful decode optimization, so v16 does not duplicate
decode tables for `ag` and `ring`.

## TP6 W4A8 Root Cause And Fix

The failing configuration was TP6, DCP3, forced A8, during FULL CUDA-graph
capture at local `m=9`. The dynamic W4A8 kernel selected `tile_m=16` and a
376-CTA grid on a 188-SM GPU. Its software resident-grid barriers require every
CTA to be resident, but it was launched as an ordinary kernel while a BF16
shared-expert GEMM could concurrently occupy SM resources on an auxiliary
stream. Scheduled CTAs then waited for CTAs that could not become resident,
eventually producing an illegal memory access/Xid 31.

B12X PR #32 fixes the execution contract instead of disabling overlap or
changing backends. The W4A8 kernel now uses a CUDA cooperative launch and a
CuTe-safe CTA-leader predicate. No stream, backend, graph, DCP, or quantization
mode is disabled.

Validation included a targeted `m=9` graph replay concurrent with 16 BF16 GEMMs,
the exact TP6/DCP3 startup and capture, a 30k prefill/decode probe, and the full
TP6 DCP1/2/3/6 benchmark below. No new Xid was recorded.

## TP6 MTP0 Results

Settings: TP6, MTP0, FP8 DMA off, `MAX_NUM_SEQS=16`, graph 64,
`MAX_BATCHED_TOKENS=2048`, max model length 128,000. DCP1 uses
`GPU_MEMORY_UTILIZATION=0.957`; DCP2/3/6 use `0.950`. Both models in a pair
were fully ready, followed by a 30-second settle period, before either client
started. Prefill was measured serially while both models remained resident.

| Case | DCP | Decode C1 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---:|---:|---:|---:|
| AMD MXFP4 A8 original | 1 | 75.75 | 5,139 | 5,280 |
| AMD MXFP4 A8 original | 2 | 61.98 | 3,658 | 3,850 |
| AMD MXFP4 A8 original | 3 | 59.23 | 3,171 | 3,212 |
| AMD MXFP4 A8 original | 6 | 45.88 | 2,118 | 2,135 |
| AMD MXFP4 A8 online MXFP8 | 1 | 82.96 | 4,906 | 5,244 |
| AMD MXFP4 A8 online MXFP8 | 2 | 66.64 | 3,514 | 3,878 |
| AMD MXFP4 A8 online MXFP8 | 3 | 63.82 | 3,124 | 3,176 |
| AMD MXFP4 A8 online MXFP8 | 6 | 50.05 | 2,105 | 2,133 |

Every online-MXFP8 regression gate was within 1.3% of the prior valid TP6
baseline. The speed run used the same B12X commit and GLM code as the final
image. Its vLLM parent was `7ec611cf0`; final commit `8f86f4251` changes only
the DS4 CUTLASS launcher's TP-aware memory default and is not reachable from
the GLM helper.

Raw result root:

```text
/root/bench-results/glm52-v16-final-b12xfe06-20260714T004300Z
```

The checked-in benchmark script verifies the immutable image ID, launches two
servers in parallel, waits for both readiness checks, settles, and only then
starts clients:

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
cd rtx6kpro
RESULT_ROOT=/root/bench-results/glm52-v16-reproduction \
  ./scripts/bench-glm52-v16-full.sh tp6
```

## Accuracy Reference

KLD was not rerun because the fix changes only B12X launch scheduling, not
weights or arithmetic. These are the current five-run means from the corrected
BF16-reference-logit campaign documented on the v15 page:

| Checkpoint / mode | KLD mean +/- sample SD |
|---|---:|
| Luke NVFP4 A4 original | 0.10228 +/- 0.00634 |
| Luke NVFP4 A4 online MXFP8 | 0.10800 +/- 0.00697 |
| Luke NVFP4 A16 original | 0.05994 +/- 0.00129 |
| Luke NVFP4 A16 online MXFP8 | 0.06587 +/- 0.00253 |
| AMD MXFP4 experts A8 original | 0.08160 +/- 0.00432 |
| AMD MXFP4 experts A8 online MXFP8 | 0.08030 +/- 0.00309 |

Use the [v15 KLD keypoint section](glm5.2_v15.md#kld-keypoint-rerun) for the
exact fast reference-logit and scoring procedure. The separate
[Unsloth-style reproduction](glm5.2/glm52-unsloth-style-prefill-kld-2026-07-07.md)
uses a different corpus/window protocol and is not numerically interchangeable.
Do not mix the superseded logits from the earlier campaign with these values.
