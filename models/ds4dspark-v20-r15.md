# DeepSeek-V4-Flash-0731 DSpark on Gilded Gnosis r15

This is the current TP2 DSpark runbook for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell. It supersedes
the deployment instructions on the [v10 page](ds4dspark-v10.md), while v10
remains the historical full TP2/TP4 sweep.

The 0731 checkpoint provides the native DSpark draft module. It is not the
standard MTP2/MTP3 checkpoint. The image helper therefore keeps the two model
families explicit:

| Mode | Checkpoint | Speculation |
|---|---|---|
| `dspark` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Native DSpark, fixed K7 by default |
| `dspark-mtp0` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Disabled, target-only baseline |
| `mtp0` | `deepseek-ai/DeepSeek-V4-Flash` | Disabled |
| `mtp2` / `mtp3` | `deepseek-ai/DeepSeek-V4-Flash` | Standard MTP with 2 or 3 drafts |

Do not use `mtp2` or `mtp3` with the 0731 checkpoint.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm0bc48c5-sieec30ff-fi801d57a-cu132-20260731-r15
sha256:e5c9d250b211d240b0939b7083305478e9f1ba65c2282b294add4a592e45d282
```

The image contains `/usr/local/bin/serve-ds4-flash.sh`; users do not need to
download a separate launch script. The maintained Compose file is
[`examples/docker-compose-ds4-v20-r15.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/a8bff7b1531ab477faede2e490d8b8140bc2a316/examples/docker-compose-ds4-v20-r15.yml).

## Start The Server

This uses the same environment-driven workflow as v10:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout a8bff7b1531ab477faede2e490d8b8140bc2a316

MODE=dspark \
BACKEND=b12x-a8 \
TP_SIZE=2 \
DCP_SIZE=1 \
GPUS=0,1 \
docker compose -f examples/docker-compose-ds4-v20-r15.yml up -d
```

Check readiness and logs:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq
docker logs -f ds4-0731-r15
```

Enable load-aware physical draft depth explicitly:

```bash
DSPARK_DEPTH_MODE=dynamic GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r15.yml up -d
```

Fixed K7 remains the release default because it was faster in the current
single-user measurements. Dynamic depth is intended for workloads where low
acceptance or higher concurrency would otherwise waste draft work.

## User-Facing Controls

| Environment | Default | Purpose |
|---|---|---|
| `GPUS` | `0,1` | Visible GPU list |
| `PORT` | `8000` | OpenAI-compatible API port |
| `MODE` | `dspark` | Model and speculative mode from the table above |
| `BACKEND` | `b12x-a8` | Backend profile; v10 backend names remain supported |
| `TP_SIZE` | `2` | Tensor parallel size |
| `DCP_SIZE` | `1` | Decode context parallel size; native DSpark currently requires 1 |
| `DSPARK_TOKENS` | `7` | Maximum fixed DSpark draft depth |
| `DSPARK_DEPTH_MODE` | `fixed` | `fixed` or load-aware `dynamic` |
| `MAX_NUM_SEQS` | `16` | Scheduler concurrency |
| `MAX_MODEL_LEN` | `131072` | Validated 0731 context envelope |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Scheduler token budget |
| `GPU_MEMORY_UTILIZATION` | `0.975` | Validated TP2 memory target |
| `LOAD_FORMAT` | `instanttensor` | Model loader |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Disk plus Linux page-cache loading policy |

The helper derives the graph cap as `MAX_NUM_SEQS * (DSPARK_TOKENS + 1)`. The
default is therefore `16 * 8 = 128`; users do not need to calculate it.

Supported backend profiles are unchanged from v10:

```text
b12x-a16
b12x-a8
b12x-a8-dglin
lucifer-default
lucifer-cutlass
```

## Release Validation

The pushed image was started through the published Compose file on GPU 0-1.
The server completed InstantTensor loading, FlashInfer autotune, model memory
profiling, and FULL CUDA graph capture. The OpenAI API returned coherent
responses and all coding probes had zero CJK/garbled runs.

Configuration: TP2, DCP1, B12X A8, fixed K7, MNS16, graph 128, context 0,
InstantTensor `BUFFERED`.

| Check | Result |
|---|---:|
| Sustained CC1, run 1 | 202.72 tok/s |
| Sustained CC1, confirmation | 201.38 tok/s |
| Coding mean, 3 runs | 302.37 tok/s |
| Coding median, 3 runs | 304.48 tok/s |
| Coding maximum | 319.04 tok/s |
| Coding CJK runs | 0/3 |

The 0731 server used 80.97 GiB/GPU for model weights, 2.26 GiB for peak
activations, 0.70 GiB for non-Torch allocations, and 0.18 GiB for captured
graphs. At GMU 0.975, 7.59 GiB of KV cache was provisioned within an 8.34 GiB
target budget, so the 131,072-token launch completed without a memory override.

## v10 Regression Check

A direct 0731 K7 versus v10 K5 comparison mixes a new checkpoint with a new
draft depth. To isolate the runtime, r15 was also started with the historical
`deepseek-ai/DeepSeek-V4-Flash-DSpark` snapshot and fixed K5.

| Same historical checkpoint, TP2 B12X A8 K5 | v10 | r15 |
|---|---:|---:|
| Sustained CC1 | 219.0 | 219.99 post-warmup |
| Coding median | 294.7 | 307.05 |
| CJK/garbled coding runs | 0/5 | 0/3 |

There is no DSpark runtime speed regression in the apples-to-apples canary.
The first r15 sustained sample was 211.35 tok/s and the immediate repeat was
219.99 tok/s, which also shows why a single short speculative run is not enough
to attribute a small difference to the runtime.

The old checkpoint at the exact v10 `MAX_MODEL_LEN=262144` no longer fit with
GMU 0.95 after the current graph/memory accounting: 4.48 GiB KV was required
and 4.21 GiB was available. GMU 0.955 provided the required capacity and was
used for the speed canary. This is a capacity difference, not a decode-kernel
failure. The 0731 release default is independently validated at 131,072 and
GMU 0.975.

## Changes Included For DS4

| Component | Change | Purpose |
|---|---|---|
| vLLM | [PR #212](https://github.com/local-inference-lab/vllm/pull/212) | Expose the logical compressed-MLA payload while preserving the physical page stride |
| vLLM | [PR #213](https://github.com/local-inference-lab/vllm/pull/213) | Run FlashInfer attention autotune before KV cache initialization without invoking an invalid pre-KV attention forward |
| vLLM | [PR #214](https://github.com/local-inference-lab/vllm/pull/214) | Add the pinned 0731 DSpark/standard-MTP environment launcher and tests |
| SparkInfer | [PR #106](https://github.com/local-inference-lab/sparkinfer/pull/106) | Dispatch compressed MLA with the physical cache-page stride for exact and padded storage |

The paired vLLM/SparkInfer page-layout changes avoid a copy: callers see the
logical 584-byte payload while kernels advance through the real physical page
stride. The release also inherits the current Gilded Gnosis and SparkInfer
master commits; no source mount is needed at runtime.

All four PRs were opened ready for review, not as drafts. They were not merged
as part of publishing this image.

## Rebuild Exactly

The build repository commit is
[`a8bff7b`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/a8bff7b1531ab477faede2e490d8b8140bc2a316).
The canonical build script is
[`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/a8bff7b1531ab477faede2e490d8b8140bc2a316/build-gilded-gnosis-v20-final-cu132.sh).

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout a8bff7b1531ab477faede2e490d8b8140bc2a316

VLLM_RELEASE_COMPOSITION=reproduce-r15 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Pinned composed trees:

| Component | Commit |
|---|---|
| vLLM base | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| vLLM release tree | `0bc48c5943561c56353ce1f8047f81d5e0517237` |
| SparkInfer base | `b0976b7fd46b5d34357a5f615822b86792676feb` |
| SparkInfer release tree | `eec30ff294c1870b59a04686fff6608fddb62089` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| LMCache release tree | `a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa` |

Exact integration patches and lock files are stored under
[`patches/releases/gilded-gnosis-v20-r15`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/a8bff7b1531ab477faede2e490d8b8140bc2a316/patches/releases/gilded-gnosis-v20-r15).
