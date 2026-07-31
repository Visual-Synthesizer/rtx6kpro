# DeepSeek-V4-Flash-0731 DSpark: Gilded Gnosis r16

This page documents the unified Gilded Gnosis r16 release for
`deepseek-ai/DeepSeek-V4-Flash-0731`. It adds the maintained DSpark launcher,
fixed-K5 release profile, InstantTensor loading, and opt-in native CPU KV
offload to the same image line used by GLM-5.2.

> **Release status:** published. The exact image passed source-composition,
> build, launcher, dependency, unit-test, no-offload, and native-offload E2E
> gates.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm1e9c9c3-sieec30ff-fi801d57a-cu132-20260731-r16
Docker manifest: sha256:48518e91cf87dd0c0483c76ff86e81dfc0f46de7e364b46f7a82c481ce08188f
Local image ID: sha256:82adcb63671885fd61a8335c58d16bead5162ad1dee36e268d21707d8e8a2a15
Local size: 25,184,893,615 bytes
```

## Start DSpark K5

The launch helper is inside the image. Users only need the small Compose file;
they do not need to download or mount a separate server script.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

The release Compose defaults are:

| Setting | Default | Meaning |
|---|---:|---|
| `MODE` | `dspark` | Native DSpark serving for the 0731 checkpoint |
| `BACKEND` | `b12x-a8` | SparkInfer/B12X W4A8 target path |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | Validated DSpark topology |
| `DSPARK_DEPTH_MODE` | `fixed` | Fixed draft depth; dynamic confidence control remains opt-in |
| `DSPARK_TOKENS` | `5` | Release K5 profile |
| `MAX_NUM_SEQS` | `16` | Scheduler concurrency |
| `MAX_MODEL_LEN` | `131072` | Conservative release envelope |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill scheduler budget |
| `GPU_MEMORY_UTILIZATION` | `0.975` | GPU memory target |
| `LOAD_FORMAT` | `instanttensor` | Required default model loader |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Reuses checkpoint pages from the Linux page cache |
| `KV_OFFLOADING_SIZE` | `0` | Native CPU KV offload is disabled unless requested |

The helper computes the graph cap from `MAX_NUM_SEQS` and the selected draft
depth. The Compose file intentionally selects K5 even though the generic
launcher retains K7 as its neutral default.

## Why K5

Matched TP2 measurements favored K5 over K7 for sustained single-user decode:

| Draft depth | Sustained decode | Coding median |
|---|---:|---:|
| K5 | 217.8 tok/s | 289.4 tok/s |
| K7 | 192.1 tok/s | 281.2 tok/s |

K5 was 13.3% faster in sustained decode and did not regress the coding probe.
Community reports also found K5 less prone to very long low-acceptance runs.
Use K7 explicitly for a matched upstream experiment:

```bash
DSPARK_TOKENS=7 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

Dynamic confidence-controlled depth remains available, but is not the release
default:

```bash
DSPARK_DEPTH_MODE=dynamic GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

## Native CPU KV Offload

Native offload is independent from LMCache and is opt-in. Set the total host
capacity in GiB across all TP ranks:

```bash
KV_OFFLOADING_SIZE=48.5 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

Positive decimal and non-power-of-two capacities are supported. `0`, `0.0`, or
an unset value disables the feature. The launcher expands a positive value to:

```text
--kv-offloading-size <GiB> --kv-offloading-backend native
```

r16 uses one process-shared host offload region rather than independent
power-of-two pinned allocations. Replay retention also preserves sliding-window
attention, MTP/EAGLE tails, the configured retention interval, the latest replay
boundary, and GG shared-prefix boundaries.

The final release gate proved all of the following on the exact release image:

1. normal K5 serving with native offload disabled;
2. model startup with a decimal non-power-of-two offload size;
3. an actual store and replay hit, not only successful CLI parsing;
4. coherent output and no material decode regression against the no-offload run.

Do not enable `LMCACHE_MODE` in the same test. Native offload and LMCache are
separate cache implementations and should be qualified independently.

## Context Length

`131072` is a conservative release default, not a model limit. Community runs
reported roughly 650k with a 4096 batched-token budget and up to 1M with a 2048
budget, but those envelopes are not certified by the r16 gate. Raise
`MAX_MODEL_LEN` only together with an appropriate batched-token budget and a
real long-context test.

## Source Provenance

| Component | Ref |
|---|---|
| Canonical GG base | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| Composed vLLM tree | `1e9c9c3475fa30ab48d5639f8882f1e93bb552bf` |
| SparkInfer base | `b0976b7fd46b5d34357a5f615822b86792676feb` |
| Composed SparkInfer tree | `eec30ff294c1870b59a04686fff6608fddb62089` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| LMCache composed tree | `a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | local-inference `2.30.4`, CUDA 13.2 build |
| PyTorch / CUDA | `2.12.0+cu132` / `13.2.1` |

Release-specific vLLM changes:

| PR | Purpose |
|---|---|
| [#214](https://github.com/local-inference-lab/vllm/pull/214) | 0731 DSpark launcher and native-offload environment control |
| [#217](https://github.com/local-inference-lab/vllm/pull/217) | Shared native CPU offload region; decimal/non-power-of-two sizing |
| [#218](https://github.com/local-inference-lab/vllm/pull/218) | SWA, MTP, replay, retention-interval, and shared-prefix preservation |

The exact release archives also retain vLLM #145, #212, and #213 and
SparkInfer #106. No PR listed here is implicitly authorized for merge by its
presence in the release image.

## Rebuild

The canonical script and release manifest are:

- [`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-gilded-gnosis-v20-final-cu132.sh)
- [`manifests/vllm/gilded-gnosis-v20.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/manifests/vllm/gilded-gnosis-v20.json)
- [`examples/docker-compose-ds4-v20-r16.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-ds4-v20-r16.yml)

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

VLLM_RELEASE_COMPOSITION=reproduce-r16 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The release archive verifies the exact base commits, PR heads, composed trees,
and patch hashes before building.

## Validation

- clean source composition and immutable archive reproduction: pass;
- all build/helper shell suites: pass;
- LMCache integrated suite: 219 passed, 131 skipped;
- native offload allocation tests: 39 passed;
- native offload retention tests: 7 passed;
- DS4 launcher tests: 10 passed;
- image runtime imports, NCCL linkage, XGrammar, InstantTensor, and source
  contract gates: pass;
- no-offload TP2/DCP1 DSpark K5 model load and coherent output: pass;
- no-offload sustained CC1 decode: 220.6 tok/s;
- native offload with a decimal 5.5 GiB capacity: pass;
- repeated native-offload sustained CC1 decode: 222.9 tok/s, no material
  regression against the no-offload baseline;
- 70k/80k/100k prefix sequence: 5.22 GB GPU-to-CPU store;
- 70k replay: 635.5 MB CPU-to-GPU load and 69,888 external prefix-cache hits;
- runtime error-signature scan: pass.
