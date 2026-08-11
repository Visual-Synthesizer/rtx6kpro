# Kimi-K3 TP16 Runtime on RTX PRO 6000 Blackwell

This page specifies the qualified vLLM runtime for `moonshotai/Kimi-K3` on 16
RTX PRO 6000 Blackwell Server Edition GPUs. The target checkpoint retains its
source MXFP4 expert weights. Three server profiles are packaged in one image:
target-only decode, BF16 DSpark K7, and online-MXFP8 DFlash K7.

## Qualified Artifact

| Item | Durable identifier |
|---|---|
| Docker image | `voipmonitor/vllm:kimi-k3-hh-vllm138eccd-b12x7617005-cu132-20260811-r2` |
| Registry digest | `sha256:7ca3d4ffc6d5812984b3164e1ec821104bfa5ae85a5467aea9e86e7462943092` |
| Local image ID used for qualification | `sha256:4d8f652e2c1f268a9b14f290a72bba745825f4ea1ecbeeec45c49dbe794e1626` |
| Docker recipe | [`build/kimi-k3-hh-pr-stack-20260811`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/build/kimi-k3-hh-pr-stack-20260811) at `d97246b15744fbda49cc0405b74111156e140293` |
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Runtime topology | TP16, DCP16, A2A DCP communication |
| Target KV dtype | FP8 |
| Weight loader | InstantTensor |
| Status | Qualified |

The image is based on the immutable CUDA runtime
`voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`.
It rebuilds all vLLM native extensions from the composed Heraldic Harbinger
source instead of reusing the base image extensions.

## Source Composition

The Docker recipe applies only the changes required by the Kimi-K3 profiles.

| Repository | Clean base | Applied PR heads | Resulting Git tree |
|---|---|---|---|
| `local-inference-lab/vllm` | `dev/heraldic-harbinger@6389c45f4d172d5414526e0969e3c689096f1959` | [#242](https://github.com/local-inference-lab/vllm/pull/242) `fbd738571739e5e629da44ed40e486a8e6cdc1b8`; [#278](https://github.com/local-inference-lab/vllm/pull/278) `23d51b1953814eca0090c930c5e6cd99bc396253` | `138eccd127bb6e2b5c52940203d36b951a3c6284` |
| `local-inference-lab/b12x` | `master@ce7f62275b12ceaeee71e603fb2419bb556e166a` | [#124](https://github.com/local-inference-lab/b12x/pull/124) `5b6a53f7e83413d717c6b797eece9ce079f86954`; [#138](https://github.com/local-inference-lab/b12x/pull/138) `8596afcf10a3c48f2329ee34ce6e310ceeb6d110`; [#139](https://github.com/local-inference-lab/b12x/pull/139) `a5089aa3c6aa8cbe40dbdaf520a6c8bddf1d50e8` | `76170059b95c2189966476ddd6eb872a886882c4` |

B12X PR #141 is an integration branch containing the same runtime changes as
#124, #138, and #139. It is not applied on top of those PRs because doing so
would duplicate the integration content.

The archived integration locks and patches are under
`patches/releases/kimi-k3-hh-runtime-r1/` in the Docker recipe. Each lock pins
the clean base, PR heads, integration patch SHA-256, and resulting Git tree.

## Checkpoints

Populate the Hugging Face cache mounted at `/root/.cache/huggingface` with the
target and the draft required by the selected profile:

| Role | Checkpoint | Runtime format |
|---|---|---|
| Target | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` | Source MXFP4 experts; InstantTensor load |
| DSpark draft | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` | BF16, TP-sharded at runtime |
| DFlash draft | `modal-labs/Kimi-K3-DFlash@c192d15a43407bf758b5ae0880d5c72052fef1de` | Online MXFP8 linears; `qkv_proj` remains BF16 |

The DSpark and DFlash profiles quantize the target KDA input projections to
online MXFP8. The target-only profile does not apply that projection
quantization. None of the three profiles uses the discontinued NF3 checkpoint.

## Cache Isolation

Use a separate host directory for each `/cache/jit` mount. The vLLM, Triton,
CuTe DSL, and B12X caches use source fingerprint
`vllm138eccd127-b12x76170059b9`. Profile-level isolation prevents target-only,
DSpark, and DFlash CUDA-graph state from being mixed.

```bash
mkdir -p /mnt/luke/kimi-k3-cache/hh-runtime-nospec
mkdir -p /mnt/luke/kimi-k3-cache/hh-runtime-dspark
mkdir -p /mnt/luke/kimi-k3-cache/hh-runtime-dflash
```

The launch commands below use host networking. On the qualified host, a
systemd proxy exposes `127.0.0.1:8001` as host port `8000`, so `PORT=8001` is
required inside the container. On a host without that proxy, set `PORT=8000`
and call port 8000 directly.

## Start Target-Only Decode

```bash
docker run -d \
  --name kimi-k3-hh-nospec \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/luke/kimi-k3-cache/hh-runtime-nospec:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-nospec \
  voipmonitor/vllm:kimi-k3-hh-vllm138eccd-b12x7617005-cu132-20260811-r2
```

The served model name is `Kimi-K3-MXFP4-HH-DCP16-1M-NoDSpark`.

## Start DSpark

The image defaults to the DSpark launcher. The explicit entrypoint below makes
the selected profile visible in container metadata. The scheduler variables
are part of the qualified single-request K7 configuration.

```bash
docker run -d \
  --name kimi-k3-hh-dspark \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  -e PORT=8001 \
  -e 'DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE=[[1,1,7]]' \
  -e DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=2 \
  -e VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=2 \
  -e DSPARK_SPS_CURVE=auto \
  -e VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/luke/kimi-k3-cache/hh-runtime-dspark:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-dspark \
  voipmonitor/vllm:kimi-k3-hh-vllm138eccd-b12x7617005-cu132-20260811-r2
```

The served model name is
`Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP16-1M-KDA-MXFP8-P4096`.

## Start DFlash

```bash
docker run -d \
  --name kimi-k3-hh-dflash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/luke/kimi-k3-cache/hh-runtime-dflash:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-dflash \
  voipmonitor/vllm:kimi-k3-hh-vllm138eccd-b12x7617005-cu132-20260811-r2
```

The served model name is `Kimi-K3-MXFP4-HH-DFlash-DCP16-1M`.

## Readiness and Smoke Test

Wait for `Application startup complete` before sending a request:

```bash
docker logs -f kimi-k3-hh-dspark
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP16-1M-KDA-MXFP8-P4096",
    "prompt": "Write one sentence about speculative decoding.",
    "max_tokens": 32,
    "temperature": 0
  }'
```

## Qualified Capacity and Throughput

Measurements used 16 RTX PRO 6000 Blackwell Server Edition GPUs, TP16/DCP16,
one active request, a 256-token prompt, and up to 1,024 generated tokens. The
target-only protocol constrained generation to token ID 13 so model output did
not affect per-token timing. Speculative measurements used natural greedy
generation and report medians across six post-warmup requests.

| Profile | Reported target KV tokens | Decode tok/s median | Target cycles/s median | Draft acceptance median |
|---|---:|---:|---:|---:|
| Target-only | 1,460,937 | 56.30 | n/a | n/a |
| DSpark K7 | 1,057,049 | 107.24 | 30.22 | 0.364 |
| DFlash K7 | 1,039,043 | 93.67 | 29.57 | 0.310 |

The DSpark target KV allocation exceeds 1,048,576 tokens by 8,473 tokens. The
DFlash allocation is 9,533 tokens, or 0.91%, below that exact length; vLLM
reports 0.99 maximum concurrency for a 1,048,576-token request.

Three independent Sieve coding requests produced these generation-only rates:

| Profile | Runs (tok/s) | Median (tok/s) | CJK characters |
|---|---|---:|---:|
| DSpark K7 | 123.49, 104.01, 137.83 | 123.49 | 0 |
| DFlash K7 | 129.04, 161.33, 159.21 | 159.21 | 0 |

Coding throughput is acceptance- and output-dependent; use target cycles per
second to compare target runtime overhead independently of generated text.

InstantTensor loaded target-only weights in 163.68 seconds and completed the
target-only model load in 186.86 seconds. The complete DSpark and DFlash model
loads took approximately 203.0 and 196.34 seconds respectively on a warm host
filesystem cache.

## Build the Image

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout build/kimi-k3-hh-pr-stack-20260811
./build-kimi-k3-hh-runtime.sh
```

The builder verifies both integration patch hashes and resulting Git trees,
then verifies the corresponding image labels. To publish the resulting tag in
the same operation, set `PUSH_IMAGE=1`.

## Evidence

The machine-readable image receipt is
[`validation/kimi-k3-hh-runtime-20260811.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/build/kimi-k3-hh-pr-stack-20260811/validation/kimi-k3-hh-runtime-20260811.json).
Full local server logs, container inspections, request summaries, and generated
outputs are stored under:

```text
/mnt/luke/kimi-k3-runs/hh-release-20260811
```

The raw local evidence is machine-specific and is not required to rebuild the
image; the committed lock files and validation receipt carry the durable source
and result identities.
