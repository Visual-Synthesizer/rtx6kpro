# Kimi-K2.6 v9 DFlash on Black Benediction

Status: runbook only, no benchmark sweep yet. This page documents the 2026-06-12
Black Benediction image and the exact DFlash launch currently used from
`/root/kimi26dflash`.

The v9 change relative to v8 is not a benchmark profile change. It moves Kimi
DFlash to the Black Benediction vLLM line with PR #13, DFlash on the V2 model
runner, and a B12X W4A8 overlay image that also fixes the `dev/w4a8` decode
split-row regression.

## Current Image

```text
voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612
```

DockerHub digest:

```text
sha256:0114603cf82584e3b14b91055243bb1daf59690a87500edbdee8592117ae93a4
```

## Component Revisions

| Component | Revision |
|---|---|
| Build recipe repo | `https://github.com/local-inference-lab/blackwell-llm-docker.git` |
| Build recipe commit | `f24137e95d09366e6ad8acf682cd87ce088bb903` |
| Build Dockerfile | `Dockerfile.vllm-b12x-cu132` |
| Build launcher | `build-black-benediction-b12xpr11-cu132.sh` |
| CUDA | `13.2.1` |
| cuBLAS active runtime | `13.4.1.2-1` |
| cuDNN apt package | `9.22.0.52-1` |
| PyTorch | `2.12.0+cu132` |
| Torch bundled NCCL | `2.29.7` |
| vLLM NCCL runtime | `/opt/libnccl-local-inference.so.2.30.4` |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical`, branch `canonical/cu132-nccl2304-amd-noxml` |
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/dflash-unified-bb-20260612` |
| vLLM commit | `72661cc997e8e775b982e2fd78b456092234da43` |
| vLLM package | `0.11.2.dev279+black.benediction.pr13.b12xw4a8.cu132.20260612` |
| B12X upstream branch | `dev/w4a8` |
| B12X upstream commit | `cce0a320b7b96cd8bd9f267ffb8c9e207af695be` |
| B12X overlay branch | `w4a8-mx-serving` |
| B12X overlay commit | `95dbdd13063d1d5d575c2cf1634e99370b81145b` |
| B12X package | `0.20.0` |
| FlashInfer branch | `refs/pull/3395/head` |
| FlashInfer commit | `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| FlashInfer package | `0.6.12+cu132` |
| DeepGEMM branch | `refs/pull/324/head` |
| DeepGEMM commit | `aced12c2c8882a945c568ace9d4a7e5778aae410` |

The overlay changes only the installed Python `b12x` package in the final image.
It adds the local `w4a8_mx` e8m0_k32 serving path and restores
`_COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS` to `256`. The Kimi launch below does
not enable B12X PCIe allreduce because it was slower with the V2 runner and
FULL decode graphs.

## Build

Full image build:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout f24137e95d09366e6ad8acf682cd87ce088bb903

IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12xcce0a32-cu132-20260612 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
BUILD_BASE_IMAGE=0 \
PUSH_BASE_IMAGE=0 \
PIN_SOURCE_COMMITS=1 \
MAX_JOBS=64 \
VLLM_MAX_JOBS=64 \
NVCC_THREADS=1 \
VLLM_NVCC_THREADS=1 \
VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
VLLM_REF=codex/dflash-unified-bb-20260612 \
VLLM_COMMIT=72661cc997e8e775b982e2fd78b456092234da43 \
LAUNCHER_REPO=https://github.com/local-inference-lab/vllm.git \
LAUNCHER_REF=codex/dflash-unified-bb-20260612 \
LAUNCHER_COMMIT=72661cc997e8e775b982e2fd78b456092234da43 \
B12X_REPO=https://github.com/lukealonso/b12x.git \
B12X_REF=dev/w4a8 \
B12X_COMMIT=cce0a320b7b96cd8bd9f267ffb8c9e207af695be \
VLLM_BUILD_VERSION=0.11.2.dev279+black.benediction.pr13.b12xw4a8.cu132.20260612 \
./build-black-benediction-b12xpr11-cu132.sh
```

Overlay build for the final v9 image.

Important: `95dbdd1` was a local B12X worktree commit at build time, not an
upstream `lukealonso/b12x` commit. The exact build context used on the build
host was `/root/vllm/worktrees/b12x-w4a8-cce0a32` on branch
`w4a8-mx-serving`. To reproduce on a clean host, first push that branch or
apply the same two local changes on top of `dev/w4a8`: the `w4a8_mx`
e8m0_k32 serving path and the decode split-row un-revert to `256`.

```bash
cd /root/vllm/worktrees/b12x-w4a8-cce0a32
git checkout w4a8-mx-serving
git rev-parse HEAD
# 95dbdd13063d1d5d575c2cf1634e99370b81145b

cat >/tmp/Dockerfile.kimi-v9-b12x-overlay <<'EOF'
FROM voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12xcce0a32-cu132-20260612

COPY b12x /tmp/b12x-overlay/b12x

RUN python3 -c "import pathlib, shutil, b12x; dst = pathlib.Path(b12x.__file__).resolve().parent; shutil.rmtree(dst); shutil.copytree('/tmp/b12x-overlay/b12x', dst); print('overlaid b12x at', dst)" \
    && python3 -c "from b12x.attention.mla.compressed_config import _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS as rows; assert rows == 256, rows; print('b12x split max rows', rows)"

LABEL local-inference.b12x.branch="w4a8-mx-serving" \
      local-inference.b12x.upstream_commit="cce0a320b7b96cd8bd9f267ffb8c9e207af695be" \
      local-inference.b12x.overlay_commit="95dbdd13063d1d5d575c2cf1634e99370b81145b" \
      local-inference.b12x.overlay_note="Python overlay: w4a8_mx e8m0_k32 serving path plus MTP decode split rows un-revert"
EOF

docker build \
  -f /tmp/Dockerfile.kimi-v9-b12x-overlay \
  -t voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612 \
  .

docker push voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612
```

Verify the final image:

```bash
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612

docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool

docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
python3 - <<PY
import importlib.metadata as md
from b12x.attention.mla.compressed_config import _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS
print("vllm", md.version("vllm"))
print("b12x", md.version("b12x"))
print("flashinfer", md.version("flashinfer-python"))
print("deep_gemm", md.version("deep_gemm"))
print("split_rows", _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS)
assert _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS == 256
PY
'
```

## Target And Draft

| Item | Value |
|---|---|
| Target model | `/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58` |
| DFlash draft | `/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a` |
| Served name | `Kimi-K2.6-dflash` |
| Port | `8402` |
| GPUs in the current launch | `8,9,10,11,12,13,14,15` |
| TP | `8` |
| DCP | `1` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_ATTN` |
| KV cache | `fp8` |
| DFlash method | `dflash` |
| DFlash tokens | `7` |
| PCIe custom allreduce | disabled, `VLLM_ENABLE_PCIE_ALLREDUCE=0` |
| Observed KV cache after start | `331,792` tokens |

The image is self-contained. No vLLM or B12X bind mounts are required for this
profile.

## Docker Compose

Create the Compose file:

```bash
cat >/tmp/kimi-k26-v9-dflash.compose.yaml <<'EOF'
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612}
    container_name: ${NAME:-kimi-v2-dflash-test}
    init: true
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-dflash-pr13}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TORCH_EXTENSIONS_DIR: /cache/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      TRITON_CACHE_DIR: /cache/triton
      XDG_CACHE_HOME: /cache
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      NCCL_P2P_LEVEL: SYS
      NCCL_IB_DISABLE: "1"
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      SAFETENSORS_FAST_GPU: "1"
      TARGET: ${TARGET:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}
      DRAFT: ${DRAFT:-/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a}
      PORT: ${PORT:-8402}
    command:
      - /bin/sh
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM
        exec vllm serve "$$TARGET" \
          --served-model-name Kimi-K2.6-dflash \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --trust-remote-code \
          --tensor-parallel-size 8 \
          --decode-context-parallel-size 1 \
          --kv-cache-dtype fp8 \
          --attention-backend TRITON_MLA \
          --gpu-memory-utilization 0.92 \
          --max-model-len 262144 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 512 \
          --mm-processor-cache-gb 0 \
          --mm-encoder-tp-mode weights \
          --reasoning-parser kimi_k2 \
          --tool-call-parser kimi_k2 \
          --enable-auto-tool-choice \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --speculative-config "{\"model\":\"$$DRAFT\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"
EOF
```

Start:

```bash
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612 \
NAME=kimi-v2-dflash-test \
CACHE_ROOT=/root/.cache/vllm-kimi-k26-dflash-pr13 \
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
PORT=8402 \
docker compose -f /tmp/kimi-k26-v9-dflash.compose.yaml up -d
```

Readiness and KV cache:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-v2-dflash-test 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|DFlash|Application startup complete' | tail -n 40
```

Stop:

```bash
docker compose -f /tmp/kimi-k26-v9-dflash.compose.yaml down
```

## Command-Line Launch

This is the exact current launch from `/root/kimi26dflash`:

```bash
#!/bin/bash
set -e

TARGET=/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58
DRAFT=/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612

for i in 1 2 3 4 5; do
  docker rm -f kimi-v2-dflash-test 2>/dev/null || true
  sleep 3
  if ! docker ps -a --format '{{.Names}}' | grep -q '^kimi-v2-dflash-test$'; then break; fi
done

docker run -d \
  --name kimi-v2-dflash-test \
  --init \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-kimi-k26-dflash-pr13:/cache \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e XDG_CACHE_HOME=/cache \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  "$IMAGE" \
  /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM; exec vllm serve "$@"' \
  -- \
  "$TARGET" \
  --served-model-name Kimi-K2.6-dflash \
  --host 0.0.0.0 \
  --port 8402 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_MLA \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 512 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --speculative-config "{\"model\":\"$DRAFT\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"
```

## Runtime Notes

- The first start on a fresh cache pays the full JIT cost. Reuse
  `/root/.cache/vllm-kimi-k26-dflash-pr13` for fast restarts.
- `NCCL_GRAPH_FILE` must be unset, not set to an empty path.
- `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` is intentionally unset for this profile.
- `VLLM_USE_B12X_FP8_GEMM` is intentionally unset for this profile.
- PCIe custom allreduce is disabled because it was slower on this V2 DFlash
  profile than PYNCCL-only.
- Current observed startup state: `Application startup complete`, GPU KV cache
  `331,792` tokens, V2 DFlash speculator active.
