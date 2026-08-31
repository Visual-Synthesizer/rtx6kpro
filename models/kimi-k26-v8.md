# Kimi-K2.6 v8 on 8x RTX PRO 6000 Blackwell

Status: initial no-MTP and MTP sweeps completed on 2026-05-30. This page documents the generic
CUDA 13.2 GLM/Kimi image with the v7 vLLM/B12X branches, FlashInfer git
`b6040ed`, cuBLAS `13.4.1.2`, and the Kimi-K2.6 runtime profile inherited from
the v5/v7 runbooks.

The first v8 measurement passes are no-MTP and standard+greedy MTP sweeps on
helper host `10.229.14.14` using DCP `1,2,4,8`. No inference benchmarks for
this page were run on the local host.

## Run Kimi First: Docker Compose

This is the canonical v8 run path. Do not translate this into an ad-hoc
`docker run` command for benchmarks; keep the Compose environment, mounts, and
unset variables identical.

Prerequisites on the target host:

- 8x RTX PRO 6000 Blackwell visible as CUDA devices `0..7`.
- Docker with NVIDIA GPU runtime and Docker Compose v2.
- Kimi target model available at the path below.
- Port `8402` free, unless `PORT` is changed.

Create the Compose file:

```bash
cat >/tmp/kimi-k26-v8.compose.yaml <<'EOF'
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529}
    container_name: ${NAME:-kimi-k26-v8}
    network_mode: host
    ipc: host
    privileged: true
    security_opt:
      - label=disable
    gpus: all
    ulimits:
      memlock:
        soft: -1
        hard: -1
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v8}/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v8}/jit:/cache/jit
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v8}/triton:/root/.cache/triton
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v8}/torchinductor:/root/.cache/torchinductor
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v8}/vllm:/root/.cache/vllm
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      NVIDIA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      VLLM_USE_V2_MODEL_RUNNER: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      PORT: ${PORT:-8402}
      MODEL: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}
      SERVED_MODEL_NAME: Kimi-K2.6
      TP_SIZE: "8"
      DCP_SIZE: ${DCP_SIZE:-8}
      GPU_MEMORY_UTILIZATION: ${GPU_MEM:-0.88}
      MAX_MODEL_LEN: "262144"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "128"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "512"
      MTP: ${MTP:-1}
      SPEC_CONFIG: >-
        {"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

        spec_args=()
        if [[ "$${MTP:-0}" != "0" ]]; then
          spec_args+=(--speculative-config "$${SPEC_CONFIG}")
        fi

        exec /opt/venv/bin/vllm serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --tensor-parallel-size "$${TP_SIZE}" \
          --pipeline-parallel-size 1 \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --async-scheduling \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-model-len "$${MAX_MODEL_LEN}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --mm-processor-cache-gb 0 \
          --mm-encoder-tp-mode weights \
          --attention-backend TRITON_MLA \
          --kv-cache-dtype fp8 \
          --enable-flashinfer-autotune \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --reasoning-parser kimi_k2 \
          --tool-call-parser kimi_k2 \
          --enable-auto-tool-choice \
          "$${spec_args[@]}"
EOF
```

Start the validated v8 DCP8+MTP profile:

```bash
IMAGE=voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529 \
NAME=kimi-k26-v8-dcp8-mtp \
CACHE_ROOT=/root/.cache/vllm-kimi-k26-v8-dcp8-mtp \
PORT=8402 \
DCP_SIZE=8 \
MTP=1 \
GPU_MEM=0.88 \
docker compose -f /tmp/kimi-k26-v8.compose.yaml up -d
```

Use these exact profile overrides for comparable runs:

```bash
# DCP4 + standard/greedy Eagle3 MTP.
NAME=kimi-k26-v8-dcp4-mtp CACHE_ROOT=/root/.cache/vllm-kimi-k26-v8-dcp4-mtp \
PORT=8402 DCP_SIZE=4 MTP=1 GPU_MEM=0.90 \
docker compose -f /tmp/kimi-k26-v8.compose.yaml up -d

# DCP8 no-MTP target-only baseline.
NAME=kimi-k26-v8-dcp8-nomtp CACHE_ROOT=/root/.cache/vllm-kimi-k26-v8-dcp8-nomtp \
PORT=8402 DCP_SIZE=8 MTP=0 GPU_MEM=0.94 \
docker compose -f /tmp/kimi-k26-v8.compose.yaml up -d
```

Check readiness and KV cache:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k26-v8-dcp8-mtp 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | tail -n 5
```

Stop the instance:

```bash
docker compose -f /tmp/kimi-k26-v8.compose.yaml down
```

## Image

```bash
voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529
```

Image ID observed locally:

```text
sha256:c53f78ceb555804b229599368848dcea054de195515f4b71d4d36bd7ca63bb3f
```

Image ID after manual load on `10.229.14.14`:

```text
sha256:cc3ebcabbb3f0662c61dcd1c391583f8055313dc4915a1895c71de65ab650ee4
```

The local and remote image IDs differ, but the tag and component labels match.
Use the tag plus labels as the reproducibility check for this page.

Image metadata:

| Component | Revision |
|---|---|
| CUDA base | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` |
| CUDA | `13.2.1` |
| cuBLAS active runtime | `13.4.1.2-1` |
| cuDNN apt package | `9.22.0.52-1` |
| PyTorch | `2.12.0+cu132` |
| Torch bundled NCCL | `2.29.7` |
| vLLM NCCL runtime | `/opt/libnccl-local-inference.so.2.30.4` |
| NCCL repo | `local-inference-lab/nccl-canonical` |
| NCCL branch | `canonical/cu132-nccl2304-amd-noxml` |
| NCCL commit | `dfab7c1ace32da250ba97757879429c341b7bcf9` |
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | `codex/glm51-v6-awq-mxfp8-clean-rebase-20260528` |
| vLLM commit | `2f5db31f9bcddf8d0cdd4d52f012759f50f37875` |
| vLLM package | `0.11.2.dev278+glm51v7v4stylecu13220260529` |
| B12X branch | `codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528` |
| B12X commit | `fbb76ca3a91491c8f26a2edf729540414323e55b` |
| B12X package | `0.15.3` |
| FlashInfer repo | `flashinfer-ai/flashinfer` |
| FlashInfer branch | `main` |
| FlashInfer commit | `b6040ed2bf7b5bce387e552b52dc454673b803bc` |
| FlashInfer package | `0.6.12+cu132` |

Note: Python package metadata still reports `nvidia-cublas 13.4.0.1` from the
PyTorch wheel dependency set. The active `/usr/local/cuda/.../libcublas.so.13`
and `libcublasLt.so.13` symlinks point at the apt-installed cuBLAS `13.4.1.2`
libraries.

Verify after pulling or loading:

```bash
IMAGE=voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529

docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
docker run --rm "$IMAGE" /usr/local/bin/verify-glm-kimi-cu132
docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
python - <<PY
import importlib.metadata as md
for p in ["vllm", "torch", "flashinfer-python", "b12x"]:
    print(p, md.version(p))
PY
readlink -f /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13
readlink -f /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13
'
```

## Build

Build recipe repository:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout main
git pull --ff-only
```

The image was built from Dockerfile `Dockerfile.glm-kimi-cu132` and launcher
`build-glm-kimi-cu132.sh`. The required cuBLAS detail is that both the system
base stage and final runtime stage must relink `/usr/local/cuda` cuBLAS
libraries to the apt-installed `libcublas13-cuda-13=13.4.1.2-1` and
`libcublas13-dev-cuda-13=13.4.1.2-1` libraries. A clean build without that
Dockerfile fix can silently resolve `/usr/local/cuda/.../libcublas.so.13` to
the older `13.4.0.1` library.

Build command used for the v8 image:

```bash
IMAGE=voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529

IMAGE="$IMAGE" \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260528 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260528 \
BUILD_BASE_IMAGE=0 \
MAX_JOBS=128 VLLM_MAX_JOBS=128 NVCC_THREADS=1 VLLM_NVCC_THREADS=1 \
VLLM_REF=codex/glm51-v6-awq-mxfp8-clean-rebase-20260528 \
VLLM_COMMIT=2f5db31f9bcddf8d0cdd4d52f012759f50f37875 \
LAUNCHER_REF=codex/glm51-v6-awq-mxfp8-clean-rebase-20260528 \
LAUNCHER_COMMIT=2f5db31f9bcddf8d0cdd4d52f012759f50f37875 \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=b6040ed2bf7b5bce387e552b52dc454673b803bc \
B12X_REF=codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528 \
B12X_COMMIT=fbb76ca3a91491c8f26a2edf729540414323e55b \
CUTLASS_REF=main \
CUTLASS_COMMIT=1732ed7da3b81d9f28b0130370ba70755a7e6dda \
NCCL_REF=canonical/cu132-nccl2304-amd-noxml \
NCCL_COMMIT=dfab7c1ace32da250ba97757879429c341b7bcf9 \
VLLM_BUILD_VERSION=0.11.2.dev278+glm51v7v4stylecu13220260529 \
./build-glm-kimi-cu132.sh
```

Manual transfer used for `10.229.14.14` while DockerHub push was still running:

```bash
IMAGE=voipmonitor/vllm:blackwell-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-cublas1341-20260529
docker save "$IMAGE" | zstd -T0 -1 | ssh 10.229.14.14 'zstd -d | docker load'
```

## Target And Draft

| Item | Value |
|---|---|
| Target model | `/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58` |
| Draft model | `festr2/kimi-k2.6-eagle3-mla-fp8` |
| Served name | `Kimi-K2.6` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_MLA` |
| Target KV cache | `fp8` |
| Draft KV cache | `fp8` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| PCIe custom allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=0` |

## MTP Policy

The default Kimi speculative profile is the v5/v7 standard+greedy profile:

```json
{
  "model": "festr2/kimi-k2.6-eagle3-mla-fp8",
  "method": "eagle3",
  "num_speculative_tokens": 3,
  "draft_attention_backend": "TRITON_MLA",
  "draft_kv_cache_dtype": "fp8",
  "rejection_sample_method": "standard",
  "draft_sample_method": "greedy"
}
```

For `MTP=0`, no speculative config is passed. The first v8 sweep below uses
`MTP=0` only.

## Runtime Profiles

Use these profile values unless doing a strict A/B:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Notes |
|---|---:|---:|---:|---|
| DCP1 no-MTP | 1 | 0 | 0.94 | target-only baseline, AR off |
| DCP2 no-MTP | 2 | 0 | 0.94 | target-only baseline, AR off |
| DCP4 no-MTP | 4 | 0 | 0.94 | target-only baseline, AR off |
| DCP8 no-MTP | 8 | 0 | 0.94 | target-only baseline, AR off |
| DCP1 + MTP | 1 | 1 | 0.90 | standard+greedy speculative profile, valid |
| DCP2 + MTP | 2 | 1 | 0.90 | standard+greedy speculative profile, valid |
| DCP4 + MTP | 4 | 1 | 0.90 | standard+greedy speculative profile, valid |
| DCP8 + MTP | 8 | 1 | 0.88 | validated; `0.90` starts but requests OOM |

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` is kept enabled for the standard
runbook. Disabling it can increase allocated KV cache, but it stops accounting
CUDA graph memory during KV sizing and is therefore not the default benchmark
profile.

## Benchmark Command

The no-MTP v8 sweep was run on `10.229.14.14` from:

```text
/root/bench-results/kimi-v8-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-nomtp-20260529/
```

Each DCP value is started in a fresh container:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,32 \
  --contexts 0,128k \
  --duration 30 \
  --skip-prefill \
  --max-tokens 8192 \
  --temperature 0 \
  --kv-budget <server KV tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v8-cu132-vllm2f5db31-b12xfbb76ca-fi-b6040ed-nomtp-20260529/dcp<DCP>/mtp0/result.json
```

## No-MTP Sweep On 10.229.14.14

Measured on `10.229.14.14` with `MTP=0`, `GPU_MEM=0.94`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, and PCIe custom allreduce
disabled. `∅` means the benchmark skipped the cell because it did not fit in
the server KV cache.

| DCP | MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c32 tok/s | 128k/c1 tok/s | 128k/c32 tok/s | errors | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.94 | 491,344 | 98.5 | 1,134.4 | 50.9 | ∅ | 0 | done |
| 2 | 0 | 0.94 | 982,688 | 87.0 | 929.8 | 60.4 | ∅ | 0 | done |
| 4 | 0 | 0.94 | 1,965,376 | 83.1 | 896.1 | 61.0 | ∅ | 0 | done |
| 8 | 0 | 0.94 | 3,930,752 | 82.3 | 475.6 | 67.3 | ∅ | 0 | done |

## MTP Sweep On 10.229.14.14

Measured on `10.229.14.14` with standard+greedy Eagle3 MTP,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, and PCIe custom allreduce
disabled. DCP1/2/4 used `GPU_MEM=0.90`; DCP8 needed `GPU_MEM=0.88` to leave
enough runtime headroom. `∅` means the benchmark skipped the cell because it
did not fit in the server KV cache.

| DCP | MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | errors | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.90 | 342,672 | 143.8 | 0.351 | 1,336.6 | 0.430 | 64.2 | 0.367 | ∅ | 0 | valid |
| 2 | 1 | 0.90 | 685,344 | 105.0 | 0.417 | 980.0 | 0.415 | 57.1 | 0.412 | ∅ | 0 | valid |
| 4 | 1 | 0.90 | 1,370,688 | 86.2 | 0.326 | 633.2 | 0.418 | 50.1 | 0.450 | ∅ | 0 | valid |
| 8 | 1 | 0.88 | 2,284,544 | 125.1 | 0.476 | 623.4 | 0.420 | 78.5 | 0.444 | ∅ | 0 | valid |
| 8 | 1 | 0.90 | 2,741,376 | 0.0 | 0.000 | 0.0 | 0.000 | 0.0 | 0.000 | ∅ | 34 | request OOM; invalid throughput |

DCP8+MTP is valid at `GPU_MEM=0.88`, which gives `2,284,544` KV cache tokens.
At `GPU_MEM=0.90`, DCP8+MTP is not a `0 tok/s` performance result. The server
initialized and reported `2,741,376` KV cache tokens, but the first decode
request hit CUDA OOM inside the Marlin MoE path:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:fused_marlin_moe
output = torch.empty_like(hidden_states)
```

The failed worker reported only `53.44 MiB` free on GPU 1. The benchmark then
recorded `34` request errors across the measured cells. Treat DCP8+MTP as a
runtime/memory issue until rerun with lower memory pressure or a fixed kernel
path.
