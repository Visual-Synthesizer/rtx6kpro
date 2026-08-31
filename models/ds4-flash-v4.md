# DeepSeek-V4-Flash v4 Chthonic B12X

Updated on 2026-06-15. This page replaces the B12X runtime target from
`ds4-flash-v3.md` with the chthonic-consecration build that includes the
reasoning/tool fixes and the GLM/ModelOpt forced-W4A16 PR20 fix. The Lucifer
FlashInfer/CUTLASS section remains as the standard non-B12X comparison path.

Important: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` is a DS4 launch-time
setting in this page. Do not bake it as a global Docker image default; it broke
models without sliding-window KV groups. The requested value `4092` is rejected
by this vLLM build because the interval must be a multiple of the scheduler
block size (`256` with `--block-size=256`).

## Compared Variants

| Variant | Image | Backend summary |
|---|---|---|
| Chthonic B12X | `voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132` | `B12X_MLA_SPARSE`, `--moe-backend=b12x`, `--linear-backend=b12x` |
| Standard Lucifer Cutlass | `voipmonitor/vllm:lucifer` | `SPARSE_MLA_SM120`, `flashinfer_cutlass` MoE |

Chthonic B12X pinned digest:

```text
voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132
voipmonitor/vllm@sha256:c64b0a145de10f3a2a59584ea46aebb1bac595e6151f4aca025af3f5cb73a8f4
```

Standard Lucifer pinned digest, carried forward from v3:

```text
voipmonitor/vllm:lucifer
voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609
voipmonitor/vllm@sha256:76f5f2cb4942d5b175908192ac07be81df077fe28cd5d3f8c7c92611895e14d4
```

## Chthonic B12X Source State

| Component | Revision |
|---|---|
| Docker recipe repo | [local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) |
| Docker recipe commit | [49f799ef2f90b1dafa13054ae96ccfd0cedbe644](https://github.com/local-inference-lab/blackwell-llm-docker/commit/49f799ef2f90b1dafa13054ae96ccfd0cedbe644) |
| Docker build helper | [`build-chthonic-consecration-thinkfix-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/49f799ef2f90b1dafa13054ae96ccfd0cedbe644/build-chthonic-consecration-thinkfix-cu132.sh) |
| CUDA | `13.2.1` |
| cuBLAS package | `13.4.1.2-1` |
| cuDNN package | `9.22.0.52-1` |
| NCCL runtime | `2.30.4`, [local-inference-lab/nccl-canonical](https://github.com/local-inference-lab/nccl-canonical) |
| PyTorch | `2.12.0+cu132` |
| vLLM branch | [local-inference-lab/vllm/tree/codex/chthonic-225f431-thinkfix-pr20-20260615](https://github.com/local-inference-lab/vllm/tree/codex/chthonic-225f431-thinkfix-pr20-20260615) |
| vLLM commit | [f1190eab40a8527be1c754492ec86764336f21bb](https://github.com/local-inference-lab/vllm/commit/f1190eab40a8527be1c754492ec86764336f21bb) |
| B12X repo | [lukealonso/b12x](https://github.com/lukealonso/b12x) |
| B12X commit | [0ff2847b0c55c599c8acabb32e694ce07faa1247](https://github.com/lukealonso/b12x/commit/0ff2847b0c55c599c8acabb32e694ce07faa1247) |
| FlashInfer ref | [flashinfer-ai/flashinfer/pull/3395](https://github.com/flashinfer-ai/flashinfer/pull/3395) |
| FlashInfer commit | [b41aa8dd2fb93c49b1c6134bd1953040f8089d51](https://github.com/flashinfer-ai/flashinfer/commit/b41aa8dd2fb93c49b1c6134bd1953040f8089d51) |
| DeepGEMM ref | [deepseek-ai/DeepGEMM/pull/324](https://github.com/deepseek-ai/DeepGEMM/pull/324) |
| DeepGEMM commit | [33a715e3d9634b64a351855c74ad64e2d9359c7e](https://github.com/deepseek-ai/DeepGEMM/commit/33a715e3d9634b64a351855c74ad64e2d9359c7e) |

This vLLM branch includes:

| Fix | Purpose |
|---|---|
| procr DeepSeek V4 tool parser recovery | Prevent tool calls trapped in reasoning from breaking DS4/M3-style tool use. |
| upstream PR44297 stack | Avoid strict-tool-calling XGrammar failures around `</think>` under reasoning + spec decode. |
| Rust tool parser build | Ensures `vllm._rust_tool_parser` exists for tool-enabled requests. |
| PR20 forced-W4A16 ModelOpt NVFP4 fix | Required for GLM A16 correctness; harmless for DS4, but present in this shared image. |

Image sanity check from the built image:

```text
vllm 0.11.2.dev279+chthonic.consecration.f1190eab.b12x0ff2847.thinkfix.pr20.cu132.20260615
torch 2.12.0+cu132 13.2
flashinfer 0.6.12+cu132
rust_tool_parser /opt/venv/lib/python3.12/site-packages/vllm/_rust_tool_parser.abi3.so
PR20 markers present in installed oracle/nvfp4.py
```

## Exact Chthonic B12X Docker Build

Build from the public recipe repo:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 49f799ef2f90b1dafa13054ae96ccfd0cedbe644

IMAGE=voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-thinkfix-pr20-cu132-20260615 \
VLLM_REF=codex/chthonic-225f431-thinkfix-pr20-20260615 \
VLLM_COMMIT=f1190eab40a8527be1c754492ec86764336f21bb \
VLLM_BUILD_VERSION=0.11.2.dev279+chthonic.consecration.f1190eab.b12x0ff2847.thinkfix.pr20.cu132.20260615 \
B12X_COMMIT=0ff2847b0c55c599c8acabb32e694ce07faa1247 \
DEEPGEMM_COMMIT=33a715e3d9634b64a351855c74ad64e2d9359c7e \
./build-chthonic-consecration-thinkfix-cu132.sh
```

Publish the exact build and the community alias:

```bash
docker push voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-thinkfix-pr20-cu132-20260615
docker tag \
  voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-thinkfix-pr20-cu132-20260615 \
  voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132
docker push voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132
```

## Model

B12X local snapshot used for the historical v2/v3 measurements:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

Model ID:

```text
deepseek-ai/DeepSeek-V4-Flash
```

Served model name:

```text
DeepSeek-V4-Flash
```

## Launch Notes

Chthonic B12X defaults:

| Setting | Value |
|---|---|
| MTP tokens | `2` |
| MTP draft sampling | `probabilistic` |
| MTP local argmax reduction | `true` |
| Max num seqs | `64` |
| Max batched tokens | `4096` for speed matrix, `8192` for larger prefill/profile runs |
| CUDA graph cap | `64` no-MTP, `192` MTP |
| Max model len | `130000` for speed matrix, `262144` for large-context/profile runs |
| KV cache dtype | `fp8` |
| GPU memory utilization | `0.875` speed matrix, `0.88-0.90` larger-context/profile runs |
| DS4 chat kwargs | `thinking=true`, `reasoning_effort=high` |
| Prefix cache retention | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` |

Important: unset empty NCCL graph variables before launch:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
```

## Chthonic B12X Docker Compose

Defaults are TP4, MTP on, `max_num_seqs=64`, graph cap `192`, and the first
four visible GPUs. For no-MTP use `MTP=0` and `MAX_CUDAGRAPH_CAPTURE_SIZE=64`.
For TP2 use `TP_SIZE=2` and two visible GPUs.

```yaml
services:
  ds4-b12x:
    image: ${IMAGE:-voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132}
    container_name: ${CONTAINER_NAME:-ds4-b12x}
    network_mode: host
    gpus: all
    runtime: nvidia
    ipc: host
    shm_size: 32g
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - /mnt:/mnt
      - /cache:/cache
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/bench-results:/root/bench-results
      - /root/vllm/artifacts:/root/vllm/artifacts
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "4096"
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_MEMORY_PROFILE_INCLUDE_ATTN: "1"
      B12X_MHC_MAX_TOKENS: "16384"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_WO_PROJECTION: "1"
      VLLM_USE_B12X_MHC: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      B12X_MLA_SM120_UNIFIED: "1"
      USES_B12X: "True"
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
      PORT: ${PORT:-5329}
      TP_SIZE: ${TP_SIZE:-4}
      MTP: ${MTP:-1}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.875}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-130000}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-4096}
      LOAD_FORMAT: ${LOAD_FORMAT:-safetensors}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
      DS4_SPEC_CONFIG_JSON: '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}'
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail
      unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
      GRAPH_CAP="$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      if [ -z "$${GRAPH_CAP}" ]; then
        if [ "$${MTP:-1}" = "1" ]; then GRAPH_CAP=192; else GRAPH_CAP=64; fi
      fi
      SPEC_ARGS=()
      if [ "$${MTP:-1}" = "1" ]; then
        SPEC_ARGS=(--speculative-config "$${DS4_SPEC_CONFIG_JSON}")
      fi
      cd /
      exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$${MODEL_PATH}" \
        --served-model-name "$${SERVED_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --load-format "$${LOAD_FORMAT}" \
        --tensor-parallel-size "$${TP_SIZE}" \
        --moe-backend b12x \
        --linear-backend b12x \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "$${MAX_MODEL_LEN}" \
        --max-num-seqs "$${MAX_NUM_SEQS}" \
        --async-scheduling \
        --no-scheduler-reserve-full-isl \
        --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
        --max-cudagraph-capture-size "$${GRAPH_CAP}" \
        --attention-backend B12X_MLA_SPARSE \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --default-chat-template-kwargs.thinking=true \
        --default-chat-template-kwargs.reasoning_effort=high \
        --enable-flashinfer-autotune \
        "$${SPEC_ARGS[@]}"
```

Equivalent TP4 MTP-on launch:

```bash
docker rm -f ds4-b12x-tp4-mtp >/dev/null 2>&1 || true
docker run -d --name ds4-b12x-tp4-mtp \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/bench-results:/root/bench-results \
  -v /root/vllm/artifacts:/root/vllm/artifacts \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e B12X_MHC_MAX_TOKENS=16384 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --served-model-name DeepSeek-V4-Flash \
    --host 0.0.0.0 \
    --port 5329 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --load-format safetensors \
    --tensor-parallel-size 4 \
    --moe-backend b12x \
    --linear-backend b12x \
    --gpu-memory-utilization 0.875 \
    --max-model-len 130000 \
    --max-num-seqs 64 \
    --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --max-num-batched-tokens 4096 \
    --max-cudagraph-capture-size 192 \
    --attention-backend B12X_MLA_SPARSE \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --compilation-config="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --default-chat-template-kwargs.thinking=true \
    --default-chat-template-kwargs.reasoning_effort=high \
    --enable-flashinfer-autotune \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"b12x\",\"use_local_argmax_reduction\":true}"'
```

For no-MTP remove the final `--speculative-config ...` argument and set
`--max-cudagraph-capture-size 64`.

## Standard Lucifer Cutlass Compose

This is carried forward from v3, with the DS4 prefix-cache retention env added.

```yaml
services:
  ds4-lucifer:
    image: ${IMAGE:-voipmonitor/vllm:lucifer}
    container_name: ${CONTAINER_NAME:-ds4-lucifer}
    network_mode: host
    gpus: all
    runtime: nvidia
    ipc: host
    shm_size: 32g
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${LUCIFER_CACHE_DIR:-/root/.cache/lucifer-vllm7c6bbf4}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-12,13,14,15}
      CUTE_DSL_ARCH: sm_120a
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-1}
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "4096"
      MODEL_ID: ${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
      PORT: ${PORT:-8000}
      TP_SIZE: ${TP_SIZE:-4}
      MTP: ${MTP:-1}
      SPECULATIVE_TOKENS: ${SPECULATIVE_TOKENS:-2}
      DRAFT_SAMPLE_METHOD: ${DRAFT_SAMPLE_METHOD:-probabilistic}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-262144}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail
      unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
      unset VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND
      unset VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS
      unset VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER
      unset VLLM_CACHE_DIR

      GRAPH_CAP="$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      if [ -z "$${GRAPH_CAP}" ]; then
        if [ "$${MTP:-1}" = "1" ]; then GRAPH_CAP=192; else GRAPH_CAP=64; fi
      fi

      SPEC_ARGS=()
      if [ "$${MTP:-1}" = "1" ]; then
        SPEC_ARGS=(
          --speculative-config.method mtp
          --speculative-config.num_speculative_tokens "$${SPECULATIVE_TOKENS}"
          --speculative-config.draft_sample_method "$${DRAFT_SAMPLE_METHOD}"
        )
      fi

      exec vllm serve "$${MODEL_ID}" \
        --served-model-name "$${SERVED_MODEL_NAME}" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --load-format auto \
        --tensor-parallel-size "$${TP_SIZE}" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "$${MAX_MODEL_LEN}" \
        --max-num-seqs "$${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
        --max-cudagraph-capture-size "$${GRAPH_CAP}" \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
        --async-scheduling \
        --no-scheduler-reserve-full-isl \
        --enable-chunked-prefill \
        --enable-flashinfer-autotune \
        --enable-prefix-caching \
        --attention-backend SPARSE_MLA_SM120 \
        --kernel-config.moe_backend flashinfer_cutlass \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --default-chat-template-kwargs.thinking=true \
        --default-chat-template-kwargs.reasoning_effort=high \
        "$${SPEC_ARGS[@]}"
```

## Validation Commands

Smoke:

```bash
python3 /mnt/test.py --port 5329 -L
```

Decode matrix:

```bash
KV_BUDGET=2600000  # TP4 full C1-C64 guard; use 600000 for TP2.
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 0k \
  --duration 30 \
  --kv-budget "$KV_BUDGET" \
  --skip-prefill \
  --display-mode plain \
  --output /root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/decode.json
```

Prefill matrix:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 \
  --prefill-only \
  --prefill-contexts 8k,64k \
  --prefill-duration 30 \
  --display-mode plain \
  --output /root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/prefill.json
```

## Chthonic B12X v4 Speed

The v4 B12X speed rerun uses the chthonic image, the B12X launch recipe above,
and `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`. To avoid GPU-group bias, each
row was measured as one isolated server at a time, always starting from GPU0:
TP2 used GPUs `0,1`; TP4 used GPUs `0,1,2,3`. No explicit generation-token
override was passed; `llm_decode_bench.py v0.4.24` used its default `8192`. The manual
`--kv-budget` is only a client-side skip-guard override for this benchmark and
does not change the server configuration. The speed-matrix launches used
`MAX_MODEL_LEN=130000`; therefore the 128k prefill cell is marked `cap` because
the benchmark's 128k prompt targets about 131k tokens. Use `MAX_MODEL_LEN=262144`
for a true 128k-prefill rerun.

Decode results:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 | Accept avg |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 132.1 | 221.0 | 362.0 | 544.9 | 784.1 | 1,115.1 | 1,566.4 | 0.000 |
| TP2 | on | probabilistic | 217.8 | 353.0 | 515.4 | 728.8 | 983.3 | 1,299.4 | 1,777.8 | 0.686 |
| TP4 | off | none | 158.9 | 280.5 | 475.4 | 762.4 | 1,142.1 | 1,689.2 | 2,342.7 | 0.000 |
| TP4 | on | probabilistic | 282.4 | 471.8 | 740.8 | 1,071.8 | 1,510.7 | 2,032.7 | 2,714.9 | 0.679 |

Prefill results:

| TP | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|---:|:---:|---:|---:|---:|
| TP2 | off | 8,027 | 7,614 | cap |
| TP2 | on | 7,826 | 7,411 | cap |
| TP4 | off | 10,182 | 9,549 | cap |
| TP4 | on | 9,877 | 9,279 | cap |

Raw v4 result files:

```text
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp2-nomtp-decode.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp2-mtp-decode.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp4-nomtp-decode.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp4-mtp-decode.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp2-nomtp-prefill.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp2-mtp-prefill.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp4-nomtp-prefill.json
/root/bench-results/ds4-chthonic-v4-gpu0-rerun-20260616/tp4-mtp-prefill.json
```

## Notes

- Runtime JIT can occur on the first launch because the `/cache` volume may be
  empty. Reusing the same cache volume avoids repeating most warmup cost.
- Chthonic B12X remains the B12X comparison target. Lucifer remains the
  FlashInfer/CUTLASS comparison target.
- Keep `VLLM_USE_BREAKABLE_CUDAGRAPH=0`; earlier DS4 decode checks showed
  breakable cudagraph can disable the intended compile path and reduce decode
  speed.
