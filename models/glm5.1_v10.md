# GLM-5.1 v10 NVFP4-MTP on Black Benediction PR11

Measured on 2026-06-09 on the local 16x RTX PRO 6000 Blackwell host. This page
records the Black Benediction image with B12X PR11, FlashInfer PR3395,
DeepGEMM PR324, and forced A16 MoE decode for GLM-5.1 NVFP4-MTP.

Status: DCP1/2/4/8 start with B12X MLA sparse attention, B12X NvFp4 MoE, V2
model runner, CUDA graphs, `max_num_seqs=64`, and `B12X_MOE_FORCE_A16=1`.
DCP8+MTP decode completed, but the long prefill-only benchmark stalled before
writing `prefill.json`.

## Image

```text
voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608
```

Pinned digest:

```text
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Local image ID:

```text
sha256:da1c3e883628cf4f5fcd507e8e906851f744820259393d4d1b4e13919e37f326
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS package | `13.4.1.2-1` |
| cuDNN package | `9.22.0.52-1` |
| NCCL runtime | `2.30.4`, `local-inference-lab/nccl-canonical` |
| PyTorch | `2.12.0+cu132` |
| vLLM branch | `dev/black-benediction` |
| vLLM commit | `bb6c5b7351fceb9d524e0d43b957415ffefcb981` |
| B12X branch | `refs/pull/11/head` |
| B12X commit | `d90d89c8353adabb56cc84bd3924ef811ef8d877` |
| FlashInfer branch | `refs/pull/3395/head` |
| FlashInfer commit | `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| DeepGEMM branch | `refs/pull/324/head` |
| DeepGEMM commit | `aced12c2c8882a945c568ace9d4a7e5778aae410` |

## Model

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

Served model name:

```text
GLM-5.1-NVFP4-MTP
```

## Runtime Profile

Local helper used for the measurements:

```text
/root/run-glm51-black-pr11
```

Important defaults:

| Setting | Value |
|---|---|
| GPUs | `0,1,2,3,4,5,6,7` |
| Tensor parallel | `8` |
| DCP | set by `DCP_SIZE=1|2|4|8` |
| MTP | set by `MTP=0|1` |
| MTP tokens | `3` |
| MTP draft sampling | `probabilistic` |
| Max num seqs | `64` |
| Max batched tokens | `8192` |
| CUDA graph cap | `64` no-MTP, `256` MTP |
| KV cache dtype | `fp8` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| Quantization | `modelopt_fp4` |
| GPU memory utilization | `0.96` no-MTP, `0.94` MTP |
| A16 force | `B12X_MOE_FORCE_A16=1` |
| A16 decode kernel flag | `B12X_W4A16_TC_DECODE=1` |

The MTP runs used lower GPU memory utilization because `0.96` OOMed during MTP
CUDA graph capture. The healthy startup logs include:

```text
B12X_MOE_FORCE_A16=1 forcing B12X MoE quant_mode=w4a16 for NVFP4 weights.
Using AttentionBackendEnum.B12X_MLA_SPARSE backend.
Using 'B12X' NvFp4 MoE backend
Using V2 Model Runner
```

Important: unset empty NCCL graph variables before `vllm serve`:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

Example launches:

```bash
# DCP4, MTP on, force A16.
DCP_SIZE=4 MTP=1 GPU_MEMORY_UTILIZATION=0.94 PORT=5329 /root/run-glm51-black-pr11

# DCP4, MTP off, force A16.
DCP_SIZE=4 MTP=0 GPU_MEMORY_UTILIZATION=0.96 PORT=5329 /root/run-glm51-black-pr11

# DCP8, MTP on. Starts and decodes, but long prefill benchmark stalled.
DCP_SIZE=8 MTP=1 GPU_MEMORY_UTILIZATION=0.94 PORT=5329 /root/run-glm51-black-pr11
```

## Docker Compose

This compose recipe is the direct equivalent of `/root/run-glm51-black-pr11`.
Defaults are set for the main tested configuration: DCP4, MTP on, force A16,
port `5329`, and 8 GPUs.

```yaml
services:
  glm51-v10:
    image: ${IMAGE:-voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132}
    container_name: ${NAME:-glm51-black-pr11}
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
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
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      OMP_NUM_THREADS: "16"
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      USES_B12X: "True"
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: "1"
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989}
      MTP_MODEL: ${MTP_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.1-NVFP4-MTP}
      PORT: ${PORT:-5329}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-4}
      MTP: ${MTP:-1}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.94}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      NUM_SPECULATIVE_TOKENS: ${NUM_SPECULATIVE_TOKENS:-3}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
      SPEC_CONFIG: ${SPEC_CONFIG:-}
      HF_OVERRIDES: '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
    entrypoint:
      - bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        if [ -z "$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]; then
          if [ "$${MTP}" = "1" ]; then
            MAX_CUDAGRAPH_CAPTURE_SIZE=$$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
          else
            MAX_CUDAGRAPH_CAPTURE_SIZE="$${MAX_NUM_SEQS}"
          fi
        fi
        SPEC_ARGS=()
        if [ "$${MTP}" = "1" ]; then
          SPEC_CONFIG="$${SPEC_CONFIG:-$$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}' "$${MTP_MODEL}" "$${NUM_SPECULATIVE_TOKENS}")}"
          SPEC_ARGS=(--speculative-config "$${SPEC_CONFIG}")
        fi
        cd /
        exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --tensor-parallel-size "$${TP_SIZE}" \
          --pipeline-parallel-size 1 \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --async-scheduling \
          -cc.pass_config.fuse_allreduce_rms=True \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --quantization modelopt_fp4 \
          --attention-backend B12X_MLA_SPARSE \
          --moe-backend b12x \
          --kv-cache-dtype fp8 \
          --tool-call-parser glm47 \
          --enable-auto-tool-choice \
          --reasoning-parser glm45 \
          --hf-overrides "$${HF_OVERRIDES}" \
          "$${SPEC_ARGS[@]}"
```

Useful overrides:

```bash
# DCP4, MTP on.
DCP_SIZE=4 MTP=1 GPU_MEMORY_UTILIZATION=0.94 docker compose up -d

# DCP4, MTP off.
DCP_SIZE=4 MTP=0 GPU_MEMORY_UTILIZATION=0.96 docker compose up -d

# DCP8, MTP on.
DCP_SIZE=8 MTP=1 GPU_MEMORY_UTILIZATION=0.94 docker compose up -d
```

## Startup Capacity

| DCP | MTP | GPU util | KV tokens | Max concurrency @ 202,752 | Graph memory |
|---:|:---:|---:|---:|---:|---:|
| 1 | off | 0.96 | not captured | not captured | not captured |
| 1 | on | 0.94 | 437,440 | 2.16x | 2.73 GiB |
| 2 | off | 0.96 | 990,976 | 4.89x | 0.81 GiB |
| 2 | on | 0.94 | 874,880 | 4.32x | 3.01 GiB |
| 4 | off | 0.96 | 1,981,952 | 9.78x | 0.74 GiB |
| 4 | on | 0.94 | 1,749,760 | 8.63x | 2.93 GiB |
| 8 | off | 0.96 | 3,963,904 | 19.55x | 0.75 GiB |
| 8 | on | 0.94 | 3,499,520 | 17.26x | 2.92 GiB |

## Benchmark Method

Prefill:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5329 \
  --model GLM-5.1-NVFP4-MTP \
  --prefill-only \
  --prefill-contexts 8k,32k,128k \
  --prefill-duration 10 \
  --display-mode plain \
  --output OUT/prefill.json
```

Decode:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5329 \
  --model GLM-5.1-NVFP4-MTP \
  --contexts 0 \
  --concurrency 1,2,4,8,16,32,64 \
  --duration 30 \
  --max-tokens 2048 \
  --skip-prefill \
  --kv-budget 999999999 \
  --display-mode plain \
  --output OUT/decode-2048.json
```

Results are under:

```text
/root/bench-results/glm51-black-pr11-v10/
```

## Prefill Speed

Client prompt tok/s from standalone cold prefill. DCP8+MTP did not complete
the long prefill-only run; the client was killed after about 7m50s with server
metrics stuck at one running request and no `prefill.json`.

| DCP | MTP | 8k | 32k | 128k | Notes |
|---:|:---:|---:|---:|---:|---|
| 1 | off | 4,342 | 4,384 | 3,596 |  |
| 1 | on | 4,583 | 4,353 | 3,531 |  |
| 2 | off | 3,772 | 3,900 | 3,524 |  |
| 2 | on | 3,720 | 3,831 | 3,463 |  |
| 4 | off | 3,000 | 2,813 | 2,653 |  |
| 4 | on | 3,047 | 2,767 | 2,610 |  |
| 8 | off | 2,176 | 1,849 | 1,716 |  |
| 8 | on | - | - | - | stalled during long prefill |

## Decode Speed

Aggregate decode tok/s, `ctx=0`, `max_tokens=2048`, 30s per cell.

| DCP | MTP | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | off | 74.4 | 121.9 | 209.5 | 316.0 | 492.5 | 687.8 | 887.7 |
| 1 | on | 101.4 | 162.7 | 274.0 | 451.8 | 674.2 | 1,033.6 | 1,542.0 |
| 2 | off | 59.5 | 96.7 | 165.8 | 267.1 | 417.9 | 593.7 | 795.3 |
| 2 | on | 80.0 | 139.3 | 233.2 | 358.9 | 522.3 | 786.4 | 1,184.2 |
| 4 | off | 57.6 | 91.0 | 149.8 | 235.3 | 344.2 | 461.6 | 585.7 |
| 4 | on | 76.5 | 134.8 | 220.3 | 339.4 | 498.2 | 729.1 | 1,061.1 |
| 8 | off | 52.3 | 80.5 | 127.0 | 188.1 | 256.0 | 326.0 | 386.6 |
| 8 | on | 72.8 | 127.6 | 211.7 | 317.3 | 435.3 | 601.2 | 837.7 |

## Notes

- All completed GLM runs used `B12X_MOE_FORCE_A16=1`; this is verified by
  startup logs for the automated runs.
- DCP8+MTP starts and decodes, including full CUDA graph capture size `256`.
- DCP8+MTP long prefill needs follow-up investigation. It did not OOM or crash,
  but the standalone prefill client did not finish and no prefill JSON was
  produced.
