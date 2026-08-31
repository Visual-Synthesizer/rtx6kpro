# DeepSeek-V4-Flash v2 B12X PR11 vs Lucifer Cutlass

Measured on 2026-06-08 and 2026-06-09 on the local 16x RTX PRO 6000 Blackwell
host. This page records the Black Benediction B12X PR11 image and compares it
with the Lucifer Cutlass image for DeepSeek-V4-Flash quality profiles,
prefill, and decode.

Important: the Lucifer variant below is a Cutlass/FlashInfer variant. It uses
`--attention-backend SPARSE_MLA_SM120` and
`--kernel-config.moe_backend flashinfer_cutlass`; it is not the B12X MoE
backend.

## Compared Variants

| Variant | Image | Backend summary |
|---|---|---|
| B12X PR11 | `voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132` | `B12X_MLA_SPARSE`, `--moe-backend=b12x`, `--linear-backend=b12x` |
| Lucifer Cutlass | `voipmonitor/dsv4-flash:lucifer-mxfp4-cutlass-20260603` | `SPARSE_MLA_SM120`, `flashinfer_cutlass` MoE |

B12X pinned digest:

```text
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Lucifer pinned digest:

```text
voipmonitor/dsv4-flash@sha256:71341a1a3fe8cba8283b2289d49c03023008b90426af51d86cba958e0684d385
```

Relevant B12X source state:

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

Local snapshot used for the B12X measurements:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

Served model names:

```text
DeepSeek-V4-Flash
deepseek-v4-flash
```

## Launch Notes

B12X helper:

```bash
# TP4 with MTP2.
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 TP_SIZE=4 MTP=1 PORT=5329 /root/run-ds4-flash-black-pr11

# TP4 without MTP.
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 TP_SIZE=4 MTP=0 PORT=5329 /root/run-ds4-flash-black-pr11

# TP2 with MTP2.
CUDA_VISIBLE_DEVICES_VALUE=0,1 TP_SIZE=2 MTP=1 PORT=5329 /root/run-ds4-flash-black-pr11
```

B12X important defaults:

| Setting | Value |
|---|---|
| MTP tokens | `2` |
| MTP draft sampling | `probabilistic`; greedy measured separately |
| MTP local argmax reduction | `true` |
| Max num seqs | `64` |
| Max batched tokens | `4096` for original speed matrix, `8192` for 2026-06-09 reruns |
| CUDA graph cap | `64` no-MTP, `192` MTP |
| Max model len | `130000` original speed matrix, `262144` profile/prefill/greedy reruns |
| KV cache dtype | `fp8` |
| GPU memory utilization | `0.875` speed matrix, `0.88` prefill/greedy reruns, `0.90` profile farm |
| DS4 chat kwargs for quality farm | `{"thinking": true, "reasoning_effort": "high"}` |

Lucifer speed sweep launch shape:

```bash
docker run --rm --name ds4-lucifer-speed \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/luci-official:/cache \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_IB_DISABLE=1 \
  voipmonitor/dsv4-flash:lucifer-mxfp4-cutlass-20260603 serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 5610 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format auto \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.86 \
  --max-model-len 393216 \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 192 \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --max-num-batched-tokens 8192 \
  --attention-backend SPARSE_MLA_SM120 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-flashinfer-autotune \
  --kernel-config.moe_backend flashinfer_cutlass \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "high"}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.draft_sample_method greedy
```

Lucifer MTP measurements used the default vLLM draft sampling behavior. The
launch logs show only `{'method': 'mtp', 'num_speculative_tokens': 2}` in
`speculative_config`; in this vLLM branch the default `draft_sample_method` is
`greedy`. The compose below sets it explicitly to keep the recipe
reproducible.

Lucifer MTP probabilistic was remeasured by changing only
`--speculative-config.draft_sample_method probabilistic`. The probabilistic
rerun confirmed the setting in the server log and is listed separately in the
speed tables.

Lucifer TP2 MTP speed sweep used `--max-model-len 245760`, because
`393216` did not fit with MTP graph capture at `gpu_memory_utilization=0.88`.
The measured prefill contexts still include 128k, so this does not affect the
headline prefill/decode cells below.

Important: unset empty NCCL graph variables before any B12X `vllm serve`:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

## Docker Compose

B12X PR11 compose. Defaults are TP4, MTP on, `max_num_seqs=64`, and graph cap
`192`. For no-MTP use `MTP=0` and `MAX_CUDAGRAPH_CAPTURE_SIZE=64`. For TP2 use
`TP_SIZE=2` and two visible GPUs.

```yaml
services:
  ds4-b12x:
    image: ${IMAGE:-voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132}
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
        --enable-flashinfer-autotune \
        "$${SPEC_ARGS[@]}"
```

Lucifer Cutlass compose. Defaults are TP4, MTP on, probabilistic draft
sampling, forced C++ PCIe all-reduce selector with `56KB` cutoff,
`flashinfer_cutlass` MoE, and graph cap `192`. Keep
`DRAFT_SAMPLE_METHOD=probabilistic` unless a benchmark explicitly states
otherwise. For no-MTP use `MTP=0` and `MAX_CUDAGRAPH_CAPTURE_SIZE=64`.
For TP2 use `TP_SIZE=2` and two visible GPUs.

```yaml
services:
  ds4-lucifer-cutlass:
    image: ${IMAGE:-hg436/vllm-public:lucifer-9d9a0a0}
    container_name: ${CONTAINER_NAME:-ds4-lucifer-cutlass}
    network_mode: host
    gpus: all
    runtime: nvidia
    ipc: host
    shm_size: 32g
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - ${HOME}/.cache/luci-official:/cache
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${CPP_AR_SELECTOR:-/root/vllm/overlays/lucifer-cpp-ar-selector/vllm/distributed/device_communicators/custom_all_reduce.py}:/opt/vllm/vllm/distributed/device_communicators/custom_all_reduce.py:ro
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: cpp
      VLLM_CPP_AR_1STAGE_NCCL_CUTOFF: ${VLLM_CPP_AR_1STAGE_NCCL_CUTOFF:-56KB}
      VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS: ${VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS:-8}
      MODEL_ID: ${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-deepseek-v4-flash}
      PORT: ${PORT:-5610}
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
      GRAPH_CAP="$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      if [ -z "$${GRAPH_CAP}" ]; then
        if [ "$${MTP:-1}" = "1" ]; then GRAPH_CAP=192; else GRAPH_CAP=64; fi
      fi
      SPEC_ARGS=()
      if [ "$${MTP:-1}" = "1" ]; then
        SPEC_ARGS=(--speculative-config.method mtp --speculative-config.num_speculative_tokens "$${SPECULATIVE_TOKENS}" --speculative-config.draft_sample_method "$${DRAFT_SAMPLE_METHOD}")
      fi
      exec vllm serve "$${MODEL_ID}" \
        --served-model-name "$${SERVED_MODEL_NAME}" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --load-format auto \
        --tensor-parallel-size "$${TP_SIZE}" \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "$${MAX_MODEL_LEN}" \
        --max-num-seqs "$${MAX_NUM_SEQS}" \
        --max-cudagraph-capture-size "$${GRAPH_CAP}" \
        --async-scheduling \
        --no-scheduler-reserve-full-isl \
        --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
        --attention-backend SPARSE_MLA_SM120 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --enable-flashinfer-autotune \
        --kernel-config.moe_backend flashinfer_cutlass \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "high"}' \
        "$${SPEC_ARGS[@]}"
```

## Profile Quality

Both farms used 30 full invocations per profile, not 30 total samples. B12X ran
eight TP2+MTP2 replicas on ports `5500-5507`. Lucifer ran four TP4+MTP2
Cutlass replicas on ports `5600-5603`. All profile runs used
`thinking=true` and `reasoning_effort=high`.

Result roots:

```text
/root/bench-results/ds4-black-pr11-20260609/profile-30x-thinking-high/
/root/bench-results/ds4-lucifer-cutlass-20260609/profile-30x-thinking-high/
```

Quality summary:

| Profile | Samples | B12X score | B12X success | Lucifer score | Lucifer success | Delta |
|---|---:|---|---:|---|---:|---:|
| estonia | 900 / 900 | PASS 879 / FAIL 21 | 97.7% | PASS 878 / FAIL 22 | 97.6% | -0.1 pp |
| lavd-test | 300 / 300 | EXACT 266 / NEAR 21 / FAIL 13 | 95.7% | EXACT 272 / NEAR 19 / FAIL 9 | 97.0% | +1.3 pp |
| hotel-lights | 900 / 900 | EXACT 816 / FAIL 84 | 90.7% | EXACT 814 / FAIL 86 | 90.4% | -0.2 pp |

Profile speed and latency:

| Profile | B12X gen tok/s avg | Lucifer gen tok/s avg | Lucifer/B12X | B12X elapsed avg | Lucifer elapsed avg | Elapsed ratio |
|---|---:|---:|---:|---:|---:|---:|
| estonia | 41.1 | 80.6 | 1.96x | 69.8s | 45.4s | 0.65x |
| lavd-test | 84.0 | 129.2 | 1.54x | 226.8s | 141.7s | 0.62x |
| hotel-lights | 46.2 | 93.1 | 2.02x | 448.4s | 221.6s | 0.49x |

Output length and TTFT:

| Profile | B12X tok avg | B12X tok p50/p90 | Lucifer tok avg | Lucifer tok p50/p90 | B12X TTFT | Lucifer TTFT |
|---|---:|---:|---:|---:|---:|---:|
| estonia | 2,496.9 | 2,146/4,516 | 2,763.1 | 2,385/5,018 | 11.33s | 12.29s |
| lavd-test | 19,470.5 | 16,772/26,873 | 18,397.6 | 16,482/28,233 | 3.64s | 1.78s |
| hotel-lights | 21,073.6 | 19,288/28,850 | 20,684.2 | 19,074/28,492 | 0.98s | 1.53s |

Interpretation: quality is effectively tied on estonia and hotel-lights.
Lucifer is slightly better on LAVD-test (+1.3 pp success). Lucifer is
substantially faster in these profile workloads, especially estonia and
hotel-lights.

## B12X Speed

Decode benchmark:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port PORT \
  --model DeepSeek-V4-Flash \
  --contexts 0 \
  --concurrency 1,2,4,8,16,32,64 \
  --duration 30 \
  --max-tokens 2048 \
  --skip-prefill \
  --kv-budget KV_TOKENS \
  --display-mode plain \
  --output OUT.json
```

B12X aggregate decode tok/s:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 | Accept avg |
|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 131.7 | 220.4 | 359.7 | 541.6 | 780.3 | 1,091.0 | 1,486.6 | 0.000 |
| TP2 | on | probabilistic | 222.5 | 355.4 | 521.7 | 738.8 | 1,006.6 | 1,369.5 | 1,786.6 | 0.687 |
| TP2 | on | greedy | 189.3 | 294.3 | 433.7 | 596.5 | 821.3 | 1,055.6 | 1,311.4 | 0.547 |
| TP4 | off | none | 159.2 | 279.7 | 472.1 | 759.5 | 1,135.4 | 1,656.6 | 2,299.8 | 0.000 |
| TP4 | on | probabilistic | 285.4 | 470.9 | 724.5 | 1,071.1 | 1,504.3 | 1,996.9 | 2,544.7 | 0.706 |
| TP4 | on | greedy | 247.3 | 404.8 | 607.7 | 915.4 | 1,203.5 | 1,689.8 | 2,032.2 | 0.538 |

B12X warm prefill rerun, MTP on:

| TP | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|:---:|:---:|---:|---:|---:|
| TP2 | on | 6,978 | 6,644 | 6,154 |
| TP4 | on | 8,236 | 7,790 | 7,177 |

B12X no-MTP prefill was not remeasured in the 2026-06-09 rerun, so it is not
used for prefill speedup claims.

Result JSONs:

```text
/root/bench-results/ds4-black-pr11-20260608/decode-sweep/
/root/bench-results/ds4-black-pr11-20260609/decode-greedy/
/root/bench-results/ds4-black-pr11-20260609/prefill-rerun/
```

## Lucifer Cutlass Speed

Current recommended Lucifer speed run:

```text
/root/bench-results/ds4-hg436-lucifer-9d9a0a0-20260609/cpp56-serial-speed-probabilistic/
```

Method:

- Image `hg436/vllm-public:lucifer-9d9a0a0`.
- Serial run only: TP2 used GPUs `0,1`; TP4 used GPUs `0,1,2,3`; all other GPUs were idle.
- Forced C++ PCIe all-reduce selector: `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=cpp`, `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB`, `VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=8`.
- MTP runs use `--speculative-config.draft_sample_method probabilistic`.
- Decode uses `contexts=0`, `concurrency=1,2,4,8,16,32,64`, `duration=30`, `max_tokens=8192`.
- Runtime uses `max_num_seqs=64`, `max_num_batched_tokens=8192`, `max_model_len=262144`, graph cap `64` for no-MTP and `192` for MTP.

Startup logs confirm the cpp selector cutoff. With `56KB`, no-MTP decode routes
rows `1,2,4,8` through custom all-reduce and MTP routes rows `3,6` through
custom all-reduce; larger rows fall back to NCCL.

Current Lucifer aggregate decode tok/s:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 123.8 | 205.1 | 350.2 | 565.5 | 827.3 | 1,237.7 | 1,924.2 |
| TP2 | on | probabilistic | 207.1 | 346.4 | 400.4 | 787.1 | 1,153.1 | 1,796.3 | 2,752.5 |
| TP4 | off | none | 146.8 | 260.4 | 452.6 | 745.7 | 1,178.8 | 1,809.8 | 2,739.4 |
| TP4 | on | probabilistic | 257.2 | 439.8 | 583.4 | 1,129.2 | 1,707.6 | 2,686.7 | 3,932.4 |

Current Lucifer prefill:

MTP and draft sampling should not materially change the prefill path, but both
launch modes were measured in the same serial run.

| TP | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---|---:|---:|---:|
| TP2 | off | 13,409 | 12,712 | 11,670 |
| TP2 | on | 12,956 | 12,348 | 11,318 |
| TP4 | off | 15,593 | 14,770 | 13,475 |
| TP4 | on | 15,054 | 14,329 | 13,142 |

## Speed Comparison

Decode speedup is Lucifer/B12X. Values above `1.00x` mean Lucifer is faster.
The Lucifer side is the current recommended cpp56/probabilistic run above.

| TP | MTP | Lucifer draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 0.94x | 0.93x | 0.97x | 1.04x | 1.06x | 1.13x | 1.29x |
| TP2 | on | probabilistic | 0.93x | 0.97x | 0.77x | 1.07x | 1.15x | 1.31x | 1.54x |
| TP4 | off | none | 0.92x | 0.93x | 0.96x | 0.98x | 1.04x | 1.09x | 1.19x |
| TP4 | on | probabilistic | 0.90x | 0.93x | 0.81x | 1.05x | 1.14x | 1.35x | 1.55x |

Prefill speedup is available only for MTP-on, because B12X no-MTP prefill was
not remeasured in the comparable 2026-06-09 rerun.

| TP | MTP | Lucifer draft sampling | 8k | 64k | 128k |
|---|---|---|---:|---:|---:|
| TP2 | on | probabilistic | 1.86x | 1.86x | 1.84x |
| TP4 | on | probabilistic | 1.83x | 1.84x | 1.83x |

Interpretation: B12X remains faster for low-concurrency decode cells, but
Lucifer becomes faster at higher concurrency. Current Lucifer prefill is
roughly 1.8x faster than the B12X MTP-on rerun.

## Historical Lucifer Speed

These rows are kept for regression/debugging only. They were measured before
the current forced cpp56 PCIe all-reduce recipe, or with non-recommended greedy
MTP sampling.

Historical result roots:

```text
/root/bench-results/ds4-lucifer-cutlass-20260609/speed-sweep-v2/
/root/bench-results/ds4-lucifer-cutlass-20260609/speed-sweep-probabilistic/
```

Historical Lucifer decode without forced cpp56 PCIe all-reduce:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 124.4 | 207.8 | 352.0 | 570.8 | 856.9 | 1,267.4 | 1,940.9 |
| TP2 | on | probabilistic | 207.7 | 354.7 | 410.1 | 808.1 | 1,192.0 | 1,871.2 | 2,823.6 |
| TP4 | off | none | 137.8 | 243.7 | 436.2 | 751.2 | 1,176.8 | 1,790.0 | 2,686.2 |
| TP4 | on | probabilistic | 242.1 | 434.3 | 579.1 | 1,140.7 | 1,689.4 | 2,662.8 | 3,912.9 |

Historical greedy/default MTP decode:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | on | greedy/default | 200.9 | 337.2 | 381.8 | 763.6 | 1,128.7 | 1,749.8 | 2,660.3 |
| TP4 | on | greedy/default | 237.0 | 412.8 | 562.3 | 1,076.0 | 1,606.3 | 2,508.5 | 3,670.9 |

Historical prefill without forced cpp56 PCIe all-reduce:

| TP | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---:|---:|---:|
| TP2 | 13,508 | 12,788 | 11,705 |
| TP4 | 15,919 | 14,906 | 13,575 |

## KLD vs H200

Reference:

```text
/root/vllm/artifacts/ds4_flash_2xh200_ref_logits_20260607/nomtp_tp2
```

Current B12X PR11 capture:

```text
/root/vllm/artifacts/ds4_flash_local_black_pr11_logits_20260609/nomtp_tp2_black_pr11_b12x_rowmeta_fullrows_fullcalls128
```

KLD summary, lower is better:

| Variant | Matched rows | Mean | Median | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| B12X PR11 current TP2 no-MTP | 216 | 0.04048 | 0.00654 | 0.11784 | 0.40691 | 0.43863 |
| lucifer cutlass rowmeta fullrows | 350 | 0.02777 | 0.00699 | 0.07651 | 0.21563 | 0.43425 |
| b12x attention/linear + cutlass MoE default | n/a | 0.02719 | 0.00526 | 0.08437 | 0.24788 | 0.50644 |
| b12x attention/linear + cutlass MoE piecewise nobreak | n/a | 0.02884 | 0.00420 | 0.08437 | 0.30857 | 0.50644 |
| b12x offline spawn rowmeta fullrows | n/a | 0.02894 | 0.00692 | 0.08168 | 0.33458 | 0.34684 |
| cstechdev default rowmeta fullrows | n/a | 0.02920 | 0.00534 | 0.09020 | 0.30459 | 0.36811 |

The current B12X PR11 mean is worse than the best historical good runs, but it
is far from the previously broken variants with mean KLD around `1.15+`.
Matched rows are still limited by capture metadata/global-row alignment, not by
missing H200 reference logits.

Result JSON:

```text
/root/vllm/artifacts/ds4_flash_local_black_pr11_logits_20260609/nomtp_tp2_black_pr11_b12x_rowmeta_fullrows_fullcalls128/kld_vs_h200_ref_global_rows_allprompts.json
```

## Notes

- The key B12X PR11 fix is the compressed MLA decode split threshold for MTP
  full graph rows up to `64 * (1 + 2) = 192`.
- B12X speed-matrix and greedy services passed the local smoke test with
  coherent output and `chinese_count=0`.
- Greedy B12X MTP works with local argmax enabled, but was slower than
  probabilistic MTP in this matrix.
- Lucifer Cutlass profile quality is close to B12X quality, with better
  LAVD-test success and much higher profile throughput.
