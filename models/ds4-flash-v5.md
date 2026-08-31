# DeepSeek-V4-Flash v5 Eldritch DS4Fix

This page documents the clean Eldritch DS4F image that can run the same
DeepSeek-V4-Flash checkpoint in three runtime variants:

| Variant | Attention | MoE / linear |
|---|---|---|
| B12X | `B12X_MLA_SPARSE` | `b12x` MoE + `b12x` linear |
| Lucifer CUTLASS | `FLASHINFER_MLA_SPARSE_DSV4` | FlashInfer CUTLASS MXFP4 MoE |
| Lucifer default | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path |

The image keeps vLLM's upstream packed KV layout and fixes the DS4 decode
throughput regression seen in the first Eldritch packed-KV image.

## Docker Image

```text
voipmonitor/vllm:eldritch-ds4fix-v3289aec-b12xf81d898-cu132-20260625
voipmonitor/vllm@sha256:9119fb130fc5d972b2a7241515f03ee99ec191e6a2198783deec630784647fab
```

Build recipe:

```text
local-inference-lab/blackwell-llm-docker: build-eldritch-ds4fix-cu132.sh
```

Rebuild:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
BUILD_BASE_IMAGE=0 PUSH_IMAGE=1 ./build-eldritch-ds4fix-cu132.sh
```

| Component | Revision |
|---|---|
| vLLM | `local-inference-lab/vllm codex/eldritch-ds4-regression-fix-20260625 @ 3289aec5ee1df516421bffe6d8013980d54daed6` |
| B12X | `voipmonitor/b12x codex/ds4-packedkv-stride-20260624 @ f81d8985e2c387d0dec5f9f310a4e7a45be72adc` |
| FlashInfer | `b3baedbbef2686df91b6dc43818ee56fe26ceba2` |
| DeepGEMM | `14073b4e1e706506e193231209738c848d092a1f` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| CUDA / torch | CUDA `13.2`, torch `2.12.0+cu132` |

Relevant patches:

| Area | Purpose |
|---|---|
| DS4 packed KV | B12X accepts vLLM's packed KV page views without materializing contiguous copies. |
| DS4 sparse attention | Lucifer/FlashInfer path keeps input GEMMs and RMSNorm in the captured graph; only sparse attention body is eager-break wrapped. This restores v3-level decode throughput. |
| FlashInfer CUTLASS MoE | No-bias DS4 experts pass `None` bias instead of synthetic zero tensors. |
| Compile wrappers | Lucifer mHC TileLang and FP8 o-proj DeepGEMM calls are wrapped so the clean Docker image starts without runtime overlays. |

## Model

```text
deepseek-ai/DeepSeek-V4-Flash
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

## Docker Compose

Set `VARIANT` to `b12x`, `lucifer-cutlass`, or `lucifer-default`. Set
`MTP=1` to enable DS4 MTP with two speculative tokens.

```yaml
services:
  ds4:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-ds4fix-v3289aec-b12xf81d898-cu132-20260625}
    container_name: ${NAME:-ds4-v5}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-ds4-v5}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "4096"
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
      PORT: ${PORT:-8000}
      VARIANT: ${VARIANT:-b12x}
      MTP: ${MTP:-0}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-128}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-4096}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      GRAPH_CAP: ${GRAPH_CAP:-256}
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

        EXTRA_ARGS=""
        SPEC_ARGS=""

        case "$${VARIANT}" in
          b12x)
            export VLLM_USE_B12X_WO_PROJECTION=1
            export VLLM_USE_B12X_MHC=1
            export VLLM_USE_B12X_FP8_GEMM=1
            export VLLM_USE_B12X_MOE=1
            export VLLM_USE_B12X_SPARSE_INDEXER=1
            export VLLM_ENABLE_PCIE_ALLREDUCE=1
            export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
            export B12X_MLA_SM120_UNIFIED=1
            export B12X_MHC_MAX_TOKENS=16384
            export B12X_DENSE_SPLITK_TURBO=1
            export B12X_W4A16_TC_DECODE=1
            EXTRA_ARGS="--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x"
            if [ "$${MTP}" = "1" ]; then
              SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"b12x\"}"
            fi
            ;;
          lucifer-cutlass)
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            EXTRA_ARGS="--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass --disable-custom-all-reduce"
            if [ "$${MTP}" = "1" ]; then
              SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_sample_method\":\"probabilistic\"}"
            fi
            ;;
          lucifer-default)
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            EXTRA_ARGS="--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --disable-custom-all-reduce"
            if [ "$${MTP}" = "1" ]; then
              SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_sample_method\":\"probabilistic\"}"
            fi
            ;;
          *)
            echo "Unknown VARIANT=$${VARIANT}" >&2
            exit 2
            ;;
        esac

        exec vllm serve "$${MODEL_PATH}" \
          --served-model-name DeepSeek-V4-Flash \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 256 \
          --load-format auto \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-model-len 262144 \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-cudagraph-capture-size "$${GRAPH_CAP}" \
          --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
          --async-scheduling \
          --no-scheduler-reserve-full-isl \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-flashinfer-autotune \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --default-chat-template-kwargs.thinking=true \
          --default-chat-template-kwargs.reasoning_effort=high \
          $${EXTRA_ARGS} \
          $${SPEC_ARGS}
```

Examples:

```bash
VARIANT=b12x MTP=0 CUDA_VISIBLE_DEVICES=0,1 PORT=8000 docker compose up -d
VARIANT=lucifer-cutlass MTP=1 MAX_NUM_BATCHED_TOKENS=8192 CUDA_VISIBLE_DEVICES=0,1 PORT=8001 docker compose up -d
VARIANT=lucifer-default MTP=1 MAX_NUM_BATCHED_TOKENS=8192 CUDA_VISIBLE_DEVICES=0,1 PORT=8002 docker compose up -d
```

The compose default uses `MAX_NUM_BATCHED_TOKENS=4096` because B12X DS4 can
hit a CUTLASS DSL illegal-address path at `8192` in the MG dual-prefill path.
The Lucifer benchmark rows below used `8192` for parity with v3/v4.

## Benchmark Method

All measurements below are TP2 on RTX PRO 6000 Blackwell, `kv-cache-dtype=fp8`,
`max-model-len=262144`, `max-num-seqs=128`, and 30 second sustained decode cells.
B12X used `max-num-batched-tokens=4096`; Lucifer used `8192`.

Command shape:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port PORT \
  --contexts 0k \
  --concurrency 1,2,4,8,16,32,64,128 \
  --skip-prefill \
  --max-tokens 2048 \
  --no-hw-monitor
```

Prefill:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port PORT \
  --prefill-contexts 8k,64k,128k \
  --prefill-only \
  --no-hw-monitor
```

## Decode Throughput

Aggregate decode tok/s, context `0k`.

| Variant | MTP | C1 | C2 | C4 | C8 | C16 | C32 | C64 | C128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B12X | off | 124.0 | 205.7 | 335.5 | 495.2 | 726.2 | 1010.5 | 1366.0 | 1890.5 |
| B12X | on | 187.4 | 285.4 | 435.5 | 606.6 | 856.1 | 1134.9 | 1470.5 | 1062.7 |
| Lucifer CUTLASS | off | 116.4 | 198.6 | 342.4 | 559.7 | 844.2 | 1217.2 | 1859.4 | 2988.0 |
| Lucifer CUTLASS | on | 191.9 | 322.3 | 519.9 | 761.4 | 1117.7 | 1710.4 | 2576.0 | 3844.8 |
| Lucifer default | off | 117.6 | 187.9 | 317.7 | 521.9 | 774.8 | 1126.6 | 1702.4 | 2789.4 |
| Lucifer default | on | 186.8 | 305.4 | 485.5 | 712.0 | 1021.2 | 1597.4 | 2398.8 | 3603.5 |

B12X MTP was measured with `max-cudagraph-capture-size=256` and
`max-num-batched-tokens=4096`. `max-cudagraph-capture-size=512` did not reach
ready state in a practical time window on this stack. The B12X MTP C128 cell is
therefore not a recommended production point; it appears to fall outside the
fast graph coverage and is slower than no-MTP.

## MTP Acceptance

| Variant | C1 accept | C128 accept |
|---|---:|---:|
| B12X | 54.3% | 59.5% |
| Lucifer CUTLASS | 59.4% | 63.8% |
| Lucifer default | 57.8% | 63.5% |

B12X MTP must not combine `draft_sample_method="probabilistic"` with
`use_local_argmax_reduction=true`. vLLM rejects that combination. If local
argmax reduction is desired, use a greedy draft sampler and measure it
separately.

## Prefill Throughput

Client-side prefill tok/s from TTFT.

| Variant | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---:|---:|---:|---:|
| B12X | off | 8,386 | 7,922 | 7,241 |
| B12X | on | 7,546 | 7,685 | 7,086 |
| Lucifer CUTLASS | off | 11,640 | 11,183 | 10,346 |
| Lucifer CUTLASS | on | 11,525 | 11,051 | 10,221 |
| Lucifer default | off | 12,139 | 11,648 | 10,759 |
| Lucifer default | on | 11,576 | 11,490 | 10,608 |

Interpretation:

| Observation | Practical consequence |
|---|---|
| Lucifer variants are much faster than B12X for DS4 prefill. | Use Lucifer attention for prefill-heavy DS4 workloads unless B12X-specific features are required. |
| Lucifer CUTLASS has the best decode at high concurrency. | It is the best all-around Lucifer setting for DS4 decode. |
| Lucifer default has the best prefill in this sample. | For pure long-prefill benchmarks, default MoE can edge out CUTLASS. |
| B12X no-MTP is stable but slower at high concurrency. | Keep it for B12X integration coverage and comparison, not as the fastest DS4 serving recipe. |
| B12X MTP improves C1..C64 but regresses C128 with cap256. | Do not use B12X MTP C128 until the cap512 start/capture issue is resolved. |

## Artifacts

Decode:

```text
/root/bench-results/ds4-v5/b12x-nomtp-tp2-decode-ctx0-c1_128-final-v3289aec-mbt4096-20260625.json
/root/bench-results/ds4-v5/b12x-mtp-tp2-decode-ctx0-c1_128-final-v3289aec-mbt4096-gcap256-20260625.json
/root/bench-results/ds4-v5/lucifer-cutlass-nomtp-tp2-decode-ctx0-c1_128-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-cutlass-mtp-tp2-decode-ctx0-c1_128-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-default-nomtp-tp2-decode-ctx0-c1_128-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-default-mtp-tp2-decode-ctx0-c1_128-final-v3289aec-20260625.json
```

Prefill:

```text
/root/bench-results/ds4-v5/b12x-nomtp-tp2-prefill-8k64k128k-final-v3289aec-mbt4096-20260625.json
/root/bench-results/ds4-v5/b12x-mtp-tp2-prefill-8k64k128k-final-v3289aec-mbt4096-gcap256-20260625.json
/root/bench-results/ds4-v5/lucifer-cutlass-nomtp-tp2-prefill-8k64k128k-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-cutlass-mtp-tp2-prefill-8k64k128k-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-default-nomtp-tp2-prefill-8k64k128k-final-v3289aec-20260625.json
/root/bench-results/ds4-v5/lucifer-default-mtp-tp2-prefill-8k64k128k-final-v3289aec-20260625.json
```
