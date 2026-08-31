# DeepSeek-V4-Flash v6 Eldritch Enlightenment

This page documents DeepSeek-V4-Flash on the shared Eldritch Enlightenment Docker
image. The same image is used for GLM-5.2 v13, Kimi 2.7, and MiMo validation.

## Docker Image

```text
voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629
voipmonitor/vllm@sha256:2ddd49e4ee162ea79a0f2e537aec3e18918bef168e3d684cfa5332151c7fbe2e
```

Build recipe:

```text
https://github.com/local-inference-lab/blackwell-llm-docker
build-eldritch-enlightenment-head66-pr20-cu132.sh
blackwell-llm-docker commit 60fa373
```

See [`eldritch-enlightenment-docker.md`](./eldritch-enlightenment-docker.md) for the exact
component pins.

## Model

```text
deepseek-ai/DeepSeek-V4-Flash
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

## Variants

| Variant | Attention | MoE / linear | Notes |
|---|---|---|---|
| B12X | `B12X_MLA_SPARSE` | `b12x` MoE + `b12x` linear | Uses B12X PCIe oneshot all-reduce. `MAX_NUM_BATCHED_TOKENS=8192` is validated with B12X PR #20. |
| Lucifer CUTLASS | `FLASHINFER_MLA_SPARSE_DSV4` | `flashinfer_cutlass` MXFP4 MoE | Fastest high-concurrency decode in this TP2 sample. |
| Lucifer default | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path (`DEEPGEMM_MXFP4`) | Best prefill in this TP2 sample. |

## Docker Compose

Set `VARIANT` to `b12x`, `lucifer-cutlass`, or `lucifer-default`. Set
`MTP_TOKENS=0`, `2`, or `3`; `0` disables MTP. The legacy `MTP=1` toggle is
still accepted and maps to `MTP_TOKENS=2`.

```yaml
services:
  ds4:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629}
    container_name: ${NAME:-ds4-v6}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    init: true
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-ds4-v6}:/cache
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
      MTP_TOKENS: ${MTP_TOKENS:-0}
      MTP: ${MTP:-}
      TP_SIZE: ${TP_SIZE:-2}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-128}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      GRAPH_CAP: ${GRAPH_CAP:-256}
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

        EXTRA_ARGS=()
        SPEC_ARGS=()
        if [ -n "$${MTP:-}" ] && [ "$${MTP}" = "1" ] && [ "$${MTP_TOKENS}" = "0" ]; then
          MTP_TOKENS=2
        fi

        case "$${VARIANT}" in
          b12x)
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-8192}"
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
            EXTRA_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
            if [ "$${MTP_TOKENS}" != "0" ]; then
              SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"b12x\"}")
            fi
            ;;
          lucifer-cutlass)
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-8192}"
            EXTRA_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass --disable-custom-all-reduce)
            if [ "$${MTP_TOKENS}" != "0" ]; then
              SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\"}")
            fi
            ;;
          lucifer-default)
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-8192}"
            EXTRA_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --disable-custom-all-reduce)
            if [ "$${MTP_TOKENS}" != "0" ]; then
              SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\"}")
            fi
            ;;
          *)
            echo "Unknown VARIANT=$${VARIANT}" >&2
            exit 2
            ;;
        esac

        if [ "$${MTP_TOKENS}" != "0" ] && [ "$${GRAPH_CAP}" = "256" ]; then
          GRAPH_CAP=512
        fi

        exec vllm serve "$${MODEL_PATH}" \
          --served-model-name DeepSeek-V4-Flash \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 256 \
          --load-format auto \
          --tensor-parallel-size "$${TP_SIZE}" \
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
          "$${EXTRA_ARGS[@]}" \
          "$${SPEC_ARGS[@]}"
```

## Single Docker Run

Example for TP2 B12X without MTP:

```bash
docker rm -f ds4-v6 2>/dev/null || true

docker run -d --name ds4-v6 \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host --init \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-ds4-v6:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_MHC_MAX_TOKENS=16384 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 --served-model-name DeepSeek-V4-Flash --host 0.0.0.0 --port 8000 --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --load-format auto --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 256 --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" --async-scheduling --no-scheduler-reserve-full-isl --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high --attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x'
```

## Benchmarks

Current `8722ac7/b12x15cd38c` clean-image smoke, TP2 B12X without MTP,
`max_num_seqs=1`, graph cap `4`, `max_num_batched_tokens=8192`: startup used
V2 runner, B12X PCIe oneshot allreduce, B12X Mxfp4 MoE, and cleanly passed the
64k prefill repro with `prompt_tokens=64021`, `completion_tokens=128`, and no
CUDA/runtime errors. `/mnt/test.py -L` on the previous same-code B12X recipe
returned coherent output with `0` CJK and about `132-133 tok/s`
generation-only. The obsolete
`--b12x-virtual-tp-moe-intermediate-alignment` option is not used by this
image.

All rows below are TP2 on RTX PRO 6000 Blackwell, `kv-cache-dtype=fp8`,
`max-model-len=262144`, `max-num-seqs=128`, 30 second sustained decode cells,
and the final `vfcc6141` image. Older B12X rows used `MAX_NUM_BATCHED_TOKENS=4096`;
Lucifer variants used `8192`. The audited rerun stores one docker log per
measured container in the artifact directory.

### Decode Throughput

Aggregate decode tok/s at `ctx=0k`. `MTP tokens=2` is the best setting in this
rerun; `MTP tokens=3` is still faster than no-MTP for cc1, but loses throughput
against `2` because acceptance drops.

| Variant | MTP tokens | cc1 | cc64 | cc128 | Accept avg |
|---|---:|---:|---:|---:|---:|
| B12X | 0 | 130.8 | 1379.1 | 1893.3 | n/a |
| B12X | 2 | 214.6 | 1487.3 | 1592.6 | 0.55-0.73 |
| B12X | 3 | 195.5 | 1264.4 | 1372.3 | 0.42-0.46 |
| Lucifer CUTLASS | 0 | 123.9 | 2004.0 | 3243.7 | n/a |
| Lucifer CUTLASS | 2 | 211.5 | 2867.1 | 4344.3 | 0.64-0.69 |
| Lucifer CUTLASS | 3 | 203.4 | 2689.6 | 3997.3 | 0.48-0.49 |
| Lucifer default | 0 | 122.7 | 1828.0 | 3006.1 | n/a |
| Lucifer default | 2 | 205.2 | 2596.8 | 4007.5 | 0.67-0.74 |
| Lucifer default | 3 | 195.4 | 2455.6 | 3685.8 | 0.49-0.52 |

B12X remains the best no-MTP cc1 path. Lucifer CUTLASS is the best high
concurrency decode path, especially with `MTP_TOKENS=2`.

### Prefill Throughput

Client-side prefill tok/s from TTFT. MTP is irrelevant for standalone prefill,
so the table records MTP-off launches.

| Variant | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---:|---:|---:|
| B12X | 7600 | 5433 | 4059 |
| Lucifer CUTLASS | 13241 | 12612 | 11574 |
| Lucifer default | 13525 | 12912 | 11847 |

### Smoke

`python3 /mnt/test.py` smoke results:

| Variant | MTP tokens | Finish | CJK | Generation tok/s |
|---|---:|---|---:|---:|
| B12X | off | `stop` | 0 | 131.9 |
| Lucifer CUTLASS | off | `stop` | 0 | 125.0 |
| Lucifer default | off | `stop` | 0 | 124.3 |

## Artifacts

```text
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-b12x-tp2-mtp0-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-b12x-tp2-mtp2-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-b12x-tp2-mtp3-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-cutlass-tp2-mtp0-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-cutlass-tp2-mtp2-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-cutlass-tp2-mtp3-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-default-tp2-mtp0-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-default-tp2-mtp2-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4-v6-lucifer-default-tp2-mtp3-decode-c1c64c128.json
/root/bench-results/ds4-v6-mtp-audit-20260627/ds4v6-*.log
/root/bench-results/final-eldritch-20260626/ds4-final-b12x-tp2-mtp0-prefill.json
/root/bench-results/final-eldritch-20260626/ds4-final-lucifer-cutlass-tp2-mtp0-prefill.json
/root/bench-results/final-eldritch-20260626/ds4-final-lucifer-default-tp2-mtp0-prefill.json
```
