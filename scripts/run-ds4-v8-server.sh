#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629}
STANDARD_MODEL=${STANDARD_MODEL:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
DSPARK_MODEL=${DSPARK_MODEL:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}

NAME=${NAME:-ds4-v8}
PORT=${PORT:-8000}
GPUS=${GPUS:-0,1}
TP=${TP:-2}
BACKEND=${BACKEND:-b12x}            # b12x | lucifer-default | lucifer-cutlass
MODE=${MODE:-standard-mtp0}         # standard-mtp0 | standard-mtp2 | dspark
MAX_NUM_SEQS=${MAX_NUM_SEQS:-${SEQ:-64}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_BATCHED=${MAX_BATCHED:-8192}
GPU_MEM_WAS_SET=0
if [[ -n "${GPU_MEM+x}" ]]; then
  GPU_MEM_WAS_SET=1
fi
GPU_MEM=${GPU_MEM:-0.90}
GRAPH=${GRAPH:-auto}
PREFIX_CACHE=${PREFIX_CACHE:-1}
DSPARK_TOKENS=${DSPARK_TOKENS:-5}
MTP_TOKENS=${MTP_TOKENS:-2}
SAMPLE=${SAMPLE:-probabilistic}
CACHE=${CACHE:-/root/.cache/vllm-ds4-v8/$NAME}
CONTAINER_TMP=${CONTAINER_TMP:-$CACHE/tmp}

mkdir -p \
  "$CACHE/vllm" \
  "$CACHE/tilelang/tmp" \
  "$CACHE/tvm" \
  "$CACHE/triton" \
  "$CACHE/torchinductor" \
  "$CACHE/torch_extensions" \
  "$CACHE/flashinfer" \
  "$CONTAINER_TMP"

case "$MODE" in
  standard-mtp0)
    MODEL=$STANDARD_MODEL
    SERVED_MODEL=DeepSeek-V4-Flash
    SPEC_ARGS=()
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=256; fi
    ;;
  standard-mtp2)
    MODEL=$STANDARD_MODEL
    SERVED_MODEL=DeepSeek-V4-Flash
    if [[ "$BACKEND" == "b12x" ]]; then
      SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","moe_backend":"b12x"}' "$MTP_TOKENS" "$SAMPLE")
    else
      SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$MTP_TOKENS" "$SAMPLE")
    fi
    SPEC_ARGS=(--speculative-config "$SPEC_JSON")
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=512; fi
    ;;
  dspark)
    MODEL=$DSPARK_MODEL
    SERVED_MODEL=DeepSeek-V4-Flash-DSpark
    SPEC_JSON=$(printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$DSPARK_MODEL" "$DSPARK_TOKENS" "$SAMPLE")
    SPEC_ARGS=(--speculative-config "$SPEC_JSON")
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=512; fi
    if [[ "$GPU_MEM_WAS_SET" == "0" ]]; then GPU_MEM=0.92; fi
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

test -f "$MODEL/config.json"

case "$BACKEND" in
  b12x)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=(
      -e VLLM_USE_B12X_WO_PROJECTION=1
      -e VLLM_USE_B12X_MHC=1
      -e VLLM_USE_B12X_FP8_GEMM=1
      -e VLLM_USE_B12X_MOE=1
      -e VLLM_USE_B12X_SPARSE_INDEXER=1
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e B12X_MLA_SM120_UNIFIED=1
      -e B12X_MHC_MAX_TOKENS=16384
      -e B12X_DENSE_SPLITK_TURBO=1
      -e B12X_W4A16_TC_DECODE=1
      -e B12X_MOE_FORCE_A16=1
    )
    ;;
  lucifer-cutlass)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
    )
    ;;
  lucifer-default)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
    )
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND" >&2
    exit 2
    ;;
esac

PREFIX_ARGS=(--enable-prefix-caching)
if [[ "$PREFIX_CACHE" != "1" ]]; then
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE:/cache:rw" \
  -v "$CONTAINER_TMP:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/vllm \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e TILELANG_TMP_DIR=/cache/tilelang/tmp \
  -e TVM_CACHE_DIR=/cache/tvm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  "${BACKEND_ENV[@]}" \
  "$IMAGE" \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format auto \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-cudagraph-capture-size "$GRAPH" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  "${SPEC_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${PREFIX_ARGS[@]}"

echo "$NAME $SERVED_MODEL $BACKEND $MODE TP=$TP GPUS=$GPUS PORT=$PORT GRAPH=$GRAPH MAX_NUM_SEQS=$MAX_NUM_SEQS"
