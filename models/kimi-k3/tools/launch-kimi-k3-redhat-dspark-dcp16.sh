#!/usr/bin/env bash
# Launch the RedHatAI BF16 DSpark profile by overlaying vLLM PR #310 onto the
# immutable Kimi-K3 CUDA 13.3 runtime image.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-kimi-k3-redhat-dspark-dcp16}"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1}"
SOURCE_DIR="${SOURCE_DIR:-/mnt/luke/vllm-k3-infernal-redhat-20260814}"
CACHE_DIR="${CACHE_DIR:-/mnt/luke/kimi-k3-cache/redhat-dspark-dcp16}"

required_files=(
  vllm/model_executor/models/qwen3_dflash.py
  vllm/v1/attention/backends/flash_attn.py
  vllm/v1/worker/cp_utils.py
  vllm/v1/worker/gpu/spec_decode/dflash/utils.py
  vllm/v1/worker/gpu/spec_decode/dspark/utils.py
)
for relative_path in "${required_files[@]}"; do
  if [[ ! -f "${SOURCE_DIR}/${relative_path}" ]]; then
    echo "Required vLLM PR #310 source file is absent: ${SOURCE_DIR}/${relative_path}" >&2
    exit 2
  fi
done
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container already exists: ${CONTAINER_NAME}" >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}"

exec docker run --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  --volume /root/.cache/huggingface:/root/.cache/huggingface:ro \
  --volume "${CACHE_DIR}:/cache/jit" \
  --volume "${SCRIPT_DIR}/serve-kimi-k3-redhat-dspark-dcp16.sh:/usr/local/bin/serve-kimi-k3-redhat-dspark-dcp16:ro" \
  --volume "${SOURCE_DIR}/vllm/model_executor/models/qwen3_dflash.py:/opt/kimi-k3/vllm/vllm/model_executor/models/qwen3_dflash.py:ro" \
  --volume "${SOURCE_DIR}/vllm/v1/attention/backends/flash_attn.py:/opt/kimi-k3/vllm/vllm/v1/attention/backends/flash_attn.py:ro" \
  --volume "${SOURCE_DIR}/vllm/v1/worker/cp_utils.py:/opt/kimi-k3/vllm/vllm/v1/worker/cp_utils.py:ro" \
  --volume "${SOURCE_DIR}/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:ro" \
  --volume "${SOURCE_DIR}/vllm/v1/worker/gpu/spec_decode/dspark/utils.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu/spec_decode/dspark/utils.py:ro" \
  --env PORT="${PORT:-8001}" \
  --env DCP_SIZE="${DCP_SIZE:-16}" \
  --env NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}" \
  --env ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  --env KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-402653184}" \
  --env MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  --env MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  /usr/local/bin/serve-kimi-k3-redhat-dspark-dcp16
