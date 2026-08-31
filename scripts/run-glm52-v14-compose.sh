#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/compose/glm52-v14.yml}"

IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v5-vllmcd272c7-b12xe44cb77-cu132-20260707}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
NAME="${NAME:-glm52-v14}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
DCP="${DCP:-1}"
DCP_BACKEND="${DCP_BACKEND:-a2a}"
DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-64}"
DCP_A2A_LARGE_BACKEND="${DCP_A2A_LARGE_BACKEND:-ag_rs}"
MTP="${MTP:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GRAPH="${GRAPH:-$((MAX_NUM_SEQS * 4))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MOE_MODE="${MOE_MODE:-a4}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
ONLINE_MXFP8="${ONLINE_MXFP8:-0}"
ONLINE_FP8="${ONLINE_FP8:-0}"
ONLINE_FP8_MXFP4="${ONLINE_FP8_MXFP4:-0}"
ONLINE_QUANT="${ONLINE_QUANT:-}"
F8_DMA="${F8_DMA:-0}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
QUANTIZATION_CONFIG_JSON="${QUANTIZATION_CONFIG_JSON:-}"
GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
CACHE_ROOT="${CACHE_ROOT:-/root/.cache/vllm-glm52-v14}"
CACHE_DIR="${CACHE_DIR:-${CACHE_ROOT}/${NAME}}"
TMP_DIR="${TMP_DIR:-/root/vllm/tmp/${NAME}}"
ENV_FILE="${ENV_FILE:-${CACHE_DIR}/compose.env}"
ACTION="${1:-up}"
CONTAINER_LAUNCHER="${CONTAINER_LAUNCHER:-/usr/local/bin/run-glm52-v14-server}"
GPU_DEVICE_REQUESTS="${GPU_DEVICE_REQUESTS:-0}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-${NAME}}"
COMPOSE_PROJECT_NAME="$(printf '%s' "${COMPOSE_PROJECT_NAME}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/_/g; s/^[^a-z0-9]+/glm52_/')"
export COMPOSE_PROJECT_NAME

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${MOE_MODE}" in
  a4|native|default)
    B12X_MOE_FORCE_A8=0
    B12X_MOE_FORCE_A16=0
    ;;
  a16|force-a16)
    B12X_MOE_FORCE_A8=0
    B12X_MOE_FORCE_A16=1
    ;;
  force-a8-experimental|a8-experimental|a8)
    B12X_MOE_FORCE_A8=1
    B12X_MOE_FORCE_A16=0
    ;;
  *)
    die "MOE_MODE must be a4, a16, or force-a8-experimental"
    ;;
esac

case "${F8_DMA}" in
  0|ag|ring) ;;
  *) die "F8_DMA must be 0, ag, or ring" ;;
esac

case "${DCP_A2A_LARGE_BACKEND}" in
  ag_rs|a2a) ;;
  *) die "DCP_A2A_LARGE_BACKEND must be ag_rs or a2a" ;;
esac

case "${MOE_BACKEND}" in
  auto|triton|deep_gemm|deep_gemm_mega_moe|b12x|cutlass|flashinfer_trtllm|flashinfer_cutlass|flashinfer_cutedsl|flashinfer_b12x|marlin|humming|triton_unfused|aiter|flydsl|emulation) ;;
  *) die "MOE_BACKEND is not a supported vLLM MoE backend: ${MOE_BACKEND}" ;;
esac

case "${LINEAR_BACKEND}" in
  auto|b12x|cutlass|flashinfer_cutlass|flashinfer_cutedsl|flashinfer_trtllm|flashinfer_cudnn|flashinfer_b12x|marlin|triton|deep_gemm|torch|aiter|machete|fbgemm|conch|exllama|emulation) ;;
  *) die "LINEAR_BACKEND is not a supported vLLM linear backend: ${LINEAR_BACKEND}" ;;
esac

[[ "${ONLINE_MXFP8}" =~ ^(0|1)$ ]] || die "ONLINE_MXFP8 must be 0 or 1"
[[ "${ONLINE_FP8}" =~ ^(0|1)$ ]] || die "ONLINE_FP8 must be 0 or 1"
[[ "${ONLINE_FP8_MXFP4}" =~ ^(0|1)$ ]] || die "ONLINE_FP8_MXFP4 must be 0 or 1"
[[ "${MTP}" =~ ^[0-9]+$ ]] || die "MTP must be an integer token count"
[[ "${DCP_A2A_MAX_TOKENS}" =~ ^[0-9]+$ ]] || die "DCP_A2A_MAX_TOKENS must be an integer token count"
[[ "${MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || die "MAX_NUM_SEQS must be an integer"
[[ "${GRAPH}" =~ ^[0-9]+$ ]] || die "GRAPH must be an integer"
[[ "${#GLM52_INDEX_TOPK_PATTERN}" -eq 78 ]] || die "GLM52_INDEX_TOPK_PATTERN must be exactly 78 characters, got ${#GLM52_INDEX_TOPK_PATTERN}"

if [[ "${MODEL}" == /* && ! -f "${MODEL}/config.json" ]]; then
  die "MODEL does not look like a local HF checkpoint: ${MODEL}"
fi

mkdir -p "${CACHE_DIR}" "${TMP_DIR}"

if [[ "${GPU_DEVICE_REQUESTS}" == "1" ]]; then
  GPU_COMPOSE_FILE="${GPU_COMPOSE_FILE:-${CACHE_DIR}/gpu-devices.yml}"
  IFS=',' read -r -a gpu_ids <<< "${GPUS}"
  {
    printf '%s\n' 'services:' '  server:' '    privileged: false' '    deploy:' '      resources:' '        reservations:' '          devices:' '            - driver: nvidia' '              device_ids:'
    for gpu_id in "${gpu_ids[@]}"; do
      gpu_id="$(printf '%s' "${gpu_id}" | xargs)"
      [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "GPUS must be a comma-separated list of numeric GPU ids when GPU_DEVICE_REQUESTS=1"
      printf '                - "%s"\n' "${gpu_id}"
    done
    printf '%s\n' '              capabilities: [gpu]'
  } > "${GPU_COMPOSE_FILE}"
  if [[ -n "${COMPOSE_FILES:-}" ]]; then
    COMPOSE_FILES="${COMPOSE_FILES}:${GPU_COMPOSE_FILE}"
  else
    COMPOSE_FILES="${COMPOSE_FILE}:${GPU_COMPOSE_FILE}"
  fi
fi

cat > "${ENV_FILE}" <<EOF
IMAGE=${IMAGE}
NAME=${NAME}
GPUS=${GPUS}
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
PORT=${PORT}
TP=${TP}
DCP=${DCP}
DCP_BACKEND=${DCP_BACKEND}
DCP_A2A_MAX_TOKENS=${DCP_A2A_MAX_TOKENS}
DCP_A2A_LARGE_BACKEND=${DCP_A2A_LARGE_BACKEND}
MTP=${MTP}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
GRAPH=${GRAPH}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MOE_MODE=${MOE_MODE}
MOE_BACKEND=${MOE_BACKEND}
LINEAR_BACKEND=${LINEAR_BACKEND}
ONLINE_MXFP8=${ONLINE_MXFP8}
ONLINE_FP8=${ONLINE_FP8}
ONLINE_FP8_MXFP4=${ONLINE_FP8_MXFP4}
ONLINE_QUANT=${ONLINE_QUANT}
F8_DMA=${F8_DMA}
LOAD_FORMAT=${LOAD_FORMAT}
INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND}
QUANTIZATION=${QUANTIZATION}
QUANTIZATION_CONFIG_JSON=${QUANTIZATION_CONFIG_JSON}
GLM52_INDEX_TOPK_PATTERN=${GLM52_INDEX_TOPK_PATTERN}
CACHE_DIR=${CACHE_DIR}
TMP_DIR=${TMP_DIR}
EOF

compose=(docker compose --env-file "${ENV_FILE}")
IFS=':' read -r -a compose_files <<< "${COMPOSE_FILES:-${COMPOSE_FILE}}"
for compose_file in "${compose_files[@]}"; do
  compose+=(-f "${compose_file}")
done

case "${ACTION}" in
  up|start)
    "${compose[@]}" up -d --remove-orphans
    ;;
  restart)
    "${compose[@]}" up -d --force-recreate --remove-orphans
    ;;
  down|stop)
    "${compose[@]}" down
    ;;
  logs)
    "${compose[@]}" logs -f --tail=200
    ;;
  ps|config)
    "${compose[@]}" "${ACTION}"
    ;;
  *)
    die "usage: $0 [up|restart|down|logs|ps|config]"
    ;;
esac

cat <<EOF
name=${NAME}
image=${IMAGE}
compose_project=${COMPOSE_PROJECT_NAME}
model=${MODEL}
port=${PORT}
gpus=${GPUS}
tp=${TP}
dcp=${DCP}
dcp_backend=${DCP_BACKEND}
dcp_a2a_max_tokens=${DCP_A2A_MAX_TOKENS}
dcp_a2a_large_backend=${DCP_A2A_LARGE_BACKEND}
mtp=${MTP}
moe_mode=${MOE_MODE}
moe_backend=${MOE_BACKEND}
linear_backend=${LINEAR_BACKEND}
online_mxfp8=${ONLINE_MXFP8}
online_fp8=${ONLINE_FP8}
online_quant=${ONLINE_QUANT}
f8_dma=${F8_DMA}
load_format=${LOAD_FORMAT}
instanttensor_backend=${INSTANTTENSOR_BACKEND}
max_num_seqs=${MAX_NUM_SEQS}
graph=${GRAPH}
container_launcher=${CONTAINER_LAUNCHER}
compose_env=${ENV_FILE}
compose_files=${COMPOSE_FILES:-${COMPOSE_FILE}}
EOF
