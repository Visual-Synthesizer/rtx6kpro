#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-voipmonitor/vllm:fathomless-firmament-ds4-v10-vllm61f32d0-b12x90172a5-fi7176f85-cu132-20260710}
STANDARD_MODEL=${STANDARD_MODEL:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
DSPARK_MODEL=${DSPARK_MODEL:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}

NAME=${NAME:-ds4-v10}
PORT=${PORT:-8000}
GPUS=${GPUS:-0,1}
TP=${TP:-${TP_SIZE:-2}}
MODE=${MODE:-dspark}
BACKEND=${BACKEND:-lucifer-cutlass}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-${SEQ:-64}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_BATCHED=${MAX_BATCHED:-${MAX_NUM_BATCHED_TOKENS:-8192}}
PREFIX_CACHE=${PREFIX_CACHE:-1}
ALLREDUCE_MODE=${ALLREDUCE_MODE:-b12x}
CACHE=${CACHE:-/root/.cache/vllm-ds4-v10/${NAME}}
CONTAINER_TMP=${CONTAINER_TMP:-${CACHE}/tmp}
ENABLE_TOPO_PIN=${ENABLE_TOPO_PIN:-0}
VLLM_INTERNAL_HOST_IP=${VLLM_INTERNAL_HOST_IP:-${VLLM_HOST_IP:-127.0.0.1}}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

case "${MODE}" in
  off|mtp0|standard-mtp0|mtp2|standard-mtp2|mtp3|standard-mtp3)
    test -f "${STANDARD_MODEL}/config.json"
    ;;
  dspark)
    test -f "${DSPARK_MODEL}/config.json"
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

IFS=, read -r -a gpu_list <<<"${GPUS}"
if (( ${#gpu_list[@]} != TP )); then
  echo "GPUS=${GPUS} exposes ${#gpu_list[@]} devices but TP=${TP}" >&2
  exit 2
fi

mkdir -p "${CACHE}" "${CONTAINER_TMP}"

pin_args=()
topo_cpuset=${PIN_CPUSET_CPUS:-}
topo_memset=${PIN_CPUSET_MEMS:-0}
if [[ "${ENABLE_TOPO_PIN}" == "1" || -n "${topo_cpuset}" ]]; then
  if [[ -z "${topo_cpuset}" ]]; then
    case "${GPUS}" in
      0,1) topo_cpuset=0-7,64-71 ;;
      2,3) topo_cpuset=8-15,72-79 ;;
      4,5) topo_cpuset=16-23,80-87 ;;
      6,7) topo_cpuset=24-31,88-95 ;;
      8,9) topo_cpuset=32-39,96-103 ;;
      10,11) topo_cpuset=40-47,104-111 ;;
      12,13) topo_cpuset=48-55,112-119 ;;
      14,15) topo_cpuset=56-63,120-127 ;;
      0,1,2,3) topo_cpuset=0-15,64-79 ;;
      4,5,6,7) topo_cpuset=16-31,80-95 ;;
      8,9,10,11) topo_cpuset=32-47,96-111 ;;
      12,13,14,15) topo_cpuset=48-63,112-127 ;;
      *) echo "No CPU topology mapping for GPUS=${GPUS}" >&2; exit 2 ;;
    esac
  fi
  pin_args=(--cpuset-cpus "${topo_cpuset}" --cpuset-mems "${topo_memset}")
fi

helper_env=(
  -e MODE="${MODE}"
  -e BACKEND="${BACKEND}"
  -e STANDARD_MODEL="${STANDARD_MODEL}"
  -e DSPARK_MODEL="${DSPARK_MODEL}"
  -e PORT="${PORT}"
  -e TP_SIZE="${TP}"
  -e DCP_SIZE="${DCP_SIZE:-1}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_BATCHED}"
  -e PREFIX_CACHE="${PREFIX_CACHE}"
  -e ALLREDUCE_MODE="${ALLREDUCE_MODE}"
)

# Forward only explicitly set expert/experimental controls. Defaults remain in
# the image helper, so this wrapper cannot silently diverge from Compose.
optional_env=(
  GPU_MEMORY_UTILIZATION BLOCK_SIZE LOAD_FORMAT ENABLE_FLASHINFER_AUTOTUNE
  DSPARK_TOKENS DRAFT_SAMPLE_METHOD REJECTION_SAMPLE_METHOD INDEXER_BACKEND
  CUDAGRAPH_CAPTURE_SIZES MAX_CUDAGRAPH_CAPTURE_SIZE GRAPH
  DSPARK_CAPACITY DSPARK_CAPACITY_VERIFICATION_MODE
  DSPARK_CONFIDENCE_THRESHOLD DSPARK_BUDGET_FRAC
  DSPARK_CONFIDENCE_TEMPERATURE DSPARK_ONLINE_STS DSPARK_SPS_CURVE
  DSPARK_SPS_OVERHEAD_MS DSPARK_FP8_DRAFT_HEAD
  DSPARK_DYNAMIC_DRAFT_DEPTH DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW
  DSPARK_CAPACITY_LOG_INTERVAL DSPARK_STS_LOG_INTERVAL DSPARK_TP_CHECK
  SP_ASYNC_TP SP_MIN_TOKEN_NUM EXTRA_VLLM_ARGS DRY_RUN
)
if [[ -n "${GPU_MEM+x}" ]]; then
  helper_env+=(-e GPU_MEMORY_UTILIZATION="${GPU_MEM}")
fi
for key in "${optional_env[@]}"; do
  if [[ -n "${!key+x}" ]]; then helper_env+=(-e "${key}=${!key}"); fi
done

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --init \
  "${pin_args[@]}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "${CACHE}:/cache:rw" \
  -v "${CONTAINER_TMP}:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e VLLM_HOST_IP="${VLLM_INTERNAL_HOST_IP}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e TMPDIR=/container-tmp \
  "${helper_env[@]}" \
  --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  "${IMAGE}"

printf '%s mode=%s backend=%s allreduce=%s tp=%s gpus=%s port=%s max_seqs=%s graph=%s cpuset=%s memset=%s\n' \
  "${NAME}" "${MODE}" "${BACKEND}" "${ALLREDUCE_MODE}" "${TP}" "${GPUS}" \
  "${PORT}" "${MAX_NUM_SEQS}" "${GRAPH:-auto}" "${topo_cpuset:-none}" "${topo_memset:-none}"
