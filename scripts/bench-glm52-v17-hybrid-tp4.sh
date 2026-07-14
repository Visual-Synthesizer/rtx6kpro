#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v17-vllm137d2eb-b12x1377d5f-fi801d57a-cu132-20260714}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:2dd2e4d3e4b1e8b4e3fdc324b7c142e69f15892773a556c14a2b7040c8292f7a}"
EXPECTED_REPO_DIGEST="${EXPECTED_REPO_DIGEST:-voipmonitor/vllm@sha256:8892c8ecae957a3cf131ebd32159aecc0b29daf03bf4ee2deb4dcc3acdafbcd0}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
MODEL="${MODEL:-/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v17-hybrid-tp4-$(date -u +%Y%m%dT%H%M%SZ)}"
CACHE_ROOT="${CACHE_ROOT:-/root/.cache/vllm-glm52-v17-hybrid-repro}"
TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp/glm52-v17-hybrid-repro}"

DCP_VALUES="${DCP_VALUES:-1 2 4}"
GPU_DCP1="${GPU_DCP1:-4,5,6,7}"
GPU_DCP2="${GPU_DCP2:-8,9,10,11}"
GPU_DCP4="${GPU_DCP4:-0,1,2,3}"
PORT_DCP1="${PORT_DCP1:-8423}"
PORT_DCP2="${PORT_DCP2:-8424}"
PORT_DCP4="${PORT_DCP4:-8422}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GRAPH="${GRAPH:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-3072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"
DECODE_DURATION="${DECODE_DURATION:-30}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
PULL_IMAGE="${PULL_IMAGE:-1}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

mkdir -p "${RESULT_ROOT}"
PROGRESS_FILE="${RESULT_ROOT}/progress.log"

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

container_name() {
  printf 'glm52-v17-hybrid-repro-dcp%s' "$1"
}

gpu_set() {
  local dcp="$1"
  case "${dcp}" in
    1) printf '%s' "${GPU_DCP1}" ;;
    2) printf '%s' "${GPU_DCP2}" ;;
    4) printf '%s' "${GPU_DCP4}" ;;
    *) printf 'unsupported DCP value: %s\n' "${dcp}" >&2; return 2 ;;
  esac
}

port_for() {
  local dcp="$1"
  case "${dcp}" in
    1) printf '%s' "${PORT_DCP1}" ;;
    2) printf '%s' "${PORT_DCP2}" ;;
    4) printf '%s' "${PORT_DCP4}" ;;
    *) printf 'unsupported DCP value: %s\n' "${dcp}" >&2; return 2 ;;
  esac
}

stop_servers() {
  local dcp name
  for dcp in ${DCP_VALUES}; do
    name="$(container_name "${dcp}")"
    docker rm -f "${name}" >/dev/null 2>&1 || true
  done
}

cleanup() {
  if [[ "${KEEP_SERVERS}" != "1" ]]; then
    stop_servers
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -f "${BENCH}" ]]; then
  printf 'benchmark client not found: %s\n' "${BENCH}" >&2
  exit 2
fi

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
fi
actual_image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  printf 'image ID mismatch: expected %s, got %s\n' \
    "${EXPECTED_IMAGE_ID}" "${actual_image_id}" >&2
  exit 3
fi
if ! docker image inspect "${IMAGE}" --format '{{range .RepoDigests}}{{println .}}{{end}}' |
    grep -Fxq "${EXPECTED_REPO_DIGEST}"; then
  printf 'repository digest mismatch: expected %s\n' "${EXPECTED_REPO_DIGEST}" >&2
  exit 3
fi
printf '%s\n' "${actual_image_id}" > "${RESULT_ROOT}/image.id"

stop_servers

start_server() {
  local dcp="$1" gpus port name cache_dir tmp_dir out
  gpus="$(gpu_set "${dcp}")"
  port="$(port_for "${dcp}")"
  name="$(container_name "${dcp}")"
  out="${RESULT_ROOT}/dcp${dcp}"
  cache_dir="${CACHE_ROOT}/dcp${dcp}"
  tmp_dir="${TMP_ROOT}/dcp${dcp}"
  mkdir -p "${out}" "${cache_dir}" "${tmp_dir}"

  progress "START dcp=${dcp} gpus=${gpus} port=${port}"
  docker run -d \
    --name "${name}" \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
    -e "MODEL=${MODEL}" \
    -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
    -e "GPUS=${gpus}" \
    -e "PORT=${port}" \
    -e TP=4 \
    -e "DCP=${dcp}" \
    -e MTP=0 \
    -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
    -e "GRAPH=${GRAPH}" \
    -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    -e "MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}" \
    -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
    -v /root/models:/root/models:ro \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -v "${cache_dir}:/cache" \
    -v "${tmp_dir}:/container-tmp" \
    "${IMAGE}" > "${out}/container.id"

  docker inspect "${name}" > "${out}/container.inspect.json"
  if docker inspect "${name}" --format '{{range .Mounts}}{{println .Destination}}{{end}}' |
      grep -Eq '^(/opt/vllm|/opt/venv|.*/site-packages)(/|$)'; then
    printf 'source overlay detected in %s\n' "${name}" >&2
    return 4
  fi
}

wait_ready() {
  local dcp="$1" port name out
  port="$(port_for "${dcp}")"
  name="$(container_name "${dcp}")"
  out="${RESULT_ROOT}/dcp${dcp}"
  for _ in $(seq 1 1800); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1
      grep -Fq 'Loading safetensors using InstantTensor loader' "${out}/server.ready.log"
      if [[ "${dcp}" != "1" ]]; then
        grep -Fq 'Using B12X PCIe DCP collectives' "${out}/server.ready.log"
        grep -Fq 'Warmed up 1 B12X DCP collective signature' "${out}/server.ready.log"
      fi
      progress "READY dcp=${dcp} port=${port}"
      return 0
    fi
    if [[ "$(docker inspect "${name}" --format '{{.State.Status}}' 2>/dev/null || true)" != running ]]; then
      docker logs "${name}" > "${out}/server.failed.log" 2>&1 || true
      progress "FAILED dcp=${dcp} log=${out}/server.failed.log"
      return 1
    fi
    sleep 2
  done
  docker logs "${name}" > "${out}/server.timeout.log" 2>&1 || true
  return 1
}

for dcp in ${DCP_VALUES}; do
  start_server "${dcp}"
done

wait_pids=()
for dcp in ${DCP_VALUES}; do
  wait_ready "${dcp}" &
  wait_pids+=("$!")
done
for pid in "${wait_pids[@]}"; do
  wait "${pid}"
done

progress "SETTLE seconds=${SETTLE_SECONDS}"
sleep "${SETTLE_SECONDS}"

for dcp in ${DCP_VALUES}; do
  port="$(port_for "${dcp}")"
  out="${RESULT_ROOT}/dcp${dcp}"
  progress "DECODE_START dcp=${dcp}"
  python3 "${BENCH}" \
    --host 127.0.0.1 --port "${port}" --model "${SERVED_MODEL_NAME}" \
    --concurrency 1 --contexts 0 --duration "${DECODE_DURATION}" \
    --skip-prefill --no-hw-monitor --display-mode plain \
    --output "${out}/decode-cc1.json" > "${out}/decode-cc1.log" 2>&1
  progress "DECODE_DONE dcp=${dcp}"

  progress "PREFILL_START dcp=${dcp}"
  python3 "${BENCH}" \
    --host 127.0.0.1 --port "${port}" --model "${SERVED_MODEL_NAME}" \
    --prefill-only --standalone-prefill --prefill-contexts 8k,64k \
    --prefill-duration "${PREFILL_DURATION}" --max-tokens 1 \
    --no-hw-monitor --display-mode plain \
    --output "${out}/prefill.json" > "${out}/prefill.log" 2>&1
  docker logs "$(container_name "${dcp}")" > "${out}/server.final.log" 2>&1
  if [[ "${dcp}" == "4" ]]; then
    grep -Fq 'Using borrowed B12X workspaces for TP4/DCP4 sparse MLA prefill' \
      "${out}/server.final.log"
  fi
  progress "PREFILL_DONE dcp=${dcp}"
done

python3 - "${RESULT_ROOT}" ${DCP_VALUES} <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for dcp in map(int, sys.argv[2:]):
    case = root / f"dcp{dcp}"
    decode = json.loads((case / "decode-cc1.json").read_text())
    prefill = json.loads((case / "prefill.json").read_text())["prefill"]
    log = (case / "server.final.log").read_text(errors="replace")
    kv = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
    rows.append({
        "dcp": dcp,
        "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "decode_cc1": decode["results"][0]["aggregate_tps"],
        "prefill_8k": prefill["8192"]["tok_per_sec"],
        "prefill_64k": prefill["65536"]["tok_per_sec"],
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("DCP\tKV tokens\tDecode cc1\tPrefill 8k\tPrefill 64k")
for row in rows:
    print(
        f'{row["dcp"]}\t{row["kv_tokens"]:,}\t{row["decode_cc1"]:.1f}'
        f'\t{row["prefill_8k"]:.0f}\t{row["prefill_64k"]:.0f}'
    )
PY

progress "DONE result_root=${RESULT_ROOT}"
