#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${HELPER:-${ROOT_DIR}/scripts/run-glm52-v15-compose.sh}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v15-reduced-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"

TP="${TP:-8}"
DCP="${DCP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GRAPH="${GRAPH:-6}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8001}"

DECODE_CONTEXTS="${DECODE_CONTEXTS:-0}"
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-1}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"

CASES="${CASES:-nvfp4-a4-orig nvfp4-a4-online-mxfp8 nvfp4-a16-orig nvfp4-a16-online-mxfp8 mxfp4-a8-orig mxfp4-a8-online-mxfp8}"
MTP3_CASE="${MTP3_CASE:-nvfp4-a16-orig}"
RUN_MTP3="${RUN_MTP3:-1}"

mkdir -p "${RESULT_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v15_reduced.out

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

safe_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//'
}

case_vars() {
  local key="$1"
  MODEL_PATH=""
  DISPLAY_NAME=""
  SHORT_NAME=""
  MOE_MODE=""
  QUANTIZATION=""
  ONLINE_QUANT=""
  case "${key}" in
    nvfp4-a4-orig)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A4 orig"
      SHORT_NAME="nvfp4-a4-orig"
      MOE_MODE="a4"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="none"
      ;;
    nvfp4-a4-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A4 online MXFP8"
      SHORT_NAME="nvfp4-a4-online-mxfp8"
      MOE_MODE="a4"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="mxfp8"
      ;;
    nvfp4-a16-orig)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A16 orig"
      SHORT_NAME="nvfp4-a16-orig"
      MOE_MODE="a16"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="none"
      ;;
    nvfp4-a16-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A16 online MXFP8"
      SHORT_NAME="nvfp4-a16-online-mxfp8"
      MOE_MODE="a16"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="mxfp8"
      ;;
    mxfp4-a8-orig)
      MODEL_PATH="${MXFP4_MODEL}"
      DISPLAY_NAME="BF16 AMD MXFP4 experts A8 orig"
      SHORT_NAME="mxfp4-a8-orig"
      MOE_MODE="force-a8-experimental"
      QUANTIZATION="mxfp4"
      ONLINE_QUANT="none"
      ;;
    mxfp4-a8-online-mxfp8)
      MODEL_PATH="${MXFP4_MODEL}"
      DISPLAY_NAME="BF16 AMD MXFP4 experts A8 online MXFP8"
      SHORT_NAME="mxfp4-a8-online-mxfp8"
      MOE_MODE="force-a8-experimental"
      QUANTIZATION="mxfp4"
      ONLINE_QUANT="mxfp8"
      ;;
    *)
      echo "unknown case: ${key}" >&2
      return 2
      ;;
  esac
}

wait_ready() {
  local name="$1" port="$2" out="$3"
  for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}/models.json" 2>/dev/null; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1 || true
      progress "READY name=${name} port=${port}"
      sleep "${SETTLE_SECONDS}"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${name}"; then
      docker logs "${name}" > "${out}/server.failed.log" 2>&1 || true
      progress "FAILED name=${name} out=${out}"
      return 1
    fi
    sleep 2
  done
  docker logs "${name}" > "${out}/server.timeout.log" 2>&1 || true
  progress "READY_TIMEOUT name=${name} port=${port}"
  return 1
}

parse_kv() {
  local out="$1"
  python3 - "$out/server.ready.log" > "$out/kv_cache_summary.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], errors="replace").read()
patterns = {
    "model_loading_gib": r"Loading weights took .*? and ([0-9.]+) GiB",
    "available_kv_cache_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "gpu_kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "max_concurrency": r"Maximum concurrency for .*?: ([0-9.]+)x",
}
out = {}
for key, pattern in patterns.items():
    m = re.search(pattern, text)
    if m:
        raw = m.group(1).replace(",", "")
        out[key] = float(raw) if "." in raw else int(raw)
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

start_case() {
  local key="$1" mtp="$2" out="$3"
  case_vars "${key}"
  local name served
  name="glm52-v15-reduced-$(safe_name "${SHORT_NAME}-dcp${DCP}-mtp${mtp}-p${PORT}")"
  served="GLM-5.2-v15-${SHORT_NAME}-tp${TP}-dcp${DCP}-mtp${mtp}"
  mkdir -p "${out}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  progress "START case=${key} dcp=${DCP} mtp=${mtp} name=${name} gpus=${GPUS} port=${PORT}"
  (
    cd "${ROOT_DIR}"
    IMAGE="${IMAGE}" \
    MODEL="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${served}" \
    NAME="${name}" \
    COMPOSE_PROJECT_NAME="${name}" \
    PORT="${PORT}" \
    GPUS="${GPUS}" \
    TP="${TP}" \
    DCP="${DCP}" \
    DCP_BACKEND=a2a \
    DCP_A2A_MAX_TOKENS=64 \
    DCP_A2A_LARGE_BACKEND=ag_rs \
    MTP="${mtp}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    GRAPH="${GRAPH}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MOE_MODE="${MOE_MODE}" \
    MOE_BACKEND=b12x \
    LINEAR_BACKEND=auto \
    QUANTIZATION="${QUANTIZATION}" \
    ONLINE_QUANT="${ONLINE_QUANT}" \
    F8_DMA=0 \
    LOAD_FORMAT=instanttensor \
    INSTANTTENSOR_BACKEND=BUFFERED \
    "${HELPER}" up
  ) > "${out}/helper.up.log" 2>&1
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${served}" > "${out}/served_model.name"
  printf '%s\n' "${PORT}" > "${out}/port"
  printf '%s\n' "${key}" > "${out}/case.key"
  printf '%s\n' "${mtp}" > "${out}/mtp"
  printf '%s\n' "${DISPLAY_NAME}" > "${out}/display.name"
}

run_decode() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "DECODE_START label=${label} cc=${DECODE_CONCURRENCY}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_before_decode.csv" 2>/dev/null || true
  python3 "${BENCH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${served}" \
    --skip-prefill \
    --contexts "${DECODE_CONTEXTS}" \
    --concurrency "${DECODE_CONCURRENCY}" \
    --duration "${DECODE_DURATION}" \
    --max-tokens "${DECODE_MAX_TOKENS}" \
    --no-hw-monitor \
    --output "${out}/decode.json" > "${out}/decode.log" 2>&1
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_after_decode.csv" 2>/dev/null || true
  progress "DECODE_DONE label=${label}"
}

run_prefill() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "PREFILL_START label=${label}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_before_prefill.csv" 2>/dev/null || true
  python3 "${BENCH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${served}" \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "${PREFILL_CONTEXTS}" \
    --prefill-duration "${PREFILL_DURATION}" \
    --max-tokens 1 \
    --no-hw-monitor \
    --output "${out}/prefill.json" > "${out}/prefill.log" 2>&1
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_after_prefill.csv" 2>/dev/null || true
  progress "PREFILL_DONE label=${label}"
}

stop_case() {
  local name="$1"
  progress "STOP name=${name}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

run_case() {
  local key="$1" mtp="$2" suffix="$3"
  local out="${RESULT_ROOT}/${key}/${suffix}"
  start_case "${key}" "${mtp}" "${out}"
  local name served port label
  name="$(cat "${out}/container.name")"
  served="$(cat "${out}/served_model.name")"
  port="$(cat "${out}/port")"
  label="${key}-${suffix}"
  wait_ready "${name}" "${port}" "${out}"
  parse_kv "${out}"
  run_decode "${label}" "${port}" "${served}" "${out}"
  if [[ "${mtp}" == "0" ]]; then
    run_prefill "${label}" "${port}" "${served}" "${out}"
  fi
  docker logs "${name}" > "${out}/server.final.log" 2>&1 || true
  stop_case "${name}"
}

summarize() {
  python3 - "${RESULT_ROOT}" > "${RESULT_ROOT}/summary.tsv" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}

def decode_tps(path):
    data = load(path / "decode.json")
    for row in data.get("results", []):
        if int(row.get("context_tokens", -1)) == 0 and int(row.get("concurrency", -1)) == 1:
            return row.get("aggregate_tps") or row.get("server_gen_throughput")
    return None

def prefill(path, key):
    pref = load(path / "prefill.json").get("prefill") or {}
    row = pref.get(key)
    if isinstance(row, dict):
        return row.get("tok_per_sec")
    return None

def kv_tokens(path):
    return load(path / "kv_cache_summary.json").get("gpu_kv_cache_tokens")

print("case\tmtp\tdecode_cc1_tps\tprefill_8k_tps\tprefill_64k_tps\tkv_tokens\tpath")
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for run_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        mtp_file = run_dir / "mtp"
        mtp = mtp_file.read_text().strip() if mtp_file.exists() else ""
        print(
            f"{case_dir.name}\t{mtp}\t"
            f"{decode_tps(run_dir) or ''}\t{prefill(run_dir, '8192') or ''}\t"
            f"{prefill(run_dir, '65536') or ''}\t{kv_tokens(run_dir) or ''}\t{run_dir}"
        )
PY
  column -t -s $'\t' "${RESULT_ROOT}/summary.tsv" | tee "${RESULT_ROOT}/summary.txt"
}

progress "RESULT_ROOT=${RESULT_ROOT}"
for case_key in ${CASES}; do
  run_case "${case_key}" 0 "mtp0"
  summarize || true
done

if [[ "${RUN_MTP3}" == "1" ]]; then
  run_case "${MTP3_CASE}" 3 "mtp3"
  summarize || true
fi

progress "DONE result_root=${RESULT_ROOT}"
summarize
