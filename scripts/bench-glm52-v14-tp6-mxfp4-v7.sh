#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${HELPER:-${ROOT_DIR}/scripts/run-glm52-v14-compose.sh}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}"
MODEL="${MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v14-v7-tp6-mxfp4-a8-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

MTP="${MTP:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
GRAPH="${GRAPH:-64}"
GPU_MEMORY_UTILIZATION_DCP1="${GPU_MEMORY_UTILIZATION_DCP1:-0.957}"
GPU_MEMORY_UTILIZATION_DCP_GT1="${GPU_MEMORY_UTILIZATION_DCP_GT1:-0.950}"
DECODE_CONTEXTS="${DECODE_CONTEXTS:-0,3000}"
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-1}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"

mkdir -p "${RESULT_ROOT}"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v14_v7_tp6_mxfp4.out

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

stop_all() {
  docker ps -a --format '{{.Names}}' |
    awk '/^glm52-v14-v7-tp6-mxfp4-/ {print}' |
    xargs -r docker rm -f >/dev/null 2>&1 || true
}

wait_ready() {
  local name="$1" port="$2" out="$3"
  for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}/models.json" 2>/dev/null; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1 || true
      progress "READY name=${name} port=${port}"
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

start_case() {
  local dcp="$1" port="$2" gpus="$3" out="$4"
  local label="dcp${dcp}"
  local name="glm52-v14-v7-tp6-mxfp4-${label}"
  local served="GLM-5.2-BF16-AMDMXFP4experts-online-mxfp8-a8-tp6-${label}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION_DCP_GT1}"
  if [[ "${dcp}" == "1" ]]; then
    gpu_memory_utilization="${GPU_MEMORY_UTILIZATION_DCP1}"
  fi
  mkdir -p "${out}"
  progress "START label=${label} name=${name} port=${port} gpus=${gpus} gmu=${gpu_memory_utilization}"
  (
    cd "${ROOT_DIR}"
    IMAGE="${IMAGE}" \
    MODEL="${MODEL}" \
    SERVED_MODEL_NAME="${served}" \
    NAME="${name}" \
    COMPOSE_PROJECT_NAME="${name}" \
    PORT="${port}" \
    GPUS="${gpus}" \
    TP=6 \
    DCP="${dcp}" \
    DCP_BACKEND=a2a \
    DCP_A2A_MAX_TOKENS=64 \
    DCP_A2A_LARGE_BACKEND=ag_rs \
    MTP="${MTP}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    GRAPH="${GRAPH}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
    MOE_MODE=force-a8-experimental \
    MOE_BACKEND=b12x \
    LINEAR_BACKEND=auto \
    QUANTIZATION=mxfp4 \
    ONLINE_QUANT=mxfp8 \
    F8_DMA=0 \
    LOAD_FORMAT=instanttensor \
    INSTANTTENSOR_BACKEND=BUFFERED \
    "${HELPER}" up
  ) > "${out}/helper.up.log" 2>&1
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${port}" > "${out}/port"
  printf '%s\n' "${served}" > "${out}/served_model.name"
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
    match = re.search(pattern, text)
    if match:
        raw = match.group(1).replace(",", "")
        out[key] = float(raw) if "." in raw else int(raw)
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

run_decode() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "DECODE_START label=${label}"
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

bench_pair() {
  local dcp_a="$1" dcp_b="$2" port_a="$3" port_b="$4" gpu_a="$5" gpu_b="$6"
  local out_a="${RESULT_ROOT}/dcp${dcp_a}"
  local out_b="${RESULT_ROOT}/dcp${dcp_b}"
  stop_all
  start_case "${dcp_a}" "${port_a}" "${gpu_a}" "${out_a}"
  start_case "${dcp_b}" "${port_b}" "${gpu_b}" "${out_b}"
  wait_ready "$(cat "${out_a}/container.name")" "${port_a}" "${out_a}" &
  local pid_a=$!
  wait_ready "$(cat "${out_b}/container.name")" "${port_b}" "${out_b}" &
  local pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
  progress "SETTLE seconds=${SETTLE_SECONDS} pair=dcp${dcp_a},dcp${dcp_b}"
  sleep "${SETTLE_SECONDS}"
  parse_kv "${out_a}"
  parse_kv "${out_b}"
  run_decode "dcp${dcp_a}" "${port_a}" "$(cat "${out_a}/served_model.name")" "${out_a}" &
  pid_a=$!
  run_decode "dcp${dcp_b}" "${port_b}" "$(cat "${out_b}/served_model.name")" "${out_b}" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
  run_prefill "dcp${dcp_a}" "${port_a}" "$(cat "${out_a}/served_model.name")" "${out_a}" &
  pid_a=$!
  run_prefill "dcp${dcp_b}" "${port_b}" "$(cat "${out_b}/served_model.name")" "${out_b}" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
}

summarize() {
  python3 - "${RESULT_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def decode_rows(path):
    data = load(path)
    out = {}
    for row in data.get("results", []):
        ctx = str(row.get("context", row.get("context_length", "")))
        cc = row.get("concurrency")
        tps = row.get("aggregate_tps", row.get("aggregate_output_tokens_per_second"))
        if cc is not None and tps is not None:
            out[f"ctx{ctx}/cc{int(cc)}"] = float(tps)
    return out

def prefill_rows(path):
    pref = load(path).get("prefill") or {}
    return {
        str(ctx): float(row["tok_per_sec"])
        for ctx, row in pref.items()
        if isinstance(row, dict) and row.get("tok_per_sec") is not None
    }

rows = []
for case in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("dcp")):
    rows.append({
        "case": case.name,
        "kv": load(case / "kv_cache_summary.json"),
        "decode": decode_rows(case / "decode.json"),
        "prefill": prefill_rows(case / "prefill.json"),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
for row in rows:
    print(row["case"], "kv", row["kv"], "decode", row["decode"], "prefill", row["prefill"])
PY
}

case "${1:-run}" in
  run)
    progress "RUN_START image=${IMAGE} model=${MODEL} mtp=${MTP} mml=${MAX_MODEL_LEN} mnbt=${MAX_BATCHED_TOKENS}"
    bench_pair 1 2 5930 5931 0,1,2,3,4,5 8,9,10,11,12,13
    bench_pair 3 6 5930 5931 0,1,2,3,4,5 8,9,10,11,12,13
    summarize | tee "${RESULT_ROOT}/summary.txt"
    progress "RUN_DONE result_root=${RESULT_ROOT}"
    ;;
  summarize)
    summarize
    ;;
  stop)
    stop_all
    ;;
  *)
    echo "usage: $0 [run|summarize|stop]" >&2
    exit 2
    ;;
esac
