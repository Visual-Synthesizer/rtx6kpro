#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${HELPER:-${ROOT_DIR}/scripts/run-glm52-v14-compose.sh}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v5-vllmcd272c7-b12xe44cb77-cu132-20260707}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v14-dcp-hybrid-v5-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"

DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-1,2,4,8,16,32}"
DECODE_CONTEXTS="${DECODE_CONTEXTS:-0}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"

mkdir -p "${RESULT_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v14_dcp_hybrid_v5.out

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

csv_count() {
  local csv="$1"
  if [[ -z "${csv}" ]]; then
    echo 0
  else
    awk -F, '{print NF}' <<< "${csv}"
  fi
}

wait_ready() {
  local name="$1" port="$2" out="$3"
  for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}/models.json" 2>/dev/null; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1 || true
      progress "GLM52_V14_V5_READY name=${name} port=${port}"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${name}"; then
      docker logs "${name}" > "${out}/server.failed.log" 2>&1 || true
      progress "GLM52_V14_V5_FAILED name=${name} out=${out}"
      return 1
    fi
    sleep 2
  done
  progress "GLM52_V14_V5_READY_TIMEOUT name=${name} port=${port}"
  return 1
}

start_with_helper() {
  local label="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6" dcp="$7" mtp="$8" moe_mode="$9" quantization="${10}" online_quant="${11}" max_model_len="${12}" max_num_seqs="${13}" max_batched_tokens="${14}" graph="${15}" gpu_mem="${16}" out="${17}"
  local name="glm52-v14-v5-${label}"
  mkdir -p "${out}"
  progress "GLM52_V14_V5_START label=${label} name=${name} model=${model} tp=${tp} dcp=${dcp} mtp=${mtp} gpus=${gpus} port=${port}"
  (
    cd "${ROOT_DIR}"
    IMAGE="${IMAGE}" \
    MODEL="${model}" \
    SERVED_MODEL_NAME="${served}" \
    NAME="${name}" \
    COMPOSE_PROJECT_NAME="${name}" \
    PORT="${port}" \
    GPUS="${gpus}" \
    TP="${tp}" \
    DCP="${dcp}" \
    DCP_BACKEND=a2a \
    DCP_A2A_MAX_TOKENS=64 \
    DCP_A2A_LARGE_BACKEND=ag_rs \
    MTP="${mtp}" \
    MAX_NUM_SEQS="${max_num_seqs}" \
    GRAPH="${graph}" \
    MAX_MODEL_LEN="${max_model_len}" \
    MAX_BATCHED_TOKENS="${max_batched_tokens}" \
    GPU_MEMORY_UTILIZATION="${gpu_mem}" \
    MOE_MODE="${moe_mode}" \
    QUANTIZATION="${quantization}" \
    ONLINE_QUANT="${online_quant}" \
    F8_DMA=0 \
    LOAD_FORMAT=instanttensor \
    INSTANTTENSOR_BACKEND=BUFFERED \
    "${HELPER}" up
  ) > "${out}/helper.up.log" 2>&1
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${served}" > "${out}/served_model.name"
  printf '%s\n' "${port}" > "${out}/port"
}

stop_case() {
  local out="$1"
  if [[ -f "${out}/container.name" ]]; then
    docker rm -f "$(cat "${out}/container.name")" >/dev/null 2>&1 || true
  fi
}

stop_all_glm52_v5() {
  docker ps -a --format '{{.Names}}' |
    awk '/^(glm52-v14-v5-|glm52-v14-v4-|glm52-v14-|glm52-mxfp4|fable-dcphyb64)/ {print}' |
    xargs -r docker rm -f >/dev/null 2>&1 || true
}

run_decode() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "GLM52_V14_V5_DECODE_START label=${label}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_before_decode.csv" 2>/dev/null || true
  if ! python3 "${BENCH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${served}" \
    --skip-prefill \
    --contexts "${DECODE_CONTEXTS}" \
    --concurrency "${DECODE_CONCURRENCY}" \
    --duration "${DECODE_DURATION}" \
    --max-tokens "${DECODE_MAX_TOKENS}" \
    --no-hw-monitor \
    --output "${out}/decode.json" > "${out}/decode.log" 2>&1; then
    progress "GLM52_V14_V5_DECODE_FAILED label=${label} out=${out}/decode.log"
    return 1
  fi
  if [[ ! -s "${out}/decode.json" ]]; then
    progress "GLM52_V14_V5_DECODE_FAILED label=${label} missing=${out}/decode.json"
    return 1
  fi
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_after_decode.csv" 2>/dev/null || true
  progress "GLM52_V14_V5_DECODE_DONE label=${label} out=${out}/decode.json"
}

run_prefill() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "GLM52_V14_V5_PREFILL_START label=${label}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_before_prefill.csv" 2>/dev/null || true
  if ! python3 "${BENCH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${served}" \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "${PREFILL_CONTEXTS}" \
    --prefill-duration "${PREFILL_DURATION}" \
    --max-tokens 1 \
    --no-hw-monitor \
    --output "${out}/prefill.json" > "${out}/prefill.log" 2>&1; then
    progress "GLM52_V14_V5_PREFILL_FAILED label=${label} out=${out}/prefill.log"
    return 1
  fi
  if [[ ! -s "${out}/prefill.json" ]]; then
    progress "GLM52_V14_V5_PREFILL_FAILED label=${label} missing=${out}/prefill.json"
    return 1
  fi
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_after_prefill.csv" 2>/dev/null || true
  progress "GLM52_V14_V5_PREFILL_DONE label=${label} out=${out}/prefill.json"
}

parse_kv() {
  local out="$1"
  python3 - "$out/server.ready.log" > "$out/kv_cache_summary.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], errors="replace").read() if len(sys.argv) > 1 else ""
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
        val = m.group(1).replace(",", "")
        out[key] = float(val) if "." in val else int(val)
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

bench_case() {
  local label="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6" dcp="$7" mtp="$8" moe_mode="$9" quantization="${10}" online_quant="${11}" max_model_len="${12}" max_num_seqs="${13}" max_batched_tokens="${14}" graph="${15}" gpu_mem="${16}" out="${17}"
  start_with_helper "$label" "$model" "$served" "$port" "$gpus" "$tp" "$dcp" "$mtp" "$moe_mode" "$quantization" "$online_quant" "$max_model_len" "$max_num_seqs" "$max_batched_tokens" "$graph" "$gpu_mem" "$out"
}

wait_pair_then_bench() {
  local out_a="$1" out_b="$2"
  local name_a name_b port_a port_b served_a served_b
  name_a="$(cat "${out_a}/container.name")"
  name_b="$(cat "${out_b}/container.name")"
  port_a="$(cat "${out_a}/port")"
  port_b="$(cat "${out_b}/port")"
  served_a="$(cat "${out_a}/served_model.name")"
  served_b="$(cat "${out_b}/served_model.name")"
  wait_ready "${name_a}" "${port_a}" "${out_a}" &
  local pid_a=$!
  wait_ready "${name_b}" "${port_b}" "${out_b}" &
  local pid_b=$!
  local ok_a=0 ok_b=0
  wait "${pid_a}" || ok_a=$?
  wait "${pid_b}" || ok_b=$?
  if (( ok_a != 0 || ok_b != 0 )); then
    progress "GLM52_V14_V5_PAIR_FAILED out_a=${out_a} status_a=${ok_a} out_b=${out_b} status_b=${ok_b}"
    return 1
  fi
  progress "GLM52_V14_V5_SETTLE seconds=${SETTLE_SECONDS} cases=${out_a},${out_b}"
  sleep "${SETTLE_SECONDS}"
  parse_kv "${out_a}"
  parse_kv "${out_b}"
  run_decode "$(basename "${out_a}")" "${port_a}" "${served_a}" "${out_a}" &
  pid_a=$!
  run_decode "$(basename "${out_b}")" "${port_b}" "${served_b}" "${out_b}" &
  pid_b=$!
  ok_a=0
  ok_b=0
  wait "${pid_a}" || ok_a=$?
  wait "${pid_b}" || ok_b=$?
  if (( ok_a != 0 || ok_b != 0 )); then
    progress "GLM52_V14_V5_DECODE_PAIR_FAILED out_a=${out_a} status_a=${ok_a} out_b=${out_b} status_b=${ok_b}"
    return 1
  fi
  run_prefill "$(basename "${out_a}")" "${port_a}" "${served_a}" "${out_a}" &
  pid_a=$!
  run_prefill "$(basename "${out_b}")" "${port_b}" "${served_b}" "${out_b}" &
  pid_b=$!
  ok_a=0
  ok_b=0
  wait "${pid_a}" || ok_a=$?
  wait "${pid_b}" || ok_b=$?
  if (( ok_a != 0 || ok_b != 0 )); then
    progress "GLM52_V14_V5_PREFILL_PAIR_FAILED out_a=${out_a} status_a=${ok_a} out_b=${out_b} status_b=${ok_b}"
    return 1
  fi
}

wait_pair_then_decode() {
  local out_a="$1" out_b="$2"
  local name_a name_b port_a port_b served_a served_b
  name_a="$(cat "${out_a}/container.name")"
  name_b="$(cat "${out_b}/container.name")"
  port_a="$(cat "${out_a}/port")"
  port_b="$(cat "${out_b}/port")"
  served_a="$(cat "${out_a}/served_model.name")"
  served_b="$(cat "${out_b}/served_model.name")"
  wait_ready "${name_a}" "${port_a}" "${out_a}" &
  local pid_a=$!
  wait_ready "${name_b}" "${port_b}" "${out_b}" &
  local pid_b=$!
  local ok_a=0 ok_b=0
  wait "${pid_a}" || ok_a=$?
  wait "${pid_b}" || ok_b=$?
  if (( ok_a != 0 || ok_b != 0 )); then
    progress "GLM52_V14_V5_PAIR_FAILED out_a=${out_a} status_a=${ok_a} out_b=${out_b} status_b=${ok_b}"
    return 1
  fi
  progress "GLM52_V14_V5_SETTLE seconds=${SETTLE_SECONDS} cases=${out_a},${out_b}"
  sleep "${SETTLE_SECONDS}"
  parse_kv "${out_a}"
  parse_kv "${out_b}"
  run_decode "$(basename "${out_a}")" "${port_a}" "${served_a}" "${out_a}" &
  pid_a=$!
  run_decode "$(basename "${out_b}")" "${port_b}" "${served_b}" "${out_b}" &
  pid_b=$!
  ok_a=0
  ok_b=0
  wait "${pid_a}" || ok_a=$?
  wait "${pid_b}" || ok_b=$?
  if (( ok_a != 0 || ok_b != 0 )); then
    progress "GLM52_V14_V5_DECODE_PAIR_FAILED out_a=${out_a} status_a=${ok_a} out_b=${out_b} status_b=${ok_b}"
    return 1
  fi
}

wait_single_then_bench() {
  local out="$1"
  local name port served
  name="$(cat "${out}/container.name")"
  port="$(cat "${out}/port")"
  served="$(cat "${out}/served_model.name")"
  if ! wait_ready "${name}" "${port}" "${out}"; then
    progress "GLM52_V14_V5_SINGLE_FAILED out=${out}"
    return 1
  fi
  progress "GLM52_V14_V5_SETTLE seconds=${SETTLE_SECONDS} cases=${out}"
  sleep "${SETTLE_SECONDS}"
  parse_kv "${out}"
  run_decode "$(basename "${out}")" "${port}" "${served}" "${out}"
  run_prefill "$(basename "${out}")" "${port}" "${served}" "${out}"
}

tp6_mxfp4() {
  local base="${RESULT_ROOT}/tp6-mxfp4-online-mxfp8"
  local dcp_a dcp_b out_a out_b
  stop_all_glm52_v5
  out_a="${base}/dcp1"
  bench_case "tp6-mxfp4-dcp1" "${MXFP4_MODEL}" "GLM-5.2-BF16-AMDMXFP4experts-online-mxfp8-tp6-dcp1" 5910 "0,1,2,3,4,5" 6 1 3 "force-a8-experimental" "mxfp4" "mxfp8" 262144 16 4096 64 0.98 "${out_a}"
  wait_single_then_bench "${out_a}" || true
  stop_case "${out_a}"

  for pair in "2 3"; do
    set -- ${pair}
    dcp_a="$1"
    dcp_b="$2"
    stop_all_glm52_v5
    out_a="${base}/dcp${dcp_a}"
    out_b="${base}/dcp${dcp_b}"
    bench_case "tp6-mxfp4-dcp${dcp_a}" "${MXFP4_MODEL}" "GLM-5.2-BF16-AMDMXFP4experts-online-mxfp8-tp6-dcp${dcp_a}" 5910 "0,1,2,3,4,5" 6 "${dcp_a}" 3 "force-a8-experimental" "mxfp4" "mxfp8" 262144 16 4096 64 0.98 "${out_a}"
    bench_case "tp6-mxfp4-dcp${dcp_b}" "${MXFP4_MODEL}" "GLM-5.2-BF16-AMDMXFP4experts-online-mxfp8-tp6-dcp${dcp_b}" 5911 "8,9,10,11,12,13" 6 "${dcp_b}" 3 "force-a8-experimental" "mxfp4" "mxfp8" 262144 16 4096 64 0.98 "${out_b}"
    wait_pair_then_bench "${out_a}" "${out_b}" || true
  done
  stop_all_glm52_v5
  out_a="${base}/dcp6"
  bench_case "tp6-mxfp4-dcp6" "${MXFP4_MODEL}" "GLM-5.2-BF16-AMDMXFP4experts-online-mxfp8-tp6-dcp6" 5910 "0,1,2,3,4,5" 6 6 3 "force-a8-experimental" "mxfp4" "mxfp8" 262144 16 4096 64 0.98 "${out_a}"
  wait_single_then_bench "${out_a}" || true
  stop_case "${out_a}"
}

tp8_decode() {
  local base="${RESULT_ROOT}/tp8-nvfp4-a16-mtp3-decode"
  local dcp_a dcp_b out_a out_b
  for pair in "2 4" "8"; do
    set -- ${pair}
    dcp_a="$1"
    dcp_b="${2:-}"
    stop_all_glm52_v5
    out_a="${base}/dcp${dcp_a}"
    bench_case "tp8-nvfp4-dcp${dcp_a}" "${NVFP4_MODEL}" "GLM-5.2-NVFP4-a16-mtp3-tp8-dcp${dcp_a}" 5920 "0,1,2,3,4,5,6,7" 8 "${dcp_a}" 3 "a16" "modelopt_fp4" "none" 131072 32 8192 128 0.90 "${out_a}"
    if [[ -n "${dcp_b}" ]]; then
      out_b="${base}/dcp${dcp_b}"
      bench_case "tp8-nvfp4-dcp${dcp_b}" "${NVFP4_MODEL}" "GLM-5.2-NVFP4-a16-mtp3-tp8-dcp${dcp_b}" 5921 "8,9,10,11,12,13,14,15" 8 "${dcp_b}" 3 "a16" "modelopt_fp4" "none" 131072 32 8192 128 0.90 "${out_b}"
      wait_pair_then_decode "${out_a}" "${out_b}" || true
    else
      if ! wait_ready "$(cat "${out_a}/container.name")" 5920 "${out_a}"; then
        progress "GLM52_V14_V5_SINGLE_FAILED out=${out_a}"
        continue
      fi
      sleep "${SETTLE_SECONDS}"
      parse_kv "${out_a}"
      run_decode "tp8-nvfp4-dcp${dcp_a}" 5920 "GLM-5.2-NVFP4-a16-mtp3-tp8-dcp${dcp_a}" "${out_a}"
      stop_case "${out_a}"
    fi
  done
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

def decode_values(path):
    data = load(path)
    rows = {}
    for row in data.get("results", []):
        cc = row.get("concurrency")
        if cc is None:
            continue
        rows[int(cc)] = row.get("aggregate_tps", row.get("aggregate_output_tokens_per_second"))
    return rows

def prefill_values(path):
    data = load(path)
    pref = data.get("prefill") or {}
    out = {}
    if isinstance(pref, dict):
        for ctx, row in pref.items():
            out[str(ctx)] = row.get("tok_per_sec") if isinstance(row, dict) else None
    return out

def kv(path):
    return load(path / "kv_cache_summary.json")

for path in sorted(root.glob("**/decode.json")):
    case = path.parent.relative_to(root)
    print(case)
    print("  decode", decode_values(path))
    pref = prefill_values(path.with_name("prefill.json"))
    if pref:
        print("  prefill", pref)
    k = kv(path.parent)
    if k:
        print("  kv", k)
PY
}

usage() {
  cat <<EOF
usage: $0 [tp6-mxfp4|tp8-decode|all|summarize|stop]

RESULT_ROOT=${RESULT_ROOT}
EOF
}

case "${1:-all}" in
  tp6-mxfp4) tp6_mxfp4; summarize ;;
  tp8-decode) tp8_decode; summarize ;;
  all) tp6_mxfp4; tp8_decode; summarize ;;
  summarize) summarize ;;
  stop) stop_all_glm52_v5 ;;
  *) usage; exit 2 ;;
esac
