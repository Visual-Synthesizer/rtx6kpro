#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER=${LAUNCHER:-$SCRIPT_DIR/run-ds4-v8-server.sh}
BENCH=${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}
OUT=${OUT:-/root/bench-results/ds4-v8-$(date -u +%Y%m%d-%H%M%S)}
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629}

TPS=${TPS:-2,4}
BACKENDS=${BACKENDS:-b12x,lucifer-default,lucifer-cutlass}
MODES=${MODES:-standard-mtp0,standard-mtp2,dspark}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-1,16,32,64}
DECODE_CONTEXTS=${DECODE_CONTEXTS:-0}
DECODE_DURATION=${DECODE_DURATION:-30}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-8192}
DECODE_TOKEN_BUDGET=${DECODE_TOKEN_BUDGET:-2000000}
PREFILL_CONTEXTS=${PREFILL_CONTEXTS:-8k,64k,128k}
PREFILL_DURATION=${PREFILL_DURATION:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
PORT_BASE=${PORT_BASE:-6100}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}

mkdir -p "$OUT"

split_csv() {
  local input=$1
  input=${input//,/ }
  printf '%s\n' $input
}

gpu_groups_for_tp() {
  case "$1" in
    2) printf '%s\n' 0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 ;;
    4) printf '%s\n' 0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 ;;
    *) echo "Unsupported TP=$1" >&2; return 2 ;;
  esac
}

model_name_for_mode() {
  case "$1" in
    standard-*) printf 'DeepSeek-V4-Flash\n' ;;
    dspark) printf 'DeepSeek-V4-Flash-DSpark\n' ;;
    *) return 2 ;;
  esac
}

wait_for_server() {
  local name=$1
  local port=$2
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      docker logs --tail 240 "$name" || true
      return 1
    fi
    sleep 5
  done
  docker logs --tail 300 "$name" || true
  echo "Timed out waiting for $name on port $port" >&2
  return 1
}

run_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name
  label="tp${tp}-${backend}-${mode}"
  name="ds4-v8-${label}"
  case_dir="$OUT/$label"
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  echo "==> launch $label on GPUs=$gpus port=$port"
  IMAGE="$IMAGE" \
  NAME="$name" \
  PORT="$port" \
  GPUS="$gpus" \
  TP="$tp" \
  BACKEND="$backend" \
  MODE="$mode" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  CACHE="$case_dir/cache" \
  CONTAINER_TMP="$case_dir/tmp" \
  "$LAUNCHER" 2>&1 | tee "$case_dir/launch.log"

  wait_for_server "$name" "$port"
  curl -fsS "http://127.0.0.1:${port}/version" > "$case_dir/version.json" || true
  curl -fsS "http://127.0.0.1:${port}/v1/models" > "$case_dir/models.json"

  echo "==> decode $label"
  python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency "$DECODE_CONCURRENCY" \
    --contexts "$DECODE_CONTEXTS" \
    --duration "$DECODE_DURATION" \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --skip-prefill \
    --display-mode plain \
    --no-hw-monitor \
    --coding-peak \
    --coding-peak-runs 5 \
    --output "$case_dir/decode.json" \
    2>&1 | tee "$case_dir/decode.log"

  echo "==> prefill $label"
  python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency 1 \
    --contexts 0 \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "$PREFILL_CONTEXTS" \
    --prefill-duration "$PREFILL_DURATION" \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --display-mode plain \
    --no-hw-monitor \
    --output "$case_dir/prefill.json" \
    2>&1 | tee "$case_dir/prefill.log"

  curl -fsS "http://127.0.0.1:${port}/metrics" > "$case_dir/final-metrics.prom" || true
  docker logs "$name" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "DONE $label"
}

run_tp_matrix() {
  local tp=$1
  local -a groups cases wave_pids
  mapfile -t groups < <(gpu_groups_for_tp "$tp")
  cases=()
  for backend in $(split_csv "$BACKENDS"); do
    for mode in $(split_csv "$MODES"); do
      cases+=("$tp:$backend:$mode")
    done
  done

  local idx=0 wave=0 failures=0
  while (( idx < ${#cases[@]} )); do
    wave_pids=()
    local slot=0
    while (( slot < ${#groups[@]} && idx < ${#cases[@]} )); do
      IFS=: read -r c_tp c_backend c_mode <<<"${cases[$idx]}"
      local port=$((PORT_BASE + tp * 100 + wave * 20 + slot))
      run_case "$c_tp" "$c_backend" "$c_mode" "${groups[$slot]}" "$port" &
      wave_pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done
    for pid in "${wave_pids[@]}"; do
      if ! wait "$pid"; then
        failures=$((failures + 1))
      fi
    done
    wave=$((wave + 1))
  done

  if (( failures > 0 )); then
    echo "TP${tp} completed with $failures failed case(s)" >&2
    return 1
  fi
}

chmod +x "$LAUNCHER"
printf 'image=%s\nout=%s\nmax_num_seqs=%s\ndecode_concurrency=%s\n' \
  "$IMAGE" "$OUT" "$MAX_NUM_SEQS" "$DECODE_CONCURRENCY" | tee "$OUT/run-config.txt"
for tp in $(split_csv "$TPS"); do
  run_tp_matrix "$tp"
done
