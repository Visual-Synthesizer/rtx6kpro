#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER=${LAUNCHER:-$SCRIPT_DIR/run-glm52-mxfp4-server.sh}
BENCH=${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}
OUT=${OUT:-/root/bench-results/glm52-mxfp4-$(date -u +%Y%m%d-%H%M%S)}
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TP=${TP:-8}
DCP=${DCP:-1}
CASES=${CASES:-baseline,dspark5,dspark7}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-1,16,32,64}
DECODE_CONTEXTS=${DECODE_CONTEXTS:-0}
DECODE_DURATION=${DECODE_DURATION:-30}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-8192}
DECODE_TOKEN_BUDGET=${DECODE_TOKEN_BUDGET:-2000000}
PREFILL_CONTEXTS=${PREFILL_CONTEXTS:-8k,64k,128k}
PREFILL_DURATION=${PREFILL_DURATION:-10}
PORT=${PORT:-6600}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
MODEL_NAME=${MODEL_NAME:-GLM-5.2-FP8-MXFP4-Experts}

mkdir -p "$OUT"

split_csv() {
  local input=$1
  input=${input//,/ }
  printf '%s\n' $input
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
      docker logs --tail 260 "$name" || true
      return 1
    fi
    sleep 5
  done
  docker logs --tail 320 "$name" || true
  echo "Timed out waiting for $name on port $port" >&2
  return 1
}

case_to_mode() {
  case "$1" in
    baseline) printf 'baseline 0\n' ;;
    dspark5) printf 'dspark 5\n' ;;
    dspark7) printf 'dspark 7\n' ;;
    *) echo "Unknown case=$1" >&2; return 2 ;;
  esac
}

run_case() {
  local case_name=$1
  local mode tokens name case_dir
  read -r mode tokens < <(case_to_mode "$case_name")
  name="glm52-mxfp4-${case_name}"
  case_dir="$OUT/$case_name"
  mkdir -p "$case_dir"

  echo "==> launch $case_name on GPUs=$GPUS port=$PORT"
  IMAGE="$IMAGE" \
  NAME="$name" \
  PORT="$PORT" \
  GPUS="$GPUS" \
  TP="$TP" \
  DCP="$DCP" \
  MODE="$mode" \
  DSPARK_TOKENS="$tokens" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  CACHE="$case_dir/cache" \
  CONTAINER_TMP="$case_dir/tmp" \
  "$LAUNCHER" 2>&1 | tee "$case_dir/launch.log"

  wait_for_server "$name" "$PORT"
  curl -fsS "http://127.0.0.1:${PORT}/version" > "$case_dir/version.json" || true
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$case_dir/models.json"

  echo "==> decode $case_name"
  python3 "$BENCH" \
    --host localhost \
    --port "$PORT" \
    --model "$MODEL_NAME" \
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

  echo "==> prefill $case_name"
  python3 "$BENCH" \
    --host localhost \
    --port "$PORT" \
    --model "$MODEL_NAME" \
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

  curl -fsS "http://127.0.0.1:${PORT}/metrics" > "$case_dir/final-metrics.prom" || true
  docker logs "$name" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "DONE $case_name"
}

chmod +x "$LAUNCHER"
printf 'image=%s\nout=%s\ngpus=%s\ntp=%s\ndcp=%s\nmax_num_seqs=%s\ncases=%s\n' \
  "$IMAGE" "$OUT" "$GPUS" "$TP" "$DCP" "$MAX_NUM_SEQS" "$CASES" | tee "$OUT/run-config.txt"

for case_name in $(split_csv "$CASES"); do
  run_case "$case_name"
done
