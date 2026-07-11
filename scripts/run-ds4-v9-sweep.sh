#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SWEEP_SCRIPT=${SWEEP_SCRIPT:-$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")}
LAUNCHER=${LAUNCHER:-$SCRIPT_DIR/run-ds4-v9-server.sh}
BENCH=${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}
OUT=${OUT:-/root/bench-results/ds4-v9-$(date -u +%Y%m%d-%H%M%S)}
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704}
PROGRESS_FILE=${PROGRESS_FILE:-/root/vllm/prubezne_vysledky}

TPS=${TPS:-2,4}
BACKENDS=${BACKENDS:-b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass}
MODES=${MODES:-standard-mtp0,standard-mtp2,standard-mtp3,dspark}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-1,16,32,64}
DECODE_CONTEXTS=${DECODE_CONTEXTS:-0}
DECODE_DURATION=${DECODE_DURATION:-30}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-8192}
DECODE_TOKEN_BUDGET=${DECODE_TOKEN_BUDGET:-2000000}
PREFILL_CONTEXTS=${PREFILL_CONTEXTS:-8k,64k,128k}
PREFILL_DURATION=${PREFILL_DURATION:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
PORT_BASE=${PORT_BASE:-7100}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
SYNC_WAVE_READY=${SYNC_WAVE_READY:-1}
ENABLE_TOPO_PIN=${ENABLE_TOPO_PIN:-1}
POST_READY_SETTLE_SECONDS=${POST_READY_SETTLE_SECONDS:-30}
RUNTIME_WARMUP=${RUNTIME_WARMUP:-1}
RUNTIME_WARMUP_DECODE_DURATION=${RUNTIME_WARMUP_DECODE_DURATION:-3}
RUNTIME_WARMUP_PREFILL_DURATION=${RUNTIME_WARMUP_PREFILL_DURATION:-1}
POST_WARMUP_SETTLE_SECONDS=${POST_WARMUP_SETTLE_SECONDS:-30}
RESUME=${RESUME:-1}
VLLM_PATCH_FILE=${VLLM_PATCH_FILE:-/root/vllm/blackwell-llm-docker/patches/vllm-b12x-indexer-warmup-fallback-20260704.patch}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-ds4-v9}
SHARED_CACHE=${SHARED_CACHE:-}
GPU_GROUPS_TP2=${GPU_GROUPS_TP2:-"0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15"}
GPU_GROUPS_TP4=${GPU_GROUPS_TP4:-"0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15"}

mkdir -p "$OUT"

record_repro_artifacts() {
  local -a script_hash_inputs
  mkdir -p "$OUT/repro"
  cp "$LAUNCHER" "$OUT/repro/$(basename "$LAUNCHER")"
  cp "$SWEEP_SCRIPT" "$OUT/repro/$(basename "$SWEEP_SCRIPT")"
  script_hash_inputs=("$LAUNCHER" "$SWEEP_SCRIPT")
  if [[ -f "$SCRIPT_DIR/render-ds4-v9-results.py" ]]; then
    cp "$SCRIPT_DIR/render-ds4-v9-results.py" "$OUT/repro/render-ds4-v9-results.py"
    script_hash_inputs+=("$SCRIPT_DIR/render-ds4-v9-results.py")
  fi
  if [[ -f "$VLLM_PATCH_FILE" ]]; then
    cp "$VLLM_PATCH_FILE" "$OUT/repro/$(basename "$VLLM_PATCH_FILE")"
    sha256sum "$VLLM_PATCH_FILE" > "$OUT/repro/vllm-patch.sha256" || true
  fi
  if [[ -f "$BENCH" ]]; then
    cp "$BENCH" "$OUT/repro/$(basename "$BENCH")"
    sha256sum "$BENCH" > "$OUT/repro/bench.sha256" || true
  fi
  sha256sum "${script_hash_inputs[@]}" > "$OUT/repro/scripts.sha256" || true
  if git -C "$SCRIPT_DIR/.." rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$SCRIPT_DIR/.." rev-parse HEAD > "$OUT/repro/rtx6kpro-head.txt" || true
    git -C "$SCRIPT_DIR/.." status --short > "$OUT/repro/rtx6kpro-status.txt" || true
  fi
  docker image inspect "$IMAGE" > "$OUT/repro/image-inspect.json" 2>/dev/null || true
  docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' > "$OUT/repro/image-labels.json" 2>/dev/null || true
  nvidia-smi topo -m > "$OUT/repro/nvidia-topo.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit --format=csv > "$OUT/repro/nvidia-gpus.csv" 2>/dev/null || true
}

on_exit() {
  local status=$?
  printf '%s sweep_process_exit status=%s out=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$OUT" >> "$PROGRESS_FILE"
}
trap on_exit EXIT

split_csv() {
  local input=$1
  input=${input//,/ }
  printf '%s\n' $input
}

gpu_groups_for_tp() {
  case "$1" in
    2) printf '%s\n' $GPU_GROUPS_TP2 ;;
    4) printf '%s\n' $GPU_GROUPS_TP4 ;;
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

append_case_summary() {
  local status=$1 label=$2 case_dir=$3
  python3 - "$status" "$label" "$case_dir" "$PROGRESS_FILE" <<'PY'
import datetime as dt
import json
import math
import pathlib
import sys

status, label, case_dir, progress_file = sys.argv[1:]
case_path = pathlib.Path(case_dir)
now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def load(name):
    path = case_path / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)

def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.1f}"

decode = load("decode.json")
prefill = load("prefill.json")

cc = {}
for row in decode.get("results", []):
    if int(row.get("context_tokens", -1)) == 0:
        cc[int(row.get("concurrency", 0))] = row.get("aggregate_tps")

coding_summary = decode.get("coding_peak", {}).get("summary", {})
pref = prefill.get("prefill", {})

line = (
    f"{now} {status} {label} "
    f"decode cc1={fmt(cc.get(1))} cc16={fmt(cc.get(16))} "
    f"cc32={fmt(cc.get(32))} cc64={fmt(cc.get(64))} "
    f"coding_median={fmt(coding_summary.get('median_generation_tok_s'))} "
    f"cjk={coding_summary.get('cjk_runs', 'NA')} "
    f"prefill 8k={fmt(pref.get('8192', {}).get('tok_per_sec'))} "
    f"64k={fmt(pref.get('65536', {}).get('tok_per_sec'))} "
    f"128k={fmt(pref.get('131072', {}).get('tok_per_sec'))} "
    f"dir={case_path}"
)
with open(progress_file, "a", encoding="utf-8") as f:
    f.write(line + "\n")
print(line)
PY
}

validate_case_results() {
  local case_dir=$1
  python3 - "$case_dir" "$DECODE_CONCURRENCY" "$PREFILL_CONTEXTS" <<'PY'
import json
import math
import pathlib
import sys

case_dir = pathlib.Path(sys.argv[1])
decode_concurrency = [int(x) for x in sys.argv[2].replace(",", " ").split()]
prefill_contexts = []
for item in sys.argv[3].replace(",", " ").split():
    item = item.lower()
    if item.endswith("k"):
        prefill_contexts.append(str(int(float(item[:-1]) * 1024)))
    else:
        prefill_contexts.append(str(int(item)))

def fail(msg: str) -> None:
    print(f"invalid benchmark results: {msg}", file=sys.stderr)
    raise SystemExit(1)

def load(name: str):
    path = case_dir / name
    if not path.exists():
        fail(f"missing {name}")
    with path.open() as f:
        return json.load(f)

def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

decode = load("decode.json")
rows = {}
for row in decode.get("results", []):
    try:
        context = int(row.get("context_tokens", -1))
        concurrency = int(row.get("concurrency", 0))
    except (TypeError, ValueError):
        continue
    if context == 0 and finite(row.get("aggregate_tps")):
        rows[concurrency] = row["aggregate_tps"]

missing_decode = [cc for cc in decode_concurrency if cc not in rows]
if missing_decode:
    fail(f"missing decode aggregate_tps for concurrency {missing_decode}")

coding = decode.get("coding_peak", {})
coding_summary = coding.get("summary", {})
if int(coding.get("runs_ok", -1)) != int(coding.get("runs_requested", -2)):
    fail(
        "incomplete coding peak: "
        f"{coding.get('runs_ok')}/{coding.get('runs_requested')} runs"
    )
if not finite(coding_summary.get("median_generation_tok_s")):
    fail("missing coding peak median_generation_tok_s")
if int(coding_summary.get("cjk_runs", -1)) != 0:
    fail(f"coding peak produced CJK in {coding_summary.get('cjk_runs')} run(s)")

prefill = load("prefill.json")
prefill_rows = prefill.get("prefill", {})

def valid_prefill(row) -> bool:
    if not isinstance(row, dict):
        return False
    if not finite(row.get("tok_per_sec")) or not finite(row.get("ttft_seconds")):
        return False
    # A failed streaming request used to be timed as an immediate TTFT and
    # could yield multi-million tok/s. Keep resume fail-closed for those old
    # result files as well as for empty samples.
    return (
        0 < float(row["tok_per_sec"]) < 1_000_000
        and float(row["ttft_seconds"]) > 0
        and int(row.get("samples", 0)) > 0
    )

missing_prefill = [
    ctx for ctx in prefill_contexts
    if ctx not in prefill_rows or not valid_prefill(prefill_rows[ctx])
]
if missing_prefill:
    fail(f"missing prefill tok_per_sec for contexts {missing_prefill}")
PY
}

validate_runtime_log() {
  local server_log=$1
  local start_line=${2:-0}
  local first_measured_line=$((start_line + 1))
  if tail -n "+${first_measured_line}" "$server_log" | rg -q \
    'JIT compilation during inference|reason=post-engine-start.*status=disk-cache-miss'; then
    echo "invalid benchmark runtime: JIT cache miss occurred during inference" >&2
    tail -n "+${first_measured_line}" "$server_log" | rg -n \
      'JIT compilation during inference|reason=post-engine-start.*status=disk-cache-miss' \
      >&2 || true
    return 1
  fi
}

validate_reusable_case() {
  local case_dir=$1
  local start_file="$case_dir/runtime-log-start-line.txt"
  [[ -f "$case_dir/server.log" && -f "$start_file" ]] || return 1
  validate_case_results "$case_dir" >/dev/null 2>&1 || return 1
  validate_runtime_log "$case_dir/server.log" "$(<"$start_file")" \
    >/dev/null 2>&1
}

settle_after_ready() {
  if (( POST_READY_SETTLE_SECONDS > 0 )); then
    echo "==> post-ready settle ${POST_READY_SETTLE_SECONDS}s"
    sleep "$POST_READY_SETTLE_SECONDS"
  fi
}

warm_runtime_case() {
  local port=$1 model_name=$2 case_dir=$3
  if [[ "$RUNTIME_WARMUP" != "1" ]]; then
    return 0
  fi

  echo "==> runtime warmup decode ${DECODE_CONCURRENCY}"
  python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency "$DECODE_CONCURRENCY" \
    --contexts "$DECODE_CONTEXTS" \
    --duration "$RUNTIME_WARMUP_DECODE_DURATION" \
    --decode-warmup-seconds 0 \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --skip-prefill \
    --display-mode plain \
    --no-hw-monitor \
    --output "$case_dir/warmup-decode.json" \
    > "$case_dir/warmup-decode.log" 2>&1

  echo "==> runtime warmup prefill ${PREFILL_CONTEXTS}"
  python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency 1 \
    --contexts 0 \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "$PREFILL_CONTEXTS" \
    --prefill-duration "$RUNTIME_WARMUP_PREFILL_DURATION" \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --display-mode plain \
    --no-hw-monitor \
    --output "$case_dir/warmup-prefill.json" \
    > "$case_dir/warmup-prefill.log" 2>&1

  if (( POST_WARMUP_SETTLE_SECONDS > 0 )); then
    echo "==> post-warmup settle ${POST_WARMUP_SETTLE_SECONDS}s"
    sleep "$POST_WARMUP_SETTLE_SECONDS"
  fi
}

launch_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name cache_dir
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  cache_dir=${SHARED_CACHE:-$case_dir/cache}
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  echo "==> launch $label on GPUs=$gpus port=$port"
  if ! IMAGE="$IMAGE" \
    NAME="$name" \
    PORT="$port" \
    GPUS="$gpus" \
    TP="$tp" \
    BACKEND="$backend" \
    MODE="$mode" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    ENABLE_TOPO_PIN="$ENABLE_TOPO_PIN" \
    CACHE="$cache_dir" \
    CONTAINER_TMP="$case_dir/tmp" \
    "$LAUNCHER" 2>&1 | tee "$case_dir/launch.log"; then
    return 1
  fi
}

bench_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  curl -fsS "http://127.0.0.1:${port}/version" > "$case_dir/version.json" || true
  curl -fsS "http://127.0.0.1:${port}/v1/models" > "$case_dir/models.json" || return 1

  warm_runtime_case "$port" "$model_name" "$case_dir" || return 1
  docker logs "$name" > "$case_dir/warmup-server.log" 2>&1 || true
  wc -l < "$case_dir/warmup-server.log" > "$case_dir/runtime-log-start-line.txt"

  echo "==> decode $label"
  if ! python3 "$BENCH" \
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
    2>&1 | tee "$case_dir/decode.log"; then
    return 1
  fi

  echo "==> prefill $label"
  if ! python3 "$BENCH" \
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
    2>&1 | tee "$case_dir/prefill.log"; then
    return 1
  fi

  curl -fsS "http://127.0.0.1:${port}/metrics" > "$case_dir/final-metrics.prom" || true
  docker logs "$name" > "$case_dir/server.log" 2>&1 || true
  validate_case_results "$case_dir" || return 1
  validate_runtime_log \
    "$case_dir/server.log" \
    "$(<"$case_dir/runtime-log-start-line.txt")" || return 1
  docker rm -f "$name" >/dev/null 2>&1 || true
  append_case_summary DONE "$label" "$case_dir"
}

fail_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  mkdir -p "$case_dir"
  docker logs "$name" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  append_case_summary FAILED "$label" "$case_dir" || true
}

run_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  launch_case "$tp" "$backend" "$mode" "$gpus" "$port" || return 1
  wait_for_server "$name" "$port" || return 1
  settle_after_ready
  bench_case "$tp" "$backend" "$mode" "$gpus" "$port"
}

run_case_guarded() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  if run_case "$tp" "$backend" "$mode" "$gpus" "$port"; then
    return 0
  fi
  fail_case "$tp" "$backend" "$mode" "$gpus" "$port"
  return 1
}

run_tp_matrix() {
  local tp=$1
  local -a groups cases wave_pids
  mapfile -t groups < <(gpu_groups_for_tp "$tp")
  cases=()
  for backend in $(split_csv "$BACKENDS"); do
    for mode in $(split_csv "$MODES"); do
      local label="tp${tp}-${backend}-${mode}"
      if [[ "$RESUME" == "1" ]] \
        && validate_reusable_case "$OUT/$label"; then
        append_case_summary REUSED "$label" "$OUT/$label"
      else
        cases+=("$tp:$backend:$mode")
      fi
    done
  done

  local idx=0 wave=0 failures=0
  while (( idx < ${#cases[@]} )); do
    wave_pids=()
    local -a ready_cases bench_pids
    ready_cases=()
    bench_pids=()
    local slot=0
    while (( slot < ${#groups[@]} && idx < ${#cases[@]} )); do
      IFS=: read -r c_tp c_backend c_mode <<<"${cases[$idx]}"
      local port=$((PORT_BASE + tp * 100 + wave * 20 + slot))
      if [[ "$SYNC_WAVE_READY" == "1" ]]; then
        local spec="$c_tp:$c_backend:$c_mode:${groups[$slot]}:$port"
        if launch_case "$c_tp" "$c_backend" "$c_mode" "${groups[$slot]}" "$port" \
          && wait_for_server \
            "${CONTAINER_PREFIX}-tp${c_tp}-${c_backend}-${c_mode}" "$port"; then
          # Load and fully initialize each server before starting the next one.
          # Benchmark clients are launched only after this whole wave is ready.
          ready_cases+=("$spec")
        else
          fail_case "$c_tp" "$c_backend" "$c_mode" "${groups[$slot]}" "$port"
          failures=$((failures + 1))
        fi
      else
        run_case_guarded "$c_tp" "$c_backend" "$c_mode" "${groups[$slot]}" "$port" &
        wave_pids+=("$!")
      fi
      idx=$((idx + 1))
      slot=$((slot + 1))
    done
    if [[ "$SYNC_WAVE_READY" == "1" ]]; then
      settle_after_ready
      for spec in "${ready_cases[@]}"; do
        IFS=: read -r c_tp c_backend c_mode c_gpus c_port <<<"$spec"
        bench_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port" &
        bench_pids+=("$!")
      done
      for i in "${!bench_pids[@]}"; do
        IFS=: read -r c_tp c_backend c_mode c_gpus c_port <<<"${ready_cases[$i]}"
        if ! wait "${bench_pids[$i]}"; then
          fail_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port"
          failures=$((failures + 1))
        fi
      done
    else
      for pid in "${wave_pids[@]}"; do
        if ! wait "$pid"; then
          failures=$((failures + 1))
        fi
      done
    fi
    wave=$((wave + 1))
  done

  if (( failures > 0 )); then
    echo "TP${tp} completed with $failures failed case(s)" >&2
    return 1
  fi
}

chmod +x "$LAUNCHER"
record_repro_artifacts
{
  printf 'image=%s\nout=%s\nprogress_file=%s\nlauncher=%s\nbench=%s\n' \
    "$IMAGE" "$OUT" "$PROGRESS_FILE" "$LAUNCHER" "$BENCH"
  printf 'max_num_seqs=%s\nport_base=%s\nstartup_timeout=%s\n' \
    "$MAX_NUM_SEQS" "$PORT_BASE" "$STARTUP_TIMEOUT"
  printf 'post_ready_settle_seconds=%s\n' "$POST_READY_SETTLE_SECONDS"
  printf 'runtime_warmup=%s\nruntime_warmup_decode_duration=%s\n' \
    "$RUNTIME_WARMUP" "$RUNTIME_WARMUP_DECODE_DURATION"
  printf 'runtime_warmup_prefill_duration=%s\npost_warmup_settle_seconds=%s\n' \
    "$RUNTIME_WARMUP_PREFILL_DURATION" "$POST_WARMUP_SETTLE_SECONDS"
  printf 'decode_concurrency=%s\ndecode_contexts=%s\ndecode_duration=%s\n' \
    "$DECODE_CONCURRENCY" "$DECODE_CONTEXTS" "$DECODE_DURATION"
  printf 'decode_max_tokens=%s\ndecode_token_budget=%s\n' \
    "$DECODE_MAX_TOKENS" "$DECODE_TOKEN_BUDGET"
  printf 'prefill_contexts=%s\nprefill_duration=%s\n' \
    "$PREFILL_CONTEXTS" "$PREFILL_DURATION"
  printf 'backends=%s\nmodes=%s\ntps=%s\nsync_wave_ready=%s\nenable_topo_pin=%s\nvllm_patch_file=%s\n' \
    "$BACKENDS" "$MODES" "$TPS" "$SYNC_WAVE_READY" "$ENABLE_TOPO_PIN" "$VLLM_PATCH_FILE"
  printf 'container_prefix=%s\ngpu_groups_tp2=%s\ngpu_groups_tp4=%s\nresume=%s\n' \
    "$CONTAINER_PREFIX" "$GPU_GROUPS_TP2" "$GPU_GROUPS_TP4" "$RESUME"
  printf 'shared_cache=%s\n' "${SHARED_CACHE:-disabled}"
} | tee "$OUT/run-config.txt"
printf '%s sweep_start image=%s out=%s backends=%s modes=%s tps=%s sync_wave_ready=%s enable_topo_pin=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$IMAGE" "$OUT" "$BACKENDS" "$MODES" "$TPS" "$SYNC_WAVE_READY" "$ENABLE_TOPO_PIN" \
  >> "$PROGRESS_FILE"
for tp in $(split_csv "$TPS"); do
  run_tp_matrix "$tp"
done
printf '%s sweep_done out=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUT" >> "$PROGRESS_FILE"
