#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${HELPER:-${ROOT_DIR}/scripts/run-glm52-v14-compose.sh}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v14-v7-tp8-hybrid-table-mtp0-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
OLD_NVFP4_ROOT="${OLD_NVFP4_ROOT:-/root/bench-results/glm52-v14-todo-and-sweep-20260706T0150Z/full}"
OLD_MXFP4_DCP1_ROOT="${OLD_MXFP4_DCP1_ROOT:-/root/bench-results/glm52-v14-v7-tp8-mxfp4-a8-dcp1-mtp0-20260707T133344Z}"

TP="${TP:-8}"
MTP="${MTP:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GRAPH="${GRAPH:-128}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DECODE_CONTEXTS="${DECODE_CONTEXTS:-0}"
DECODE_CONCURRENCY_FULL="${DECODE_CONCURRENCY_FULL:-1,32}"
DECODE_CONCURRENCY_CC32="${DECODE_CONCURRENCY_CC32:-32}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"

GPU_A="${GPU_A:-0,1,2,3,4,5,6,7}"
GPU_B="${GPU_B:-8,9,10,11,12,13,14,15}"
PORT_A="${PORT_A:-5960}"
PORT_B="${PORT_B:-5961}"

mkdir -p "${RESULT_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v14_v7_tp8_hybrid_table.out

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
  KLD_MEAN=""
  case "${key}" in
    nvfp4-a4-orig)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A4 orig"
      SHORT_NAME="nvfp4-a4-orig"
      MOE_MODE="a4"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="none"
      KLD_MEAN="0.10734"
      ;;
    nvfp4-a4-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A4 online MXFP8"
      SHORT_NAME="nvfp4-a4-online-mxfp8"
      MOE_MODE="a4"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="mxfp8"
      KLD_MEAN="0.10901"
      ;;
    nvfp4-a16-orig)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A16 orig"
      SHORT_NAME="nvfp4-a16-orig"
      MOE_MODE="a16"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="none"
      KLD_MEAN="0.06662"
      ;;
    nvfp4-a16-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"
      DISPLAY_NAME="Luke NVFP4 A16 online MXFP8"
      SHORT_NAME="nvfp4-a16-online-mxfp8"
      MOE_MODE="a16"
      QUANTIZATION="modelopt_fp4"
      ONLINE_QUANT="mxfp8"
      KLD_MEAN="0.07188"
      ;;
    mxfp4-a8-orig)
      MODEL_PATH="${MXFP4_MODEL}"
      DISPLAY_NAME="BF16 AMD MXFP4 experts A8 orig"
      SHORT_NAME="mxfp4-a8-orig"
      MOE_MODE="force-a8-experimental"
      QUANTIZATION="mxfp4"
      ONLINE_QUANT="none"
      KLD_MEAN="0.07610"
      ;;
    mxfp4-a8-online-mxfp8)
      MODEL_PATH="${MXFP4_MODEL}"
      DISPLAY_NAME="BF16 AMD MXFP4 experts A8 online MXFP8"
      SHORT_NAME="mxfp4-a8-online-mxfp8"
      MOE_MODE="force-a8-experimental"
      QUANTIZATION="mxfp4"
      ONLINE_QUANT="mxfp8"
      KLD_MEAN="0.07741"
      ;;
    *)
      echo "unknown case: ${key}" >&2
      return 2
      ;;
  esac
}

stop_own_containers() {
  docker ps -a --format '{{.Names}}' |
    awk '/^glm52-v14-hybrid-table-/ {print}' |
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
  local key="$1" dcp="$2" port="$3" gpus="$4" out="$5"
  case_vars "${key}"
  local name served
  name="glm52-v14-hybrid-table-$(safe_name "${SHORT_NAME}-dcp${dcp}-p${port}")"
  served="GLM-5.2-${SHORT_NAME}-tp8-dcp${dcp}-mtp0"
  mkdir -p "${out}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  progress "START case=${key} dcp=${dcp} name=${name} gpus=${gpus} port=${port}"
  (
    cd "${ROOT_DIR}"
    IMAGE="${IMAGE}" \
    MODEL="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${served}" \
    NAME="${name}" \
    COMPOSE_PROJECT_NAME="${name}" \
    PORT="${port}" \
    GPUS="${gpus}" \
    TP="${TP}" \
    DCP="${dcp}" \
    DCP_BACKEND=a2a \
    DCP_A2A_MAX_TOKENS=64 \
    DCP_A2A_LARGE_BACKEND=ag_rs \
    MTP="${MTP}" \
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
  printf '%s\n' "${port}" > "${out}/port"
  printf '%s\n' "${key}" > "${out}/case.key"
  printf '%s\n' "${DISPLAY_NAME}" > "${out}/display.name"
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

has_decode_cc() {
  local out="$1" cc="$2"
  python3 - "$out/decode.json" "$cc" <<'PY'
import json, sys
path, want = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
for row in data.get("results", []):
    if int(row.get("context_tokens", -1)) == 0 and int(row.get("concurrency", -1)) == want:
        if row.get("aggregate_tps") is not None or row.get("server_gen_throughput") is not None:
            sys.exit(0)
sys.exit(1)
PY
}

has_prefill_contexts() {
  local out="$1"
  python3 - "$out/prefill.json" <<'PY'
import json, sys
try:
    pref = json.load(open(sys.argv[1])).get("prefill", {})
except Exception:
    sys.exit(1)
for key in ("8192", "65536"):
    row = pref.get(key)
    if not isinstance(row, dict) or row.get("tok_per_sec") is None:
        sys.exit(1)
sys.exit(0)
PY
}

has_full_result() {
  local out="$1"
  has_decode_cc "${out}" 1 && has_decode_cc "${out}" 32 && has_prefill_contexts "${out}"
}

run_decode() {
  local label="$1" port="$2" served="$3" out="$4" conc="$5"
  progress "DECODE_START label=${label} conc=${conc}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_before_decode.csv" 2>/dev/null || true
  python3 "${BENCH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${served}" \
    --skip-prefill \
    --contexts "${DECODE_CONTEXTS}" \
    --concurrency "${conc}" \
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
  local key_a="$1" key_b="$2" dcp="$3" mode="$4"
  local out_a="${RESULT_ROOT}/${key_a}/dcp${dcp}"
  local out_b="${RESULT_ROOT}/${key_b}/dcp${dcp}"

  if [[ "${mode}" == "full" ]] && has_full_result "${out_a}" && has_full_result "${out_b}"; then
    progress "SKIP_DONE pair=${key_a},${key_b} dcp=${dcp} mode=${mode}"
    return 0
  fi
  if [[ "${mode}" == "cc32" ]] && has_decode_cc "${out_a}" 32 && has_decode_cc "${out_b}" 32; then
    progress "SKIP_DONE pair=${key_a},${key_b} dcp=${dcp} mode=${mode}"
    return 0
  fi

  stop_own_containers
  start_case "${key_a}" "${dcp}" "${PORT_A}" "${GPU_A}" "${out_a}"
  start_case "${key_b}" "${dcp}" "${PORT_B}" "${GPU_B}" "${out_b}"

  wait_ready "$(cat "${out_a}/container.name")" "${PORT_A}" "${out_a}" &
  local pid_a=$!
  wait_ready "$(cat "${out_b}/container.name")" "${PORT_B}" "${out_b}" &
  local pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"

  progress "SETTLE seconds=${SETTLE_SECONDS} pair=${key_a},${key_b} dcp=${dcp}"
  sleep "${SETTLE_SECONDS}"
  parse_kv "${out_a}"
  parse_kv "${out_b}"

  local served_a served_b conc
  served_a="$(cat "${out_a}/served_model.name")"
  served_b="$(cat "${out_b}/served_model.name")"
  if [[ "${mode}" == "cc32" ]]; then
    conc="${DECODE_CONCURRENCY_CC32}"
  else
    conc="${DECODE_CONCURRENCY_FULL}"
  fi

  run_decode "${key_a}-dcp${dcp}" "${PORT_A}" "${served_a}" "${out_a}" "${conc}" &
  pid_a=$!
  run_decode "${key_b}-dcp${dcp}" "${PORT_B}" "${served_b}" "${out_b}" "${conc}" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"

  if [[ "${mode}" == "full" ]]; then
    run_prefill "${key_a}-dcp${dcp}" "${PORT_A}" "${served_a}" "${out_a}" &
    pid_a=$!
    run_prefill "${key_b}-dcp${dcp}" "${PORT_B}" "${served_b}" "${out_b}" &
    pid_b=$!
    wait "${pid_a}"
    wait "${pid_b}"
  fi
}

run_missing() {
  progress "RUN_START result_root=${RESULT_ROOT} image=${IMAGE} mtp=${MTP} dcp_backend=a2a dcp_large=ag_rs"
  bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 1 cc32
  for dcp in 2 4 8; do
    bench_pair nvfp4-a4-orig nvfp4-a4-online-mxfp8 "${dcp}" full
    bench_pair nvfp4-a16-orig nvfp4-a16-online-mxfp8 "${dcp}" full
    bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 "${dcp}" full
  done
  stop_own_containers
  summarize
  progress "RUN_DONE result_root=${RESULT_ROOT}"
}

summarize() {
  python3 - "${RESULT_ROOT}" "${OLD_NVFP4_ROOT}" "${OLD_MXFP4_DCP1_ROOT}" <<'PY'
import json
import pathlib
import re
import sys

result_root = pathlib.Path(sys.argv[1])
old_nvfp4 = pathlib.Path(sys.argv[2])
old_mxfp4 = pathlib.Path(sys.argv[3])

cases = [
    ("nvfp4-a4-orig", "Luke NVFP4 A4 orig", "0.10734"),
    ("nvfp4-a4-online-mxfp8", "Luke NVFP4 A4 online MXFP8", "0.10901"),
    ("nvfp4-a16-orig", "Luke NVFP4 A16 orig", "0.06662"),
    ("nvfp4-a16-online-mxfp8", "Luke NVFP4 A16 online MXFP8", "0.07188"),
    ("mxfp4-a8-orig", "BF16 AMD MXFP4 experts A8 orig", "0.07610"),
    ("mxfp4-a8-online-mxfp8", "BF16 AMD MXFP4 experts A8 online MXFP8", "0.07741"),
]

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def decode_values(path):
    out = {}
    data = load(path)
    for row in data.get("results", []):
        try:
            ctx = int(row.get("context_tokens"))
            cc = int(row.get("concurrency"))
        except Exception:
            continue
        if ctx != 0 or cc not in (1, 32):
            continue
        val = row.get("aggregate_tps")
        if val is None:
            val = row.get("server_gen_throughput")
        if val is not None:
            out[f"cc{cc}"] = float(val)
    return out

def prefill_values(path):
    out = {}
    pref = load(path).get("prefill") or {}
    for key, name in (("8192", "8k"), ("65536", "64k")):
        row = pref.get(key)
        if isinstance(row, dict) and row.get("tok_per_sec") is not None:
            out[name] = float(row["tok_per_sec"])
    return out

def old_nvfp4_path(case_key):
    variant = "online" if "online" in case_key else "base"
    force = "a16" if "-a16-" in case_key else "a4"
    return old_nvfp4 / variant / force / "mtp0" / "dcp1" / "f8-0"

def source_path(case_key, dcp):
    if dcp == 1 and case_key.startswith("nvfp4"):
        return old_nvfp4_path(case_key), "old_nvfp4"
    if dcp == 1 and case_key.startswith("mxfp4"):
        sub = "online" if "online" in case_key else "orig"
        return old_mxfp4 / sub, "old_mxfp4+dcp1_cc32"
    return result_root / case_key / f"dcp{dcp}", "hybrid_v7"

def metric_files(path, source, case_key, dcp):
    if source == "old_nvfp4":
        return path / "decode_full.json", path / "prefill_full.json"
    if source == "old_mxfp4+dcp1_cc32":
        return path / "decode.json", path / "prefill.json"
    return path / "decode.json", path / "prefill.json"

def dcp1_mxfp4_cc32(case_key):
    path = result_root / case_key / "dcp1" / "decode.json"
    return decode_values(path).get("cc32")

def fmt_decode(v):
    return "" if v is None else f"{v:.2f}"

def fmt_prefill(v):
    return "" if v is None else f"{v:,.0f}"

summary = []
for key, display, kld in cases:
    row = {"case": display, "kld_mean": kld, "dcp": {}}
    for dcp in (1, 2, 4, 8):
        path, source = source_path(key, dcp)
        decode_file, prefill_file = metric_files(path, source, key, dcp)
        dec = decode_values(decode_file)
        pref = prefill_values(prefill_file)
        if dcp == 1 and key.startswith("mxfp4"):
            cc32 = dcp1_mxfp4_cc32(key)
            if cc32 is not None:
                dec["cc32"] = cc32
        row["dcp"][str(dcp)] = {
            "source": source,
            "path": str(path),
            "cc1": dec.get("cc1"),
            "cc32": dec.get("cc32"),
            "prefill_8k": pref.get("8k"),
            "prefill_64k": pref.get("64k"),
        }
    summary.append(row)

result_root.mkdir(parents=True, exist_ok=True)
(result_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

headers = ["Case", "KLD mean"]
for dcp in (1, 2, 4, 8):
    headers.extend([f"DCP{dcp} cc1", f"DCP{dcp} cc32", f"DCP{dcp} prefill 8k", f"DCP{dcp} prefill 64k"])

lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in summary:
    vals = [row["case"], row["kld_mean"]]
    for dcp in ("1", "2", "4", "8"):
        data = row["dcp"][dcp]
        vals.extend([
            fmt_decode(data["cc1"]),
            fmt_decode(data["cc32"]),
            fmt_prefill(data["prefill_8k"]),
            fmt_prefill(data["prefill_64k"]),
        ])
    lines.append("| " + " | ".join(vals) + " |")

table = "\n".join(lines) + "\n"
(result_root / "hybrid_table.md").write_text(table)
print(table)

missing = []
for row in summary:
    for dcp, data in row["dcp"].items():
        for key in ("cc1", "cc32", "prefill_8k", "prefill_64k"):
            if data[key] is None:
                missing.append(f"{row['case']} dcp{dcp} {key}")
if missing:
    print("MISSING:")
    for item in missing:
        print("  " + item)
    sys.exit(1)
PY
}

case "${1:-run}" in
  run) run_missing ;;
  summarize) summarize ;;
  stop) stop_own_containers ;;
  *) echo "usage: $0 [run|summarize|stop]" >&2; exit 2 ;;
esac
