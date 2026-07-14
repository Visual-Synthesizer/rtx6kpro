#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v16-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:d4d4739010a71c6f424c3f7a067e3fd0fdeea72b8e49040bd8e8f167b21418a7}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v16-full-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

GPU_A="${GPU_A:-0,1,2,3,4,5,6,7}"
GPU_B="${GPU_B:-8,9,10,11,12,13,14,15}"
CPU_A="${CPU_A:-0-31,64-95}"
CPU_B="${CPU_B:-32-63,96-127}"
PORT_A="${PORT_A:-8160}"
PORT_B="${PORT_B:-8161}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GRAPH="${GRAPH:-128}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
REGRESSION_LIMIT="${REGRESSION_LIMIT:-0.05}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-vllm8f86f42-b12xfe06f49}"
DCP_VALUES="${DCP_VALUES:-1 2 4 8}"
CORE_GROUPS="${CORE_GROUPS:-a4 a16 mxfp4}"
MTP3_GROUPS="${MTP3_GROUPS:-a4 a16}"

mkdir -p "${RESULT_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v16_full.out
printf '%s\n' "${EXPECTED_IMAGE_ID}" > "${RESULT_ROOT}/expected-image.id"

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

safe_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' |
    sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//'
}

case_vars() {
  local key="$1"
  MODEL_PATH= MOE_MODE= QUANTIZATION= ONLINE_QUANT= DISPLAY_NAME=
  case "${key}" in
    nvfp4-a4-orig)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a4; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=none; DISPLAY_NAME="Luke NVFP4 A4 orig" ;;
    nvfp4-a4-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a4; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="Luke NVFP4 A4 online MXFP8" ;;
    nvfp4-a16-orig)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a16; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=none; DISPLAY_NAME="Luke NVFP4 A16 orig" ;;
    nvfp4-a16-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a16; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="Luke NVFP4 A16 online MXFP8" ;;
    mxfp4-a8-orig)
      MODEL_PATH="${MXFP4_MODEL}"; MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4
      ONLINE_QUANT=none; DISPLAY_NAME="BF16 AMD MXFP4 experts A8 orig" ;;
    mxfp4-a8-online-mxfp8)
      MODEL_PATH="${MXFP4_MODEL}"; MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="BF16 AMD MXFP4 experts A8 online MXFP8" ;;
    mxfp4-a8-online-fp8)
      MODEL_PATH="${MXFP4_MODEL}"; MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4
      ONLINE_QUANT=fp8; DISPLAY_NAME="BF16 AMD MXFP4 experts A8 online FP8" ;;
    *) echo "unknown GLM case: ${key}" >&2; return 2 ;;
  esac
}

stop_own_containers() {
  docker ps -a --format '{{.Names}}' |
    awk '/^glm52-v16-sweep-/ {print}' | xargs -r docker rm -f >/dev/null 2>&1 || true
}

trap stop_own_containers EXIT INT TERM

start_case() {
  local key="$1" mtp="$2" dcp="$3" dma="$4" port="$5" gpus="$6" cpus="$7" out="$8"
  local tp="${9:-8}"
  case_vars "${key}"
  local name served cache_dir tmp_dir
  name="glm52-v16-sweep-$(safe_name "${key}-tp${tp}-dcp${dcp}-mtp${mtp}-f8${dma}-p${port}")"
  served="GLM-5.2-v16-${key}-tp${tp}-dcp${dcp}-mtp${mtp}-f8${dma}"
  cache_dir="/root/.cache/vllm-glm52-v16/${CACHE_NAMESPACE}/group-$([[ "${port}" == "${PORT_A}" ]] && echo a || echo b)"
  tmp_dir="/root/vllm/tmp/${name}"
  mkdir -p "${out}" "${cache_dir}" "${tmp_dir}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  progress "START case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} f8=${dma} gpus=${gpus} port=${port}"

  local gmu="${GPU_MEMORY_UTILIZATION}" max_len="${MAX_MODEL_LEN}"
  local max_seqs="${MAX_NUM_SEQS}" graph="${GRAPH}" batched="${MAX_BATCHED_TOKENS}"
  if [[ "${tp}" == "6" ]]; then
    max_len=128000; max_seqs=16; graph=64; batched=2048
    if [[ "${dcp}" == "1" ]]; then gmu=0.957; else gmu=0.950; fi
  fi

  docker run -d --name "${name}" --network host --ipc host --privileged --init \
    --gpus all --cpuset-cpus "${cpus}" --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /root/models:/root/models:ro \
    -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
    -v "${cache_dir}":/cache:rw -v "${tmp_dir}":/container-tmp:rw \
    -e MODEL_FAMILY=glm52 -e GPUS="${gpus}" -e MODEL="${MODEL_PATH}" \
    -e SERVED_MODEL_NAME="${served}" -e PORT="${port}" -e TP="${tp}" \
    -e DCP="${dcp}" -e DCP_BACKEND=a2a -e DCP_A2A_MAX_TOKENS=64 \
    -e DCP_A2A_LARGE_BACKEND=ag_rs -e MTP="${mtp}" \
    -e MAX_NUM_SEQS="${max_seqs}" -e GRAPH="${graph}" \
    -e MAX_MODEL_LEN="${max_len}" -e MAX_BATCHED_TOKENS="${batched}" \
    -e GPU_MEMORY_UTILIZATION="${gmu}" -e MOE_MODE="${MOE_MODE}" \
    -e MOE_BACKEND=b12x -e LINEAR_BACKEND=auto -e QUANTIZATION="${QUANTIZATION}" \
    -e ONLINE_QUANT="${ONLINE_QUANT}" -e F8_DMA="${dma}" \
    -e LOAD_FORMAT=instanttensor -e INSTANTTENSOR_BACKEND=BUFFERED \
    --entrypoint /usr/local/bin/serve-fathomless-firmament.sh "${IMAGE}" \
    > "${out}/container.id"
  docker inspect "${name}" > "${out}/container.inspect.json"
  local actual_image_id
  actual_image_id="$(docker inspect --format '{{.Image}}' "${name}")"
  printf '%s\n' "${actual_image_id}" > "${out}/image.id"
  if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
    progress "FAILED name=${name} expected_image=${EXPECTED_IMAGE_ID} actual_image=${actual_image_id}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    return 1
  fi
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${served}" > "${out}/served_model.name"
  printf '%s\n' "${DISPLAY_NAME}" > "${out}/display.name"
}

wait_ready() {
  local name="$1" port="$2" out="$3"
  for _ in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}/models.json" 2>/dev/null; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1 || true
      grep -Fq 'Loading safetensors using InstantTensor loader' "${out}/server.ready.log" || {
        progress "FAILED name=${name} reason=instanttensor-loader-not-confirmed"
        return 1
      }
      grep -Fq 'B12X_PCIE_ONESHOT_DMA' "${out}/server.ready.log" || {
        progress "FAILED name=${name} reason=b12x-pcie-dma-not-confirmed"
        return 1
      }
      grep -Fq 'Process-group interfaces: GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo' "${out}/server.ready.log" || {
        progress "FAILED name=${name} reason=process-group-interface-not-pinned"
        return 1
      }
      python3 - "${out}/container.inspect.json" <<'PY'
import json, sys
env = dict(item.split("=", 1) for item in json.load(open(sys.argv[1]))[0]["Config"]["Env"] if "=" in item)
assert env.get("INSTANTTENSOR_BACKEND") == "BUFFERED", env.get("INSTANTTENSOR_BACKEND")
PY
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
  return 1
}

parse_runtime() {
  local out="$1"
  python3 - "${out}/server.ready.log" > "${out}/runtime.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], errors="replace").read()
patterns = {
    "gpu_kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "available_kv_cache_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "max_concurrency": r"Maximum concurrency for .*?: ([0-9.]+)x",
}
out = {"instanttensor_loader": "Loading safetensors using InstantTensor loader" in text,
       "b12x_pcie_dma": "B12X_PCIE_ONESHOT_DMA" in text,
       "process_groups_on_loopback": "GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo" in text,
       "nccl_2304": "vLLM is using nccl==2.30.4" in text}
for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
        value = match.group(1).replace(",", "")
        out[key] = float(value) if "." in value else int(value)
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

run_decode() {
  local label="$1" port="$2" served="$3" out="$4" concurrency="${5:-1,2,4,8,16,32}"
  progress "DECODE_START label=${label} cc=${concurrency}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "${out}/thermal-before-decode.csv" 2>/dev/null || true
  python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
    --skip-prefill --contexts 0 --concurrency "${concurrency}" \
    --duration "${DECODE_DURATION}" --max-tokens "${DECODE_MAX_TOKENS}" \
    --no-hw-monitor --output "${out}/decode.json" > "${out}/decode.log" 2>&1
  progress "DECODE_DONE label=${label}"
}

run_prefill() {
  local label="$1" port="$2" served="$3" out="$4"
  progress "PREFILL_START label=${label}"
  python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
    --prefill-only --standalone-prefill --prefill-contexts 8k,64k \
    --prefill-duration "${PREFILL_DURATION}" --max-tokens 1 --no-hw-monitor \
    --output "${out}/prefill.json" > "${out}/prefill.log" 2>&1
  progress "PREFILL_DONE label=${label}"
}

bench_pair() {
  local key_a="$1" key_b="$2" mtp="$3" dcp="$4" dma="$5" mode="${6:-full}" tp="${7:-8}"
  local base="${RESULT_ROOT}/tp${tp}/mtp${mtp}/f8-${dma}"
  local out_a="${base}/${key_a}/dcp${dcp}" out_b="${base}/${key_b}/dcp${dcp}"
  if [[ "${FORCE_RERUN}" != "1" && -f "${out_a}/gate-passed" && -f "${out_b}/gate-passed" ]]; then
    progress "SKIP_PASSED pair=${key_a},${key_b} tp=${tp} dcp=${dcp} mtp=${mtp} f8=${dma}"
    return 0
  fi
  # A forced or incomplete rerun must earn a fresh gate marker. Otherwise an
  # earlier successful cell could remain marked valid after a failed retry.
  rm -f "${out_a}/gate-passed" "${out_b}/gate-passed"
  stop_own_containers
  start_case "${key_a}" "${mtp}" "${dcp}" "${dma}" "${PORT_A}" "${GPU_A}" "${CPU_A}" "${out_a}" "${tp}"
  start_case "${key_b}" "${mtp}" "${dcp}" "${dma}" "${PORT_B}" "${GPU_B}" "${CPU_B}" "${out_b}" "${tp}"

  wait_ready "$(cat "${out_a}/container.name")" "${PORT_A}" "${out_a}" & local pid_a=$!
  wait_ready "$(cat "${out_b}/container.name")" "${PORT_B}" "${out_b}" & local pid_b=$!
  wait "${pid_a}"; wait "${pid_b}"
  progress "SETTLE seconds=${SETTLE_SECONDS} pair=${key_a},${key_b}"
  sleep "${SETTLE_SECONDS}"
  parse_runtime "${out_a}"; parse_runtime "${out_b}"

  local served_a served_b
  served_a="$(cat "${out_a}/served_model.name")"; served_b="$(cat "${out_b}/served_model.name")"
  if [[ "${mode}" != "prefill" ]]; then
    local conc=1,2,4,8,16,32
    [[ "${tp}" == "6" ]] && conc=1,2,4,8,16
    run_decode "${key_a}-dcp${dcp}" "${PORT_A}" "${served_a}" "${out_a}" "${conc}" & pid_a=$!
    run_decode "${key_b}-dcp${dcp}" "${PORT_B}" "${served_b}" "${out_b}" "${conc}" & pid_b=$!
    wait "${pid_a}"; wait "${pid_b}"
  fi
  if [[ "${mode}" != "decode" ]]; then
    # Long prefill from two TP8 servers can contend on the shared host PCIe
    # fabric. Keep both models resident, but measure one endpoint at a time.
    run_prefill "${key_a}-dcp${dcp}" "${PORT_A}" "${served_a}" "${out_a}"
    run_prefill "${key_b}-dcp${dcp}" "${PORT_B}" "${served_b}" "${out_b}"
  fi
  docker logs "$(cat "${out_a}/container.name")" > "${out_a}/server.final.log" 2>&1 || true
  docker logs "$(cat "${out_b}/container.name")" > "${out_b}/server.final.log" 2>&1 || true
  stop_own_containers
  summarize
  if [[ "${tp}" == "8" && "${dcp}" == "1" && "${key_a}" == "mxfp4-a8-online-fp8" ]]; then
    fp8_pair_gate "${mtp}" "${out_a}" "${out_b}"
  elif [[ "${mtp}" == "0" && "${dma}" == "0" && "${tp}" == "8" && "${mode}" == "full" ]]; then
    regression_gate "${key_a}" "${dcp}" "${out_a}"
    regression_gate "${key_b}" "${dcp}" "${out_b}"
  elif [[ "${mtp}" == "3" && "${dma}" == "0" && "${tp}" == "8" && "${mode}" == "full" && "${key_a}" == nvfp4-* ]]; then
    mtp3_regression_gate "${key_a}" "${dcp}" "${out_a}"
    mtp3_regression_gate "${key_b}" "${dcp}" "${out_b}"
  elif [[ "${mtp}" == "3" && "${tp}" == "8" && "${mode}" == "prefill" ]]; then
    dma_regression_gate "${key_a}" "${dcp}" "${dma}" "${out_a}"
    dma_regression_gate "${key_b}" "${dcp}" "${dma}" "${out_b}"
  elif [[ "${tp}" == "6" && "${mode}" == "full" ]]; then
    tp6_regression_gate "${key_b}" "${dcp}" "${out_b}"
  fi
  touch "${out_a}/gate-passed" "${out_b}/gate-passed"
}

regression_gate() {
  local key="$1" dcp="$2" out="$3"
  python3 - "${key}" "${dcp}" "${out}" "${REGRESSION_LIMIT}" <<'PY'
import json, pathlib, sys
key, dcp, root, limit = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), float(sys.argv[4])
baseline = {
 "nvfp4-a4-orig": {1:(87.99,934.07,6557,6257),2:(72.44,838.57,4679,4710),4:(71.65,747.11,3415,3455),8:(67.29,606.35,2197,2209)},
 "nvfp4-a4-online-mxfp8": {1:(94.96,953.24,6681,6351),2:(76.26,847.24,4636,4718),4:(75.32,760.87,3402,3468),8:(70.84,617.18,2188,2212)},
 "nvfp4-a16-orig": {1:(86.56,932.72,6140,5849),2:(71.48,828.30,4455,4481),4:(70.74,750.20,3301,3326),8:(66.11,610.88,2147,2157)},
 "nvfp4-a16-online-mxfp8": {1:(93.30,954.52,6239,5941),2:(74.85,837.81,4385,4471),4:(73.99,752.91,3270,3331),8:(69.45,610.40,2132,2155)},
 "mxfp4-a8-orig": {1:(88.72,938.10,6698,6307),2:(71.84,832.28,4747,4786),4:(71.73,745.91,3450,3491),8:(67.15,613.70,2206,2220)},
 "mxfp4-a8-online-mxfp8": {1:(94.03,956.30,6731,6364),2:(75.66,840.02,4702,4781),4:(75.37,761.43,3427,3495),8:(71.01,607.69,2200,2223)},
}
if key not in baseline:
    raise SystemExit(0)
decode = json.load(open(root / "decode.json"))
values = {}
for row in decode.get("results", []):
    if int(row.get("context_tokens", -1)) == 0:
        cc = int(row.get("concurrency", -1))
        if cc in (1, 32):
            values[f"cc{cc}"] = float(row.get("aggregate_tps") or row.get("server_gen_throughput"))
pref = json.load(open(root / "prefill.json"))["prefill"]
values["p8k"] = float(pref["8192"]["tok_per_sec"])
values["p64k"] = float(pref["65536"]["tok_per_sec"])
expected = dict(zip(("cc1", "cc32", "p8k", "p64k"), baseline[key][dcp]))
bad = []
for metric, old in expected.items():
    new = values[metric]
    delta = new / old - 1
    print(f"GATE {key} dcp{dcp} {metric}: {new:.2f} vs {old:.2f} ({delta:+.2%})")
    # The short 8k prefill point is retained for comparison but is too noisy
    # to gate the release. Decode and the representative 64k prefill do gate.
    if metric != "p8k" and delta < -limit:
        bad.append((metric, new, old, delta))
if bad:
    print("REGRESSION: rerun this cell before release", file=sys.stderr)
    raise SystemExit(20)
PY
}

mtp3_regression_gate() {
  local key="$1" dcp="$2" out="$3"
  python3 - "${key}" "${dcp}" "${out}" "${REGRESSION_LIMIT}" <<'PY'
import json, pathlib, sys
key, dcp, root, limit = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), float(sys.argv[4])
baseline = {
 "nvfp4-a4-orig": {1:(125.90,1427.00,6136),2:(100.78,1186.00,4570),4:(99.30,1070.00,3392),8:(95.84,827.86,2156)},
 "nvfp4-a4-online-mxfp8": {1:(129.37,1461.00,6222),2:(104.96,1225.00,4618),4:(100.28,1085.00,3422),8:(98.23,842.56,2166)},
 "nvfp4-a16-orig": {1:(119.62,1345.00,5740),2:(90.69,1134.00,4335),4:(89.44,1030.00,3267),8:(90.48,793.75,2100)},
 "nvfp4-a16-online-mxfp8": {1:(120.69,1378.00,5833),2:(92.47,1163.00,4392),4:(95.56,1051.00,3294),8:(92.51,803.96,2114)},
}
decode = json.load(open(root / "decode.json"))
values = {}
for row in decode.get("results", []):
    if int(row.get("context_tokens", -1)) == 0:
        cc = int(row.get("concurrency", -1))
        if cc in (1, 32):
            values[f"cc{cc}"] = float(row.get("aggregate_tps") or row.get("server_gen_throughput"))
values["p64k"] = float(json.load(open(root / "prefill.json"))["prefill"]["65536"]["tok_per_sec"])
expected = dict(zip(("cc1", "cc32", "p64k"), baseline[key][dcp]))
bad = []
for metric, old in expected.items():
    new = values[metric]
    delta = new / old - 1
    print(f"MTP3_GATE {key} dcp{dcp} {metric}: {new:.2f} vs {old:.2f} ({delta:+.2%})")
    if delta < -limit:
        bad.append((metric, new, old, delta))
if bad:
    raise SystemExit("MTP3 regression: rerun and investigate before release")
PY
}

dma_regression_gate() {
  local key="$1" dcp="$2" dma="$3" out="$4"
  python3 - "${key}" "${dcp}" "${dma}" "${out}" "${REGRESSION_LIMIT}" <<'PY'
import json, pathlib, sys
key, dcp, dma, root, limit = sys.argv[1], int(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4]), float(sys.argv[5])
baseline = {
 ("nvfp4-a4-orig", "ag"):{1:6738,2:4894,4:3571,8:2226},
 ("nvfp4-a4-orig", "ring"):{1:7435,2:5272,4:3757,8:2300},
 ("nvfp4-a4-online-mxfp8", "ag"):{1:6843,2:4963,4:3602,8:2240},
 ("nvfp4-a4-online-mxfp8", "ring"):{1:7564,2:5328,4:3791,8:2314},
}
new = float(json.load(open(root / "prefill.json"))["prefill"]["65536"]["tok_per_sec"])
old = baseline[(key, dma)][dcp]
delta = new / old - 1
print(f"DMA_GATE {key} {dma} dcp{dcp} p64k: {new:.2f} vs {old:.2f} ({delta:+.2%})")
if delta < -limit:
    raise SystemExit("FP8 DMA prefill regression: rerun and investigate before release")
PY
}

fp8_pair_gate() {
  local mtp="$1" fp8_out="$2" mxfp8_out="$3"
  python3 - "${mtp}" "${fp8_out}" "${mxfp8_out}" <<'PY'
import json, pathlib, sys
mtp, fp8_root, mx_root = int(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
def cc1(root):
    for row in json.load(open(root / "decode.json")).get("results", []):
        if int(row.get("context_tokens", -1)) == 0 and int(row.get("concurrency", -1)) == 1:
            return float(row.get("aggregate_tps") or row.get("server_gen_throughput"))
    raise RuntimeError(f"missing cc1 in {root}")
def p64(root):
    return float(json.load(open(root / "prefill.json"))["prefill"]["65536"]["tok_per_sec"])
fp8_cc1, mx_cc1, fp8_p64, mx_p64 = cc1(fp8_root), cc1(mx_root), p64(fp8_root), p64(mx_root)
print(f"FP8_PAIR_GATE mtp{mtp} cc1: fp8={fp8_cc1:.2f} mxfp8={mx_cc1:.2f} delta={fp8_cc1 / mx_cc1 - 1:+.2%}")
print(f"FP8_PAIR_GATE mtp{mtp} p64k: fp8={fp8_p64:.2f} mxfp8={mx_p64:.2f} delta={fp8_p64 / mx_p64 - 1:+.2%}")
if fp8_cc1 < mx_cc1 * 1.02:
    raise SystemExit("online FP8 did not retain its expected decode advantage")
if fp8_p64 < mx_p64 * 0.95:
    raise SystemExit("online FP8 prefill regressed against online MXFP8")
PY
}

tp6_regression_gate() {
  local key="$1" dcp="$2" out="$3"
  python3 - "${key}" "${dcp}" "${out}" "${REGRESSION_LIMIT}" <<'PY'
import json, pathlib, sys
key, dcp, root, limit = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), float(sys.argv[4])
if key != "mxfp4-a8-online-mxfp8":
    raise SystemExit(0)
baseline = {1:(83.4318,5315),2:(65.8970,3862),3:(63.9962,3218),6:(49.5353,2133)}
cc1 = None
for row in json.load(open(root / "decode.json")).get("results", []):
    if int(row.get("context_tokens", -1)) == 0 and int(row.get("concurrency", -1)) == 1:
        cc1 = float(row.get("aggregate_tps") or row.get("server_gen_throughput"))
p64 = float(json.load(open(root / "prefill.json"))["prefill"]["65536"]["tok_per_sec"])
bad = []
for metric, new, old in (("cc1", cc1, baseline[dcp][0]), ("p64k", p64, baseline[dcp][1])):
    delta = new / old - 1
    print(f"TP6_GATE {key} dcp{dcp} {metric}: {new:.2f} vs {old:.2f} ({delta:+.2%})")
    if delta < -limit:
        bad.append((metric, delta))
if bad:
    raise SystemExit("TP6 regression: rerun and investigate before release")
PY
}

summarize() {
  python3 - "${RESULT_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("tp*/mtp*/f8-*/*/dcp*")):
    row = {"path": str(path), "case": path.parent.name,
           "dcp": int(path.name[3:]), "f8": path.parent.parent.name[3:],
           "mtp": int(path.parent.parent.parent.name[3:]),
           "tp": int(path.parent.parent.parent.parent.name[2:])}
    try:
        data = json.load(open(path / "decode.json"))
        row["decode"] = {str(int(x["concurrency"])): float(x.get("aggregate_tps") or x.get("server_gen_throughput"))
                         for x in data.get("results", []) if int(x.get("context_tokens", -1)) == 0}
    except Exception: row["decode"] = {}
    try:
        data = json.load(open(path / "prefill.json"))["prefill"]
        row["prefill"] = {k: float(v["tok_per_sec"]) for k, v in data.items() if v.get("tok_per_sec") is not None}
    except Exception: row["prefill"] = {}
    try: row["runtime"] = json.load(open(path / "runtime.json"))
    except Exception: row["runtime"] = {}
    rows.append(row)
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
print("case\tmtp\tf8\tdcp\tcc1\tcc32\tprefill8k\tprefill64k")
for r in rows:
    print(f'{r["case"]}\t{r["mtp"]}\t{r["f8"]}\t{r["dcp"]}\t'
          f'{r["decode"].get("1", "")}\t{r["decode"].get("32", "")}\t'
          f'{r["prefill"].get("8192", "")}\t{r["prefill"].get("65536", "")}')
PY
}

run_core() {
  local dcp group
  for dcp in ${DCP_VALUES}; do
    for group in ${CORE_GROUPS}; do
      case "${group}" in
        a4) bench_pair nvfp4-a4-orig nvfp4-a4-online-mxfp8 0 "${dcp}" 0 ;;
        a16) bench_pair nvfp4-a16-orig nvfp4-a16-online-mxfp8 0 "${dcp}" 0 ;;
        mxfp4) bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 0 "${dcp}" 0 ;;
        *) echo "unknown CORE_GROUPS entry: ${group}" >&2; return 2 ;;
      esac
    done
  done
}

run_mtp3() {
  local dcp group
  for dcp in ${DCP_VALUES}; do
    for group in ${MTP3_GROUPS}; do
      case "${group}" in
        a4) bench_pair nvfp4-a4-orig nvfp4-a4-online-mxfp8 3 "${dcp}" 0 ;;
        a16) bench_pair nvfp4-a16-orig nvfp4-a16-online-mxfp8 3 "${dcp}" 0 ;;
        *) echo "unknown MTP3_GROUPS entry: ${group}" >&2; return 2 ;;
      esac
    done
  done
}

run_dma() {
  for dma in ag ring; do
    for dcp in 1 2 4 8; do
      bench_pair nvfp4-a4-orig nvfp4-a4-online-mxfp8 3 "${dcp}" "${dma}" prefill
    done
  done
}

run_fp8() {
  bench_pair mxfp4-a8-online-fp8 mxfp4-a8-online-mxfp8 0 1 0
  bench_pair mxfp4-a8-online-fp8 mxfp4-a8-online-mxfp8 3 1 0
}

run_tp6() {
  local old_a="${GPU_A}" old_b="${GPU_B}"
  GPU_A=0,1,2,3,4,5; GPU_B=8,9,10,11,12,13
  bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 0 1 0 full 6
  bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 0 2 0 full 6
  bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 0 3 0 full 6
  bench_pair mxfp4-a8-orig mxfp4-a8-online-mxfp8 0 6 0 full 6
  GPU_A="${old_a}"; GPU_B="${old_b}"
}

mode="${1:-all}"
progress "RUN_START mode=${mode} image=${IMAGE} result_root=${RESULT_ROOT}"
case "${mode}" in
  core) run_core ;;
  mtp3) run_mtp3 ;;
  dma) run_dma ;;
  fp8) run_fp8 ;;
  tp6) run_tp6 ;;
  all) run_core; run_mtp3; run_dma; run_fp8; run_tp6 ;;
  summarize) summarize ;;
  stop) stop_own_containers ;;
  *) echo "usage: $0 [core|mtp3|dma|fp8|tp6|all|summarize|stop]" >&2; exit 2 ;;
esac
summarize | tee "${RESULT_ROOT}/summary.tsv"
progress "RUN_DONE mode=${mode} result_root=${RESULT_ROOT}"
