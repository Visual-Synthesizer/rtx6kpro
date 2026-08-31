#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v17-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-20260714}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:988415592c05e2d3dc12cbc8ab36af8b6557221849f095ec3d5442602a02e304}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v17-selective-dcp-prefill-$(date -u +%Y%m%dT%H%M%SZ)}"
PROGRESS_FILE="${PROGRESS_FILE:-${RESULT_ROOT}/progress.log}"

PORT_A="${PORT_A:-8170}"
PORT_B="${PORT_B:-8171}"
GPU_A_TP8="${GPU_A_TP8:-0,1,2,3,4,5,6,7}"
GPU_B_TP8="${GPU_B_TP8:-8,9,10,11,12,13,14,15}"
GPU_A_TP6="${GPU_A_TP6:-0,1,2,3,4,5}"
GPU_B_TP6="${GPU_B_TP6:-8,9,10,11,12,13}"
CPU_A="${CPU_A:-0-31,64-95}"
CPU_B="${CPU_B:-32-63,96-127}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
PREFILL_REPEATS="${PREFILL_REPEATS:-3}"
BETWEEN_REPEATS_SECONDS="${BETWEEN_REPEATS_SECONDS:-10}"
TOKEN_TARGETING="${TOKEN_TARGETING:-estimate}"
FORCE_RERUN="${FORCE_RERUN:-0}"
INCLUDE_IMPORTED="${INCLUDE_IMPORTED:-0}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-v17-vllm6ccc3eb-b12x1377d5f}"

mkdir -p "${RESULT_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v17_selective_dcp_prefill.out
printf '%s\n' "${EXPECTED_IMAGE_ID}" > "${RESULT_ROOT}/expected-image.id"

[[ "${FORCE_RERUN}" =~ ^[01]$ ]] || { echo "FORCE_RERUN must be 0 or 1" >&2; exit 2; }
[[ "${INCLUDE_IMPORTED}" =~ ^[01]$ ]] || { echo "INCLUDE_IMPORTED must be 0 or 1" >&2; exit 2; }
[[ "${TOKEN_TARGETING}" =~ ^(estimate|exact)$ ]] || {
  echo "TOKEN_TARGETING must be estimate or exact" >&2
  exit 2
}

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
      ONLINE_QUANT=none; DISPLAY_NAME="Luke NVFP4 A4 original" ;;
    nvfp4-a4-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a4; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="Luke NVFP4 A4 online MXFP8" ;;
    nvfp4-a16-orig)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a16; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=none; DISPLAY_NAME="Luke NVFP4 A16 original" ;;
    nvfp4-a16-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; MOE_MODE=a16; QUANTIZATION=modelopt_fp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="Luke NVFP4 A16 online MXFP8" ;;
    mxfp4-a8-orig)
      MODEL_PATH="${MXFP4_MODEL}"; MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4
      ONLINE_QUANT=none; DISPLAY_NAME="BF16 AMD MXFP4 experts A8 original" ;;
    mxfp4-a8-online-mxfp8)
      MODEL_PATH="${MXFP4_MODEL}"; MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4
      ONLINE_QUANT=mxfp8; DISPLAY_NAME="BF16 AMD MXFP4 experts A8 online MXFP8" ;;
    *) echo "unknown GLM case: ${key}" >&2; return 2 ;;
  esac
}

stop_own_containers() {
  docker ps -a --format '{{.Names}}' |
    awk '/^glm52-v17-prefill-/ {print}' | xargs -r docker rm -f >/dev/null 2>&1 || true
}

trap stop_own_containers EXIT INT TERM

config_path() {
  local key="$1" tp="$2" dcp="$3" mtp="$4" dma="$5"
  printf '%s/tp%s/mtp%s/f8-%s/%s/dcp%s' "${RESULT_ROOT}" "${tp}" "${mtp}" "${dma}" "${key}" "${dcp}"
}

result_complete() {
  local out="$1"
  [[ "${FORCE_RERUN}" != "1" && -f "${out}/complete" && -s "${out}/prefill-summary.json" ]]
}

start_case() {
  local spec="$1" slot="$2"
  read -r key tp dcp mtp dma <<< "${spec}"
  case_vars "${key}"

  local port gpus cpus out name served cache_dir tmp_dir
  if [[ "${slot}" == "a" ]]; then
    port="${PORT_A}"; cpus="${CPU_A}"
    [[ "${tp}" == "8" ]] && gpus="${GPU_A_TP8}" || gpus="${GPU_A_TP6}"
  else
    port="${PORT_B}"; cpus="${CPU_B}"
    [[ "${tp}" == "8" ]] && gpus="${GPU_B_TP8}" || gpus="${GPU_B_TP6}"
  fi
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}" "${dma}")"
  name="glm52-v17-prefill-$(safe_name "${slot}-${key}-tp${tp}-dcp${dcp}-mtp${mtp}-f8${dma}")"
  served="GLM-5.2-v17-${key}-tp${tp}-dcp${dcp}-mtp${mtp}-f8${dma}"
  cache_dir="/root/.cache/vllm-glm52-v17/${CACHE_NAMESPACE}/slot-${slot}"
  tmp_dir="/root/vllm/tmp/${name}"

  mkdir -p "${out}" "${cache_dir}" "${tmp_dir}"
  rm -f "${out}/complete"
  docker rm -f "${name}" >/dev/null 2>&1 || true

  local max_len=131072 max_seqs=32 graph=128 batched=8192 gmu=0.90
  if [[ "${tp}" == "6" ]]; then
    max_len=128000; max_seqs=16; graph=64; batched=2048; gmu=0.950
  fi

  progress "START slot=${slot} case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} f8=${dma} gpus=${gpus} port=${port}"
  docker run -d --name "${name}" --network host --ipc host --privileged --init \
    --gpus all --cpuset-cpus "${cpus}" --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /root/models:/root/models:ro \
    -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
    -v "${cache_dir}":/cache:rw -v "${tmp_dir}":/container-tmp:rw \
    -e MODEL_FAMILY=glm52 -e GPUS="${gpus}" -e MODEL="${MODEL_PATH}" \
    -e SERVED_MODEL_NAME="${served}" -e PORT="${port}" -e TP="${tp}" \
    -e DCP="${dcp}" -e DCP_BACKEND=a2a -e DCP_A2A_MAX_TOKENS=64 \
    -e DCP_A2A_LARGE_BACKEND=ag_rs -e DCP_PREFILL_WORKSPACE=auto \
    -e MTP="${mtp}" -e MAX_NUM_SEQS="${max_seqs}" -e GRAPH="${graph}" \
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
    return 1
  fi
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${served}" > "${out}/served-model.name"
  printf '%s\n' "${port}" > "${out}/port"
  printf '%s\n' "${DISPLAY_NAME}" > "${out}/display.name"
}

wait_ready() {
  local spec="$1"
  read -r key tp dcp mtp dma <<< "${spec}"
  local out name port
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}" "${dma}")"
  name="$(cat "${out}/container.name")"
  port="$(cat "${out}/port")"
  for _ in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}/models.json" 2>/dev/null; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1 || true
      grep -Fq 'Loading safetensors using InstantTensor loader' "${out}/server.ready.log" || {
        progress "FAILED name=${name} reason=instanttensor-loader-not-confirmed"; return 1;
      }
      grep -Fq 'Process-group interfaces: GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo' "${out}/server.ready.log" || {
        progress "FAILED name=${name} reason=process-groups-not-pinned"; return 1;
      }
      python3 - "${out}/container.inspect.json" "${tp}" "${dcp}" <<'PY'
import json, sys
env = dict(x.split("=", 1) for x in json.load(open(sys.argv[1]))[0]["Config"]["Env"] if "=" in x)
assert env["LOAD_FORMAT"] == "instanttensor"
assert env["INSTANTTENSOR_BACKEND"] == "BUFFERED"
assert env["DCP_PREFILL_WORKSPACE"] == "auto"
assert env["TP"] == sys.argv[2] and env["DCP"] == sys.argv[3]
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
  progress "FAILED name=${name} reason=ready-timeout"
  return 1
}

run_prefill() {
  local spec="$1"
  read -r key tp dcp mtp dma <<< "${spec}"
  local out port served run
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}" "${dma}")"
  port="$(cat "${out}/port")"
  served="$(cat "${out}/served-model.name")"
  progress "PREFILL_START case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} f8=${dma} repeats=${PREFILL_REPEATS}"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "${out}/thermal-before.csv" 2>/dev/null || true
  for run in $(seq 1 "${PREFILL_REPEATS}"); do
    python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
      --prefill-only --standalone-prefill --prefill-contexts 8k,64k \
      --prefill-duration "${PREFILL_DURATION}" --token-targeting "${TOKEN_TARGETING}" \
      --max-tokens 1 --no-hw-monitor \
      --output "${out}/prefill-run${run}.json" > "${out}/prefill-run${run}.log" 2>&1
    if ((run < PREFILL_REPEATS)); then sleep "${BETWEEN_REPEATS_SECONDS}"; fi
  done
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "${out}/thermal-after.csv" 2>/dev/null || true
  docker logs "$(cat "${out}/container.name")" > "${out}/server.final.log" 2>&1 || true
  grep -Fq 'Using borrowed B12X workspaces for sparse MLA DCP prefill' "${out}/server.final.log" || {
    progress "FAILED case=${key} tp=${tp} dcp=${dcp} reason=dcp-prefill-workspace-not-observed"
    return 1
  }
  python3 - "${out}" "${PREFILL_REPEATS}" <<'PY'
import json, pathlib, statistics, sys
root, repeats = pathlib.Path(sys.argv[1]), int(sys.argv[2])
values = {"8192": [], "65536": []}
for run in range(1, repeats + 1):
    data = json.load(open(root / f"prefill-run{run}.json"))["prefill"]
    for context in values:
        values[context].append(float(data[context]["tok_per_sec"]))
summary = {
    context: {
        "runs": runs,
        "mean_tok_per_sec": statistics.fmean(runs),
        "median_tok_per_sec": statistics.median(runs),
        "min_tok_per_sec": min(runs),
        "max_tok_per_sec": max(runs),
        "relative_spread": (max(runs) - min(runs)) / statistics.fmean(runs),
    }
    for context, runs in values.items()
}
(root / "prefill-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
  touch "${out}/complete"
  progress "PREFILL_DONE case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} f8=${dma}"
}

run_pair() {
  local spec_a="$1" spec_b="${2:-}"
  local out_a out_b
  read -r key_a tp_a dcp_a mtp_a dma_a <<< "${spec_a}"
  out_a="$(config_path "${key_a}" "${tp_a}" "${dcp_a}" "${mtp_a}" "${dma_a}")"
  if [[ -n "${spec_b}" ]]; then
    read -r key_b tp_b dcp_b mtp_b dma_b <<< "${spec_b}"
    out_b="$(config_path "${key_b}" "${tp_b}" "${dcp_b}" "${mtp_b}" "${dma_b}")"
  else
    out_b=""
  fi

  if result_complete "${out_a}" && { [[ -z "${out_b}" ]] || result_complete "${out_b}"; }; then
    progress "SKIP_COMPLETE pair=${spec_a}|${spec_b}"
    return 0
  fi

  stop_own_containers
  if ! result_complete "${out_a}"; then start_case "${spec_a}" a; fi
  if [[ -n "${spec_b}" ]] && ! result_complete "${out_b}"; then start_case "${spec_b}" b; fi

  local pids=() pid
  if ! result_complete "${out_a}"; then wait_ready "${spec_a}" & pids+=("$!"); fi
  if [[ -n "${spec_b}" ]] && ! result_complete "${out_b}"; then wait_ready "${spec_b}" & pids+=("$!"); fi
  for pid in "${pids[@]}"; do wait "${pid}"; done

  progress "ALL_READY settle_seconds=${SETTLE_SECONDS} pair=${spec_a}|${spec_b}"
  sleep "${SETTLE_SECONDS}"
  if ! result_complete "${out_a}"; then run_prefill "${spec_a}"; fi
  if [[ -n "${spec_b}" ]] && ! result_complete "${out_b}"; then run_prefill "${spec_b}"; fi
  stop_own_containers
}

declare -a CONFIGS=()

add_tp8_mtp0() {
  local dcp
  for dcp in 2 4 8; do
    CONFIGS+=("nvfp4-a4-orig 8 ${dcp} 0 0")
    CONFIGS+=("nvfp4-a4-online-mxfp8 8 ${dcp} 0 0")
    # A16 original is already measured twice on the final v17 image.
    if [[ "${INCLUDE_IMPORTED}" == "1" ]]; then
      CONFIGS+=("nvfp4-a16-orig 8 ${dcp} 0 0")
    fi
    CONFIGS+=("nvfp4-a16-online-mxfp8 8 ${dcp} 0 0")
    CONFIGS+=("mxfp4-a8-orig 8 ${dcp} 0 0")
    CONFIGS+=("mxfp4-a8-online-mxfp8 8 ${dcp} 0 0")
  done
}

add_tp8_mtp3() {
  local dcp key
  for dcp in 2 4 8; do
    for key in nvfp4-a4-orig nvfp4-a4-online-mxfp8 nvfp4-a16-orig nvfp4-a16-online-mxfp8; do
      CONFIGS+=("${key} 8 ${dcp} 3 0")
    done
  done
}

add_tp8_dma() {
  local dcp key dma
  for dma in ag ring; do
    for dcp in 2 4 8; do
      for key in nvfp4-a4-orig nvfp4-a4-online-mxfp8; do
        CONFIGS+=("${key} 8 ${dcp} 3 ${dma}")
      done
    done
  done
}

add_tp6() {
  local dcp
  for dcp in 2 3 6; do
    # A8 original is already measured twice on the final v17 image.
    if [[ "${INCLUDE_IMPORTED}" == "1" ]]; then
      CONFIGS+=("mxfp4-a8-orig 6 ${dcp} 0 0")
    fi
    CONFIGS+=("mxfp4-a8-online-mxfp8 6 ${dcp} 0 0")
  done
}

summarize() {
  python3 - "${RESULT_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("tp*/mtp*/f8-*/*/dcp*")):
    summary_path = path / "prefill-summary.json"
    if not summary_path.exists():
        continue
    data = json.load(open(summary_path))
    rows.append({
        "tp": int(path.parents[3].name[2:]),
        "mtp": int(path.parents[2].name[3:]),
        "f8": path.parents[1].name[3:],
        "case": path.parent.name,
        "dcp": int(path.name[3:]),
        "prefill_8k": data["8192"].get("median_tok_per_sec", data["8192"]["mean_tok_per_sec"]),
        "prefill_64k": data["65536"].get("median_tok_per_sec", data["65536"]["mean_tok_per_sec"]),
        "path": str(path),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
print("tp\tmtp\tf8\tcase\tdcp\tprefill8k\tprefill64k")
for row in rows:
    print(f'{row["tp"]}\t{row["mtp"]}\t{row["f8"]}\t{row["case"]}\t{row["dcp"]}\t'
          f'{row["prefill_8k"]:.2f}\t{row["prefill_64k"]:.2f}')
PY
}

mode="${1:-all}"
if (($#)); then shift; fi
case "${mode}" in
  tp8-mtp0) add_tp8_mtp0 ;;
  tp8-mtp3) add_tp8_mtp3 ;;
  tp8-dma) add_tp8_dma ;;
  tp6) add_tp6 ;;
  all) add_tp8_mtp0; add_tp8_mtp3; add_tp8_dma; add_tp6 ;;
  configs)
    (($#)) || { echo "configs mode requires at least one quoted config" >&2; exit 2; }
    CONFIGS=("$@")
    ;;
  summarize) summarize; exit 0 ;;
  stop) stop_own_containers; exit 0 ;;
  *)
    echo "usage: $0 [tp8-mtp0|tp8-mtp3|tp8-dma|tp6|all|summarize|stop]" >&2
    echo "       $0 configs \"CASE TP DCP MTP F8\" [\"CASE TP DCP MTP F8\" ...]" >&2
    exit 2
    ;;
esac

progress "RUN_START mode=${mode} image=${IMAGE} result_root=${RESULT_ROOT} configs=${#CONFIGS[@]} include_imported=${INCLUDE_IMPORTED} token_targeting=${TOKEN_TARGETING}"
for ((i = 0; i < ${#CONFIGS[@]}; i += 2)); do
  run_pair "${CONFIGS[i]}" "${CONFIGS[i + 1]:-}"
  summarize | tee "${RESULT_ROOT}/summary.tsv"
done
progress "RUN_DONE mode=${mode} result_root=${RESULT_ROOT}"
