#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-glm52-dcp-fp8nvfp4fix-cu132-20260705}"
ONLINE_IMAGE="${ONLINE_IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-pr74-mxfp8overlay-cu132-20260705}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
if [[ -z "${QUANTIZATION_CONFIG_JSON+x}" ]]; then
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}'
fi

PREFILL_REF="${PREFILL_REF:-/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618}"
PATTERN="${PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${PATTERN}\"}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v14-todo-and-sweep-${RUN_ID}}"
KLD_ROOT="${KLD_ROOT:-/root/kld/glm52_v14_todo_${RUN_ID}}"
PROGRESS_FILE="${PROGRESS_FILE:-/root/vllm/prubezne_vysledky}"

GPU_A="${GPU_A:-0,1,2,3,4,5,6,7}"
GPU_B="${GPU_B:-8,9,10,11,12,13,14,15}"
CUDA_A="${CUDA_A:-0,1,2,3,4,5,6,7}"
CUDA_B="${CUDA_B:-8,9,10,11,12,13,14,15}"
PORT_A="${PORT_A:-5711}"
PORT_B="${PORT_B:-5713}"
SETTLE_SECONDS="${SETTLE_SECONDS:-30}"

TABLE_MAX_NUM_SEQS="${TABLE_MAX_NUM_SEQS:-64}"
TABLE_MAX_CUDAGRAPH_CAPTURE_SIZE="${TABLE_MAX_CUDAGRAPH_CAPTURE_SIZE:-256}"
SWEEP_MAX_NUM_SEQS="${SWEEP_MAX_NUM_SEQS:-32}"
SWEEP_MAX_CUDAGRAPH_CAPTURE_SIZE="${SWEEP_MAX_CUDAGRAPH_CAPTURE_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-1,2,4,8,16,32}"
DECODE_CONTEXTS="${DECODE_CONTEXTS:-0,16k,32k,64k,128k}"
DECODE_DURATION="${DECODE_DURATION:-30}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-8192}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k,128k}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
RUN_STANDALONE_PREFILL="${RUN_STANDALONE_PREFILL:-1}"

TABLE_PREFILL_CONTEXTS="${TABLE_PREFILL_CONTEXTS:-30k,64k,120k}"
TABLE_DECODE_DURATION="${TABLE_DECODE_DURATION:-30}"
TABLE_DECODE_MAX_TOKENS="${TABLE_DECODE_MAX_TOKENS:-2048}"
TABLE_CODING_PEAK_RUNS="${TABLE_CODING_PEAK_RUNS:-5}"
KLD_RUNS="${KLD_RUNS:-5}"

if [[ "${#PATTERN}" -ne 78 ]]; then
  echo "ERROR: index_topk_pattern must be 78 chars, got ${#PATTERN}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}" "${KLD_ROOT}" "$(dirname "${PROGRESS_FILE}")"
printf '%s\n' "${RESULT_ROOT}" > /root/bench-results/latest_glm52_v14_todo_and_sweep.out
printf '%s\n' "${KLD_ROOT}" > /root/kld/latest_glm52_v14_todo.out

QUANTIZATION_CONFIG_JSON="$(
  QUANTIZATION_CONFIG_JSON="${QUANTIZATION_CONFIG_JSON}" python3 - <<'PY'
import json
import os
print(json.dumps(json.loads(os.environ["QUANTIZATION_CONFIG_JSON"]),
                 separators=(",", ":")))
PY
)"

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

force_env() {
  local force="$1"
  case "${force}" in
    a4) echo "0 0" ;;
    a16) echo "0 1" ;;
    *) echo "unknown force ${force}" >&2; return 2 ;;
  esac
}

variant_image() {
  case "$1" in
    base) echo "${BASE_IMAGE}" ;;
    online) echo "${ONLINE_IMAGE}" ;;
    *) echo "unknown variant $1" >&2; return 2 ;;
  esac
}

variant_label() {
  case "$1" in
    base) echo "luke-nvfp4" ;;
    online) echo "luke-nvfp4-online-mxfp8" ;;
    *) echo "unknown variant $1" >&2; return 2 ;;
  esac
}

common_env_args() {
  local force="$1"
  local dma="$2"
  local force_a8 force_a16
  read -r force_a8 force_a16 < <(force_env "${force}")
  cat <<ARGS
-e
CUDA_DEVICE_ORDER=PCI_BUS_ID
-e
CUDA_DEVICE_MAX_CONNECTIONS=32
-e
CUTE_DSL_ARCH=sm_120a
-e
TORCH_CUDA_ARCH_LIST=12.0a
-e
OMP_NUM_THREADS=16
-e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-e
VLLM_WORKER_MULTIPROC_METHOD=spawn
-e
VLLM_USE_AOT_COMPILE=1
-e
VLLM_USE_BREAKABLE_CUDAGRAPH=0
-e
VLLM_USE_MEGA_AOT_ARTIFACT=1
-e
VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
-e
VLLM_USE_FLASHINFER_SAMPLER=1
-e
VLLM_USE_B12X_WO_PROJECTION=1
-e
VLLM_USE_B12X_MHC=1
-e
VLLM_USE_B12X_FP8_GEMM=1
-e
VLLM_USE_B12X_MOE=1
-e
VLLM_USE_B12X_SPARSE_INDEXER=1
-e
VLLM_USE_V2_MODEL_RUNNER=1
-e
VLLM_ENABLE_PCIE_ALLREDUCE=1
-e
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
-e
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
-e
VLLM_PCIE_DMA_FP8=${dma}
-e
B12X_PCIE_DMA_FP8=${dma}
-e
B12X_MLA_SM120_UNIFIED=1
-e
B12X_DENSE_SPLITK_TURBO=1
-e
B12X_W4A16_TC_DECODE=1
-e
B12X_W4A8_TINY_DECODE=1
-e
B12X_MOE_FORCE_A8=${force_a8}
-e
B12X_MOE_FORCE_A16=${force_a16}
-e
NCCL_PROTO=LL,LL128,Simple
-e
NCCL_P2P_LEVEL=SYS
-e
NCCL_IB_DISABLE=1
-e
LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
-e
VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4
ARGS
}

safe_name() {
  tr '/:,' '---' <<< "$1"
}

config_out_dir() {
  local family="$1"
  local variant="$2"
  local force="$3"
  local mtp="$4"
  local dcp="$5"
  local dma="$6"
  printf '%s/%s/%s/%s/mtp%s/dcp%s/f8-%s\n' \
    "${RESULT_ROOT}" "${family}" "${variant}" "${force}" "${mtp}" "${dcp}" "${dma}"
}

spec_args_for_mtp() {
  local mtp="$1"
  if [[ "${mtp}" == "0" ]]; then
    return 0
  fi
  printf -- '--speculative-config\n'
  printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}\n' "${MODEL}" "${mtp}"
}

assert_gpus_free() {
  local gpu_devices="$1"
  local out="$2"
  local label="$3"
  local gpu_map uuid uuids busy gpu
  local -a gpus

  gpu_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)"
  IFS=',' read -ra gpus <<< "${gpu_devices}"
  uuids=""
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    uuid="$(awk -F', *' -v idx="${gpu}" '$1 == idx {print $2}' <<< "${gpu_map}")"
    if [[ -z "${uuid}" ]]; then
      progress "GLM52_V14_GPU_GUARD_FAILED label=${label} reason=unknown_gpu gpu=${gpu}"
      return 1
    fi
    uuids="${uuids},${uuid}"
  done

  busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r app_uuid pid process_name used_memory; do
    app_uuid="${app_uuid//[[:space:]]/}"
    if [[ ",${uuids}," == *",${app_uuid},"* ]]; then
      printf '%s,%s,%s,%s\n' "${app_uuid}" "${pid//[[:space:]]/}" "${process_name#"${process_name%%[![:space:]]*}"}" "${used_memory//[[:space:]]/}"
    fi
  done)"

  if [[ -n "${busy}" ]]; then
    printf '%s\n' "${busy}" > "${out}/gpu_busy_${label}.csv"
    progress "GLM52_V14_GPU_BUSY label=${label} gpus=${gpu_devices} busy=${out}/gpu_busy_${label}.csv"
    return 1
  fi
}

start_server() {
  local family="$1"
  local variant="$2"
  local force="$3"
  local mtp="$4"
  local dcp="$5"
  local dma="$6"
  local gpu_devices="$7"
  local cuda_visible="$8"
  local port="$9"
  local out="${10}"
  local max_num_seqs="${11}"
  local max_graph="${12}"

  local image label name served
  image="$(variant_image "${variant}")"
  label="$(variant_label "${variant}")"
  name="glm52-v14-$(safe_name "${family}-${variant}-${force}-mtp${mtp}-dcp${dcp}-f8${dma}-p${port}")"
  served="glm52-v14-${label}-${force}-mtp${mtp}-dcp${dcp}-f8${dma}"

  mkdir -p "${out}"
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${served}" > "${out}/served_model.name"
  printf '%s\n' "${image}" > "${out}/image.name"
  printf '%s\n' "${MODEL}" > "${out}/model.path"

  docker rm -f "${name}" >/dev/null 2>&1 || true
  assert_gpus_free "${gpu_devices}" "${out}" "${name}"

  local -a env_args quant_args spec_args
  mapfile -t env_args < <(common_env_args "${force}" "${dma}")
  quant_args=()
  if [[ "${variant}" == "online" ]]; then
    quant_args=(--quantization-config "${QUANTIZATION_CONFIG_JSON}")
    printf '%s\n' "${QUANTIZATION_CONFIG_JSON}" > "${out}/quantization_config.json"
  fi
  mapfile -t spec_args < <(spec_args_for_mtp "${mtp}")

  progress "GLM52_V14_SERVER_START family=${family} variant=${variant} force=${force} mtp=${mtp} dcp=${dcp} f8=${dma} gpus=${gpu_devices} port=${port} out=${out}"
  {
    printf 'docker run ... %s vllm serve %q\n' "${image}" "${MODEL}"
    printf 'served=%s port=%s tp=8 dcp=%s mtp=%s f8=%s max_num_seqs=%s graph=%s\n' "${served}" "${port}" "${dcp}" "${mtp}" "${dma}" "${max_num_seqs}" "${max_graph}"
  } > "${out}/server.cmd"

  docker run -d \
    --name "${name}" \
    --network host \
    --ipc host \
    --privileged \
    --security-opt label=disable \
    --gpus "\"device=${gpu_devices}\"" \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -v /root/models:/root/models:ro \
    -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
    -v /cache:/cache:rw \
    -e CUDA_VISIBLE_DEVICES="${cuda_visible}" \
    "${env_args[@]}" \
    "${image}" \
    vllm serve "${MODEL}" \
      --served-model-name "${served}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --trust-remote-code \
      --tensor-parallel-size 8 \
      --decode-context-parallel-size "${dcp}" \
      --dcp-comm-backend ag_rs \
      --dcp-kv-cache-interleave-size 1 \
      --kv-cache-dtype fp8 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --quantization "${QUANTIZATION}" \
      "${quant_args[@]}" \
      --load-format fastsafetensors \
      -cc.pass_config.fuse_allreduce_rms=True \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${max_num_seqs}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-cudagraph-capture-size "${max_graph}" \
      --async-scheduling \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
      --enable-prompt-tokens-details \
      --enable-force-include-usage \
      --enable-request-id-headers \
      --hf-overrides "${HF_OVERRIDES}" \
      "${spec_args[@]}" > "${out}/container.id"
}

wait_pair_ready() {
  local name_a="$1" port_a="$2" out_a="$3" label_a="$4"
  local name_b="${5:-}" port_b="${6:-}" out_b="${7:-}" label_b="${8:-}"
  local ready_a=0 ready_b=0

  [[ -z "${name_b}" ]] && ready_b=1
  for _ in $(seq 1 1200); do
    if [[ "${ready_a}" == "0" ]] && curl -fsS "http://127.0.0.1:${port_a}/v1/models" > "${out_a}/server.models.json" 2>/dev/null; then
      ready_a=1
      progress "GLM52_V14_SERVER_READY ${label_a} port=${port_a}"
    fi
    if [[ "${ready_b}" == "0" ]] && curl -fsS "http://127.0.0.1:${port_b}/v1/models" > "${out_b}/server.models.json" 2>/dev/null; then
      ready_b=1
      progress "GLM52_V14_SERVER_READY ${label_b} port=${port_b}"
    fi

    if [[ "${ready_a}" == "1" && "${ready_b}" == "1" ]]; then
      docker logs "${name_a}" > "${out_a}/server.ready.log" 2>&1 || true
      if [[ -n "${name_b}" ]]; then
        docker logs "${name_b}" > "${out_b}/server.ready.log" 2>&1 || true
      fi
      progress "GLM52_V14_PAIR_READY ${label_a}${label_b:+ ${label_b}} settle=${SETTLE_SECONDS}s"
      sleep "${SETTLE_SECONDS}"
      return 0
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "${name_a}"; then
      docker logs "${name_a}" > "${out_a}/server.failed.log" 2>&1 || true
      progress "GLM52_V14_SERVER_FAILED ${label_a} reason=exit out=${out_a}"
      return 1
    fi
    if [[ -n "${name_b}" ]] && ! docker ps --format '{{.Names}}' | grep -qx "${name_b}"; then
      docker logs "${name_b}" > "${out_b}/server.failed.log" 2>&1 || true
      progress "GLM52_V14_SERVER_FAILED ${label_b} reason=exit out=${out_b}"
      return 1
    fi
    sleep 2
  done

  progress "GLM52_V14_PAIR_FAILED reason=ready_timeout ${label_a}${label_b:+ ${label_b}}"
  return 1
}

capture_thermal() {
  local out="$1"
  local label="$2"
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv,noheader,nounits > "${out}/thermal_${label}.csv" 2>/dev/null || true
}

summarize_json_one_line() {
  local json="$1"
  python3 - "$json" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    data = json.load(open(path))
except Exception as exc:
    print(f"parse_error={exc}")
    raise SystemExit(0)
summary = data.get("summary_table") or {}
parts = []
for ctx in ("0", "0k", "0.0"):
    row = summary.get(ctx)
    if isinstance(row, dict):
        for cc in ("1", "2", "4", "8", "16", "32"):
            val = row.get(cc)
            if isinstance(val, dict):
                tok = val.get("aggregate_output_tokens_per_second") or val.get("tok_s") or val.get("tokens_per_second")
            else:
                tok = val
            if isinstance(tok, (int, float)):
                parts.append(f"cc{cc}={tok:.2f}")
        break
coding = data.get("coding_peak") or {}
if isinstance(coding, dict):
    tok = coding.get("aggregate_output_tokens_per_second") or coding.get("mean_output_tokens_per_second")
    if not isinstance(tok, (int, float)):
        summary = coding.get("summary") or {}
        tok = summary.get("mean_generation_tok_s")
    if isinstance(tok, (int, float)):
        parts.append(f"coding={tok:.2f}")
print(" ".join(parts) if parts else "no_summary")
PY
}

run_table_bench() {
  local port="$1"
  local out="$2"
  local label="$3"
  local served
  served="$(<"${out}/served_model.name")"

  capture_thermal "${out}" "before_decode"
  progress "GLM52_V14_TABLE_DECODE_START ${label} port=${port}"
  python3 /root/llm-inference-bench/llm_decode_bench.py \
    --port "${port}" \
    --model "${served}" \
    --skip-prefill \
    --contexts 0 \
    --concurrency 1 \
    --duration "${TABLE_DECODE_DURATION}" \
    --max-tokens "${TABLE_DECODE_MAX_TOKENS}" \
    --coding-peak \
    --coding-peak-runs "${TABLE_CODING_PEAK_RUNS}" \
    --display-mode plain \
    --no-hw-monitor \
    --output "${out}/decode_table.json" > "${out}/decode_table.txt" 2>&1
  capture_thermal "${out}" "after_decode"
  progress "GLM52_V14_TABLE_DECODE_DONE ${label} $(summarize_json_one_line "${out}/decode_table.json") out=${out}"

  progress "GLM52_V14_TABLE_PREFILL_START ${label} port=${port}"
  python3 /root/llm-inference-bench/llm_decode_bench.py \
    --port "${port}" \
    --model "${served}" \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "${TABLE_PREFILL_CONTEXTS}" \
    --token-targeting exact \
    --calibration-cache "${out}/calibration_table.json" \
    --max-tokens 1 \
    --display-mode plain \
    --no-hw-monitor \
    --output "${out}/prefill_table.json" > "${out}/prefill_table.txt" 2>&1
  capture_thermal "${out}" "after_prefill"
  progress "GLM52_V14_TABLE_PREFILL_DONE ${label} out=${out}"
}

run_full_decode_bench() {
  local port="$1"
  local out="$2"
  local label="$3"
  local dcp="$4"
  local served
  served="$(<"${out}/served_model.name")"

  capture_thermal "${out}" "before_full_decode"
  progress "GLM52_V14_FULL_DECODE_START ${label} port=${port} contexts=${DECODE_CONTEXTS} cc=${DECODE_CONCURRENCY}"
  python3 /root/llm-inference-bench/llm_decode_bench.py \
    --port "${port}" \
    --model "${served}" \
    --contexts "${DECODE_CONTEXTS}" \
    --concurrency "${DECODE_CONCURRENCY}" \
    --duration "${DECODE_DURATION}" \
    --max-tokens "${DECODE_MAX_TOKENS}" \
    --dcp-size "${dcp}" \
    --display-mode plain \
    --no-hw-monitor \
    --output "${out}/decode_full.json" > "${out}/decode_full.txt" 2>&1
  capture_thermal "${out}" "after_full_decode"
  progress "GLM52_V14_FULL_DECODE_DONE ${label} $(summarize_json_one_line "${out}/decode_full.json") out=${out}"
}

run_full_prefill_bench() {
  local port="$1"
  local out="$2"
  local label="$3"
  local served
  served="$(<"${out}/served_model.name")"

  progress "GLM52_V14_FULL_PREFILL_START ${label} port=${port} contexts=${PREFILL_CONTEXTS}"
  python3 /root/llm-inference-bench/llm_decode_bench.py \
    --port "${port}" \
    --model "${served}" \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "${PREFILL_CONTEXTS}" \
    --prefill-duration "${PREFILL_DURATION}" \
    --display-mode plain \
    --no-hw-monitor \
    --output "${out}/prefill_full.json" > "${out}/prefill_full.txt" 2>&1
  capture_thermal "${out}" "after_full_prefill"
  progress "GLM52_V14_FULL_PREFILL_DONE ${label} out=${out}"
}

stop_server_from_out() {
  local out="$1"
  local name
  [[ -f "${out}/container.name" ]] || return 0
  name="$(<"${out}/container.name")"
  docker logs "${name}" > "${out}/server.final.log" 2>&1 || true
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

run_pair() {
  local family="$1"
  local cfg_a="$2"
  local cfg_b="${3:-}"
  local max_num_seqs="$4"
  local max_graph="$5"
  local bench_kind="$6"

  local variant_a force_a mtp_a dcp_a dma_a out_a name_a label_a
  IFS='|' read -r variant_a force_a mtp_a dcp_a dma_a <<< "${cfg_a}"
  out_a="$(config_out_dir "${family}" "${variant_a}" "${force_a}" "${mtp_a}" "${dcp_a}" "${dma_a}")"
  label_a="${variant_a}/${force_a}/mtp${mtp_a}/dcp${dcp_a}/f8${dma_a}"

  local variant_b="" force_b="" mtp_b="" dcp_b="" dma_b="" out_b="" name_b="" label_b=""
  if [[ -n "${cfg_b}" ]]; then
    IFS='|' read -r variant_b force_b mtp_b dcp_b dma_b <<< "${cfg_b}"
    out_b="$(config_out_dir "${family}" "${variant_b}" "${force_b}" "${mtp_b}" "${dcp_b}" "${dma_b}")"
    label_b="${variant_b}/${force_b}/mtp${mtp_b}/dcp${dcp_b}/f8${dma_b}"
  fi

  start_server "${family}" "${variant_a}" "${force_a}" "${mtp_a}" "${dcp_a}" "${dma_a}" "${GPU_A}" "${CUDA_A}" "${PORT_A}" "${out_a}" "${max_num_seqs}" "${max_graph}"
  name_a="$(<"${out_a}/container.name")"
  if [[ -n "${cfg_b}" ]]; then
    start_server "${family}" "${variant_b}" "${force_b}" "${mtp_b}" "${dcp_b}" "${dma_b}" "${GPU_B}" "${CUDA_B}" "${PORT_B}" "${out_b}" "${max_num_seqs}" "${max_graph}"
    name_b="$(<"${out_b}/container.name")"
  fi

  wait_pair_ready "${name_a}" "${PORT_A}" "${out_a}" "${label_a}" "${name_b}" "${PORT_B}" "${out_b}" "${label_b}"

  if [[ "${bench_kind}" == "table" ]]; then
    run_table_bench "${PORT_A}" "${out_a}" "${label_a}" &
  else
    run_full_decode_bench "${PORT_A}" "${out_a}" "${label_a}" "${dcp_a}" &
  fi
  local pid_a=$!

  local pid_b=""
  if [[ -n "${cfg_b}" ]]; then
    if [[ "${bench_kind}" == "table" ]]; then
      run_table_bench "${PORT_B}" "${out_b}" "${label_b}" &
    else
      run_full_decode_bench "${PORT_B}" "${out_b}" "${label_b}" "${dcp_b}" &
    fi
    pid_b=$!
  fi

  if [[ -n "${pid_b}" ]]; then
    wait "${pid_a}" "${pid_b}"
  else
    wait "${pid_a}"
  fi

  if [[ "${bench_kind}" != "table" && "${RUN_STANDALONE_PREFILL}" == "1" ]]; then
    run_full_prefill_bench "${PORT_A}" "${out_a}" "${label_a}" &
    pid_a=$!
    pid_b=""
    if [[ -n "${cfg_b}" ]]; then
      run_full_prefill_bench "${PORT_B}" "${out_b}" "${label_b}" &
      pid_b=$!
    fi
    if [[ -n "${pid_b}" ]]; then
      wait "${pid_a}" "${pid_b}"
    else
      wait "${pid_a}"
    fi
  fi

  stop_server_from_out "${out_a}"
  if [[ -n "${out_b}" ]]; then
    stop_server_from_out "${out_b}"
  fi
}

table_todo_configs() {
  cat <<'EOF'
base|a4|0|1|0
base|a16|0|1|0
online|a4|0|1|0
online|a4|0|1|ag
online|a4|0|1|ring
EOF
}

full_sweep_configs() {
  local variant force mtp dcp dma
  for variant in base online; do
    for force in a4 a16; do
      for mtp in 0 3; do
        for dcp in 1 2 4 8; do
          printf '%s|%s|%s|%s|0\n' "${variant}" "${force}" "${mtp}" "${dcp}"
        done
      done
    done
  done

  for variant in base online; do
    for dma in ag ring; do
      for dcp in 1 2 4 8; do
        printf '%s|a4|3|%s|%s\n' "${variant}" "${dcp}" "${dma}"
      done
    done
  done
}

run_config_list() {
  local family="$1"
  local max_num_seqs="$2"
  local max_graph="$3"
  local bench_kind="$4"
  shift 4
  local -a configs=("$@")
  local i cfg_b

  for ((i=0; i<${#configs[@]}; i+=2)); do
    cfg_b=""
    if (( i + 1 < ${#configs[@]} )); then
      cfg_b="${configs[i+1]}"
    fi
    run_pair "${family}" "${configs[i]}" "${cfg_b}" "${max_num_seqs}" "${max_graph}" "${bench_kind}"
  done
}

run_table_todos() {
  local -a configs
  mapfile -t configs < <(table_todo_configs)
  progress "GLM52_V14_TABLE_TODOS_START result_root=${RESULT_ROOT} kld_root=${KLD_ROOT}"
  run_config_list table "${TABLE_MAX_NUM_SEQS}" "${TABLE_MAX_CUDAGRAPH_CAPTURE_SIZE}" table "${configs[@]}"
  progress "GLM52_V14_TABLE_TODOS_DONE result_root=${RESULT_ROOT}"
}

kld_llm_extra_json() {
  local variant="$1"
  if [[ "${variant}" == "online" ]]; then
    printf '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"quantization_config":%s}' "${QUANTIZATION_CONFIG_JSON}"
  else
    printf '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true}'
  fi
}

run_kld_one() {
  local variant="$1"
  local force="$2"
  local dma="$3"
  local gpu_devices="$4"
  local run="$5"
  local label="${variant}/${force}/f8${dma}/run${run}"
  local out="${KLD_ROOT}/${variant}/${force}/f8-${dma}/run${run}"
  local image force_a8 force_a16 llm_extra name
  image="$(variant_image "${variant}")"
  read -r force_a8 force_a16 < <(force_env "${force}")
  llm_extra="$(kld_llm_extra_json "${variant}")"
  name="glm52-v14-kld-$(safe_name "${variant}-${force}-f8${dma}-run${run}")"

  mkdir -p "${out}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  assert_gpus_free "${gpu_devices}" "${out}" "${name}"
  progress "GLM52_V14_KLD_START ${label} gpus=${gpu_devices} out=${out}"

  docker run --rm --name "${name}" \
    --gpus "\"device=${gpu_devices}\"" \
    --ipc=host \
    --network=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -v /root/models:/root/models:ro \
    -v /root/kld:/root/kld:rw \
    -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
    -v /cache:/cache:rw \
    -v /cache/kld_pydeps:/cache/kld_pydeps:ro \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTHONPATH=/root/kld:/cache/kld_pydeps \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e CUTE_DSL_ARCH=sm_120a \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e OMP_NUM_THREADS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 \
    -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
    -e VLLM_PCIE_DMA_FP8="${dma}" \
    -e B12X_PCIE_DMA_FP8="${dma}" \
    -e B12X_MLA_SM120_UNIFIED=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_W4A16_TC_DECODE=1 \
    -e B12X_W4A8_TINY_DECODE=1 \
    -e B12X_MOE_FORCE_A8="${force_a8}" \
    -e B12X_MOE_FORCE_A16="${force_a16}" \
    -e NCCL_PROTO=LL,LL128,Simple \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_IB_DISABLE=1 \
    -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
    -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
    -e LLM_EXTRA_JSON="${llm_extra}" \
    --entrypoint bash \
    "${image}" \
    -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; /opt/venv/bin/python /root/kld/prefill_kld_fallback.py \
      --model '${MODEL}' \
      --reference-logits '${PREFILL_REF}' \
      --context-length 2048 \
      --stride 512 \
      --max-windows 1 \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.74 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 512 \
      --max-num-seqs 1 \
      --quantization '${QUANTIZATION}' \
      --attention-backend B12X_MLA_SPARSE \
      --hf-overrides '${HF_OVERRIDES}' \
      --llm-extra-json \"\${LLM_EXTRA_JSON}\" \
      --kld-chunk-rows 32 2>&1 | tee '${out}/prefill_dcp1.log'"
  progress "GLM52_V14_KLD_DONE ${label} out=${out}"
}

run_kld_todos() {
  local -a configs
  mapfile -t configs < <(table_todo_configs)
  local run i cfg_a cfg_b variant_a force_a _mtp_a _dcp_a dma_a variant_b force_b _mtp_b _dcp_b dma_b

  progress "GLM52_V14_KLD_TODOS_START runs=${KLD_RUNS} kld_root=${KLD_ROOT}"
  for run in $(seq 1 "${KLD_RUNS}"); do
    for ((i=0; i<${#configs[@]}; i+=2)); do
      cfg_a="${configs[i]}"
      IFS='|' read -r variant_a force_a _mtp_a _dcp_a dma_a <<< "${cfg_a}"
      run_kld_one "${variant_a}" "${force_a}" "${dma_a}" "${GPU_A}" "${run}" &
      local pid_a=$!

      if (( i + 1 < ${#configs[@]} )); then
        cfg_b="${configs[i+1]}"
        IFS='|' read -r variant_b force_b _mtp_b _dcp_b dma_b <<< "${cfg_b}"
        run_kld_one "${variant_b}" "${force_b}" "${dma_b}" "${GPU_B}" "${run}" &
        local pid_b=$!
        wait "${pid_a}" "${pid_b}"
      else
        wait "${pid_a}"
      fi
    done
  done
  progress "GLM52_V14_KLD_TODOS_DONE kld_root=${KLD_ROOT}"
}

run_full_sweep() {
  local -a configs
  mapfile -t configs < <(full_sweep_configs)
  progress "GLM52_V14_FULL_SWEEP_START configs=${#configs[@]} result_root=${RESULT_ROOT} max_num_seqs=${SWEEP_MAX_NUM_SEQS} graph=${SWEEP_MAX_CUDAGRAPH_CAPTURE_SIZE}"
  run_config_list full "${SWEEP_MAX_NUM_SEQS}" "${SWEEP_MAX_CUDAGRAPH_CAPTURE_SIZE}" full "${configs[@]}"
  progress "GLM52_V14_FULL_SWEEP_DONE result_root=${RESULT_ROOT}"
}

summarize_results() {
  python3 - "${RESULT_ROOT}" "${KLD_ROOT}" <<'PY'
import json
import math
import re
import statistics
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
kld_root = Path(sys.argv[2])

def num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def get_tok(cell):
    if isinstance(cell, dict):
        for key in ("aggregate_output_tokens_per_second", "tok_s", "tokens_per_second"):
            val = cell.get(key)
            if num(val):
                return float(val)
    if num(cell):
        return float(cell)
    return None

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def summary_value(path, ctx, cc):
    data = load_json(path)
    if not data:
        return None
    table = data.get("summary_table") or {}
    row = table.get(str(ctx))
    if row is None and str(ctx) == "0":
        row = table.get("0k")
    if not isinstance(row, dict):
        return None
    return get_tok(row.get(str(cc)))

def prefill_values(path):
    data = load_json(path)
    vals = {}
    if not data:
        return vals
    prefill = data.get("prefill") or {}
    if isinstance(prefill, dict):
        items = []
        for ctx, item in prefill.items():
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("context", ctx)
                items.append(item)
    elif isinstance(prefill, list):
        items = prefill
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ctx = item.get("context") or item.get("context_length") or item.get("ctx")
        val = (
            item.get("tokens_per_second")
            or item.get("tok_s")
            or item.get("tok_per_sec")
            or item.get("client_tokens_per_second")
            or item.get("client_tok_per_sec")
        )
        if ctx is not None and num(val):
            vals[str(ctx)] = float(val)
    return vals

def coding_peak(path):
    data = load_json(path)
    if not data:
        return None
    cp = data.get("coding_peak") or {}
    for key in ("aggregate_output_tokens_per_second", "mean_output_tokens_per_second", "tok_s"):
        val = cp.get(key)
        if num(val):
            return float(val)
    summary = cp.get("summary") or {}
    if num(summary.get("mean_generation_tok_s")):
        return float(summary["mean_generation_tok_s"])
    runs = cp.get("runs")
    if isinstance(runs, list):
        vals = []
        for r in runs:
            if isinstance(r, dict):
                for key in ("output_tokens_per_second", "tokens_per_second", "tok_s"):
                    if num(r.get(key)):
                        vals.append(float(r[key]))
                        break
        if vals:
            return statistics.mean(vals)
    return None

def kld_values():
    values = {}
    for log in kld_root.glob("*/*/f8-*/*/prefill_dcp1.log"):
        rel = log.relative_to(kld_root).parts
        if len(rel) < 5:
            continue
        variant, force, f8dir, run = rel[:4]
        dma = f8dir.replace("f8-", "")
        text = log.read_text(errors="ignore")
        found = None
        for pat in (r"mean[_ ]kld[=: ]+([0-9.]+)", r"kld_mean[=: ]+([0-9.]+)", r"\bmean\b[^0-9]+([0-9]+\.[0-9]+)"):
            m = re.search(pat, text, re.I)
            if m:
                found = float(m.group(1))
                break
        if found is None:
            nums = [float(x) for x in re.findall(r"\bKLD\b[^0-9]+([0-9]+\.[0-9]+)", text, re.I)]
            if nums:
                found = nums[-1]
        if found is not None:
            values.setdefault((variant, force, dma), []).append(found)
    return values

def fmt(x):
    if x is None:
        return "TODO"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.2f}"

klds = kld_values()
lines = []
lines.append(f"# GLM-5.2 v14 Sweep Summary\n")
lines.append(f"Result root: `{result_root}`  ")
lines.append(f"KLD root: `{kld_root}`\n")

lines.append("## Table TODO Measurements\n")
lines.append("| Variant | Mode | f8 | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | KLD mean +/- sd |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
for variant, force, dma in [
    ("base", "a4", "0"),
    ("base", "a16", "0"),
    ("online", "a4", "0"),
    ("online", "a4", "ag"),
    ("online", "a4", "ring"),
]:
    path = result_root / "table" / variant / force / "mtp0" / "dcp1" / f"f8-{dma}"
    dec = summary_value(path / "decode_table.json", "0", "1")
    cp = coding_peak(path / "decode_table.json")
    pref = prefill_values(path / "prefill_table.json")
    kvals = klds.get((variant, force, dma), [])
    if kvals:
        kld = f"{statistics.mean(kvals):.5f} +/- {(statistics.stdev(kvals) if len(kvals) > 1 else 0.0):.5f}"
    else:
        kld = "TODO"
    lines.append(f"| {variant} | {force.upper()} | `{dma}` | {fmt(dec)} | {fmt(cp)} | {fmt(pref.get('30720') or pref.get('30000') or pref.get('30k'))} | {fmt(pref.get('65536') or pref.get('64000') or pref.get('64k'))} | {fmt(pref.get('122880') or pref.get('120000') or pref.get('120k'))} | {kld} |")

def collect_full_keys():
    keys = []
    base = result_root / "full"
    for path in sorted(base.glob("*/*/mtp*/dcp*/f8-*/decode_full.json")):
        rel = path.relative_to(base).parts
        if len(rel) < 6:
            continue
        variant, force, mtpdir, dcpdir, f8dir, _ = rel
        keys.append((variant, force, mtpdir.replace("mtp", ""), dcpdir.replace("dcp", ""), f8dir.replace("f8-", ""), path))
    return keys

keys = collect_full_keys()
for variant in ("base", "online"):
    for force in ("a4", "a16"):
        for dma in ("0", "ag", "ring"):
            subset = [k for k in keys if k[0] == variant and k[1] == force and k[4] == dma]
            if not subset:
                continue
            lines.append(f"\n## Full Sweep ctx0 Aggregate tok/s: {variant} {force.upper()} f8={dma}\n")
            lines.append("| MTP | DCP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for mtp in ("0", "3"):
                for dcp in ("1", "2", "4", "8"):
                    item = next((x for x in subset if x[2] == mtp and x[3] == dcp), None)
                    if not item:
                        continue
                    path = item[5]
                    vals = [fmt(summary_value(path, "0", cc)) for cc in ("1", "2", "4", "8", "16", "32")]
                    lines.append(f"| {mtp} | {dcp} | " + " | ".join(vals) + " |")

            lines.append(f"\n## Standalone Prefill tok/s: {variant} {force.upper()} f8={dma}\n")
            lines.append("Standalone prefill stores the contexts that fit under the 131,072-token model length; the requested 128k row is skipped by `llm_decode_bench` for this model.\n")
            lines.append("| MTP | DCP | 8k | 64k |")
            lines.append("|---:|---:|---:|---:|")
            for mtp in ("0", "3"):
                for dcp in ("1", "2", "4", "8"):
                    item = next((x for x in subset if x[2] == mtp and x[3] == dcp), None)
                    if not item:
                        continue
                    pref = prefill_values(item[5].with_name("prefill_full.json"))
                    vals = [
                        fmt(pref.get("8192") or pref.get("8000") or pref.get("8k")),
                        fmt(pref.get("65536") or pref.get("64000") or pref.get("64k")),
                    ]
                    lines.append(f"| {mtp} | {dcp} | " + " | ".join(vals) + " |")

out = result_root / "summary.md"
out.write_text("\n".join(lines) + "\n")
print(out)
PY
}

ACTION="${1:-all}"
if [[ "${ACTION}" != "summarize" ]]; then
  progress "GLM52_V14_SCRIPT_START result_root=${RESULT_ROOT} kld_root=${KLD_ROOT} base_image=${BASE_IMAGE} online_image=${ONLINE_IMAGE} quant_config=${QUANTIZATION_CONFIG_JSON}"
fi

case "${ACTION}" in
  table-todos)
    run_table_todos
    summarize_results
    ;;
  kld-todos)
    run_kld_todos
    summarize_results
    ;;
  full-sweep)
    run_full_sweep
    summarize_results
    ;;
  summarize)
    summarize_results
    ;;
  all)
    run_table_todos
    run_kld_todos
    run_full_sweep
    summarize_results
    ;;
  *)
    echo "usage: $0 [all|table-todos|kld-todos|full-sweep|summarize]" >&2
    exit 2
    ;;
esac

if [[ "${ACTION}" != "summarize" ]]; then
  progress "GLM52_V14_SCRIPT_DONE result_root=${RESULT_ROOT} kld_root=${KLD_ROOT}"
fi
