#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v5-vllmcd272c7-b12xe44cb77-cu132-20260707}"
LUKE_MODEL="${LUKE_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
PREFILL_REF="${PREFILL_REF:-/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
KLD_ROOT="${KLD_ROOT:-/root/kld/glm52_v14_keypoints_${RUN_ID}}"
RUNS="${RUNS:-5}"
GPU_A="${GPU_A:-0,1,2,3,4,5,6,7}"
GPU_B="${GPU_B:-8,9,10,11,12,13,14,15}"
PATTERN="${PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.74}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
KLD_CHUNK_ROWS="${KLD_CHUNK_ROWS:-32}"
F8_DMA="${F8_DMA:-0}"
CASES="${CASES:-luke-a4-orig luke-a16-orig luke-a4-online-mxfp8 luke-a16-online-mxfp8 mxfp4-a8-orig mxfp4-a8-online-mxfp8}"

if [[ "${#PATTERN}" -ne 78 ]]; then
  echo "ERROR: index_topk_pattern must be 78 chars, got ${#PATTERN}" >&2
  exit 2
fi

HF_OVERRIDES="$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "${PATTERN}")"
ONLINE_MXFP8_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}'

mkdir -p "${KLD_ROOT}"
printf '%s\n' "${KLD_ROOT}" > /root/kld/latest_glm52_v14_keypoints.out

case_force_env() {
  case "$1" in
    a4) echo "0 0" ;;
    a16) echo "0 1" ;;
    a8) echo "1 0" ;;
    *) echo "unknown moe mode $1" >&2; return 2 ;;
  esac
}

case_model() {
  case "$1" in
    luke-a4-orig|luke-a16-orig|luke-a4-online-mxfp8|luke-a16-online-mxfp8)
      echo "${LUKE_MODEL}" ;;
    mxfp4-a8-orig|mxfp4-a8-online-mxfp8)
      echo "${MXFP4_MODEL}" ;;
    *) echo "unknown case $1" >&2; return 2 ;;
  esac
}

case_quantization() {
  case "$1" in
    luke-*) echo "modelopt_fp4" ;;
    mxfp4-*) echo "mxfp4" ;;
    *) echo "unknown case $1" >&2; return 2 ;;
  esac
}

case_moe_mode() {
  case "$1" in
    luke-a4-*) echo "a4" ;;
    luke-a16-*) echo "a16" ;;
    mxfp4-a8-*) echo "a8" ;;
    *) echo "unknown case $1" >&2; return 2 ;;
  esac
}

case_online_config() {
  case "$1" in
    *online-mxfp8) echo "${ONLINE_MXFP8_CONFIG}" ;;
    *) echo "" ;;
  esac
}

llm_extra_json() {
  local quant_config="$1"
  if [[ -n "${quant_config}" ]]; then
    QUANT_CONFIG="${quant_config}" python3 - <<'PY'
import json, os
print(json.dumps({
    "decode_context_parallel_size": 1,
    "moe_backend": "b12x",
    "enforce_eager": True,
    "quantization_config": json.loads(os.environ["QUANT_CONFIG"]),
}, separators=(",", ":")))
PY
  else
    printf '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true}\n'
  fi
}

run_one() {
  local case_name="$1"
  local run="$2"
  local gpu_devices="$3"
  local model quantization moe_mode quant_config llm_extra force_a8 force_a16 out name

  model="$(case_model "${case_name}")"
  quantization="$(case_quantization "${case_name}")"
  moe_mode="$(case_moe_mode "${case_name}")"
  quant_config="$(case_online_config "${case_name}")"
  llm_extra="$(llm_extra_json "${quant_config}")"
  read -r force_a8 force_a16 < <(case_force_env "${moe_mode}")

  out="${KLD_ROOT}/${case_name}/run${run}"
  name="glm52-v14-kld-${case_name}-f8${F8_DMA}-run${run}"
  mkdir -p "${out}"
  docker rm -f "${name}" >/dev/null 2>&1 || true

  cat > "${out}/config.json" <<EOF
{
  "case": "${case_name}",
  "run": ${run},
  "image": "${IMAGE}",
  "model": "${model}",
  "quantization": "${quantization}",
  "moe_mode": "${moe_mode}",
  "online_quantization_config": ${quant_config:-null},
  "load_format": "${LOAD_FORMAT}",
  "instanttensor_backend": "${INSTANTTENSOR_BACKEND}",
  "f8_dma": "${F8_DMA}",
  "gpus": "${gpu_devices}",
  "reference_logits": "${PREFILL_REF}",
  "hf_overrides": ${HF_OVERRIDES}
}
EOF

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_START case=${case_name} run=${run} gpus=${gpu_devices} out=${out}"

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
    -e SAFETENSORS_FAST_GPU=1 \
    -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND}" \
    -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 \
    -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_B12X_DCP_A2A=1 \
    -e VLLM_DCP_A2A_MAX_TOKENS=64 \
    -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
    -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB \
    -e VLLM_PCIE_DMA_FP8="${F8_DMA}" \
    -e B12X_PCIE_DMA_FP8="${F8_DMA}" \
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
    "${IMAGE}" \
    -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; /opt/venv/bin/python /root/kld/prefill_kld_fallback.py \
      --model '${model}' \
      --reference-logits '${PREFILL_REF}' \
      --context-length 2048 \
      --stride 512 \
      --max-windows 1 \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format '${LOAD_FORMAT}' \
      --max-model-len 4096 \
      --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
      --max-num-seqs 1 \
      --quantization '${quantization}' \
      --attention-backend B12X_MLA_SPARSE \
      --hf-overrides '${HF_OVERRIDES}' \
      --llm-extra-json \"\${LLM_EXTRA_JSON}\" \
      --kld-chunk-rows '${KLD_CHUNK_ROWS}' 2>&1 | tee '${out}/prefill_dcp1.log'"

  python3 - "${out}" <<'PY'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
text = (out / "prefill_dcp1.log").read_text(errors="replace")
m = re.search(r"fallback_prefill_kld_done\s+(\{.*\})", text)
if not m:
    raise SystemExit(f"missing fallback_prefill_kld_done in {out}")
summary = json.loads((out / "config.json").read_text())
summary["prefill"] = json.loads(m.group(1))
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, sort_keys=True))
PY

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_DONE case=${case_name} run=${run} out=${out}"
}

summarize() {
  python3 - "${KLD_ROOT}" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    vals = []
    configs = []
    for summary_path in sorted(case_dir.glob("run*/summary.json")):
        data = json.loads(summary_path.read_text())
        vals.append(float(data["prefill"]["mean_kld"]))
        configs.append(data)
    if not vals:
        continue
    first = configs[0]
    rows.append({
        "case": case_dir.name,
        "model": first["model"],
        "quantization": first["quantization"],
        "moe_mode": first["moe_mode"],
        "online": first["online_quantization_config"] is not None,
        "runs": len(vals),
        "mean_kld": statistics.mean(vals),
        "sd_kld": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min_kld": min(vals),
        "max_kld": max(vals),
        "values": vals,
    })

aggregate = {"root": str(root), "rows": rows}
(root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))

lines = [
    "| Case | Quantization | MoE mode | Online MXFP8 | Runs | KLD mean +/- sd | Min | Max |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['case']} | `{row['quantization']}` | {row['moe_mode'].upper()} | "
        f"{'yes' if row['online'] else 'no'} | {row['runs']} | "
        f"{row['mean_kld']:.5f} +/- {row['sd_kld']:.5f} | "
        f"{row['min_kld']:.5f} | {row['max_kld']:.5f} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

run_all() {
  local -a cases
  read -r -a cases <<< "${CASES}"

  for run in $(seq 1 "${RUNS}"); do
    local i case_a case_b pid_a pid_b
    for ((i=0; i<${#cases[@]}; i+=2)); do
      case_a="${cases[i]}"
      case_b="${cases[i+1]:-}"
      run_one "${case_a}" "${run}" "${GPU_A}" &
      pid_a=$!
      if [[ -n "${case_b}" ]]; then
        run_one "${case_b}" "${run}" "${GPU_B}" &
        pid_b=$!
        wait "${pid_a}" "${pid_b}"
      else
        wait "${pid_a}"
      fi
    done
  done
  summarize
}

case "${1:-all}" in
  all) run_all ;;
  summarize) summarize ;;
  *)
    echo "usage: $0 [all|summarize]" >&2
    exit 2
    ;;
esac
