#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}"
MODEL="${MODEL:?MODEL is required}"
TOKENIZER="${TOKENIZER:-}"
REF="${REF:-/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref}"
OUT_ROOT="${OUT_ROOT:-/root/kld/glm52_scoremode_kld_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
REF_OUTPUT="${REF_OUTPUT:-${OUT_ROOT}/ref}"
LOGITS_OVERLAY="${LOGITS_OVERLAY:-/root/vllm/worktrees/vllm-release-kld-logits-export-20260622}"
OVERLAY_MODEL_PY="${OVERLAY_MODEL_PY:-}"
OVERLAY_DEEPSEEK_V2_PY="${OVERLAY_DEEPSEEK_V2_PY:-}"
OVERLAY_DEEPGEMM_MOE_PY="${OVERLAY_DEEPGEMM_MOE_PY:-}"
OVERLAY_MXFP8_ORACLE_PY="${OVERLAY_MXFP8_ORACLE_PY:-}"
OVERLAY_MXFP4_ORACLE_PY="${OVERLAY_MXFP4_ORACLE_PY:-}"
OVERLAY_OCP_MX_EMULATION_MOE_PY="${OVERLAY_OCP_MX_EMULATION_MOE_PY:-}"
OVERLAY_NVFP4_ORACLE_PY="${OVERLAY_NVFP4_ORACLE_PY:-}"
OVERLAY_NVFP4_EMULATION_MOE_PY="${OVERLAY_NVFP4_EMULATION_MOE_PY:-}"
OVERLAY_MODELOPT_PY="${OVERLAY_MODELOPT_PY:-}"
OVERLAY_GGUF_PY="${OVERLAY_GGUF_PY:-}"
OVERLAY_GGUF_LOADER_INIT_PY="${OVERLAY_GGUF_LOADER_INIT_PY:-}"
OVERLAY_GGUF_LOADER_PY="${OVERLAY_GGUF_LOADER_PY:-}"
OVERLAY_WEIGHT_UTILS_PY="${OVERLAY_WEIGHT_UTILS_PY:-}"
OVERLAY_GGUF_UTILS_PY="${OVERLAY_GGUF_UTILS_PY:-}"
OVERLAY_QUANT_INIT_PY="${OVERLAY_QUANT_INIT_PY:-}"
OVERLAY_LINEAR_PY="${OVERLAY_LINEAR_PY:-}"
OVERLAY_GGUF_DEQUANT_EXT_SO="${OVERLAY_GGUF_DEQUANT_EXT_SO:-}"
OVERLAY_VLLM_C_SO="${OVERLAY_VLLM_C_SO:-}"
OVERLAY_VLLM_C_STABLE_SO="${OVERLAY_VLLM_C_STABLE_SO:-}"
OVERLAY_QUTLASS_SO="${OVERLAY_QUTLASS_SO:-}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
CUDA_VISIBLE_DEVICES_INNER="${CUDA_VISIBLE_DEVICES_INNER:-${GPU_DEVICES}}"
TP="${TP:-16}"
DCP="${DCP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-80}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
QUANTIZATION="${QUANTIZATION:-}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
CONTAINER_NAME="${CONTAINER_NAME:-glm52-scoremode-kld-probe}"
PATTERN="${PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES="${HF_OVERRIDES:-$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "${PATTERN}")}"
LLM_EXTRA_JSON="${LLM_EXTRA_JSON:-$(printf '{"decode_context_parallel_size":%s}' "${DCP}")}"
PROBE_RUNNER="${PROBE_RUNNER:-scoremode}"
FALLBACK_QUANTIZATION="${FALLBACK_QUANTIZATION:-${QUANTIZATION:-auto}}"
FALLBACK_LLM_EXTRA_JSON="${FALLBACK_LLM_EXTRA_JSON:-$(printf '{"decode_context_parallel_size":%s,"moe_backend":"%s","enforce_eager":true,"disable_custom_all_reduce":true,"cpu_offload_gb":%s}' "${DCP}" "${MOE_BACKEND}" "${CPU_OFFLOAD_GB}")}"

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/scoremode_kld.log"
CONFIG="${OUT_ROOT}/config.env"

cat >"${CONFIG}" <<EOF
IMAGE=${IMAGE}
MODEL=${MODEL}
TOKENIZER=${TOKENIZER}
REF=${REF}
REF_OUTPUT=${REF_OUTPUT}
OUT_ROOT=${OUT_ROOT}
LOGITS_OVERLAY=${LOGITS_OVERLAY}
OVERLAY_MODEL_PY=${OVERLAY_MODEL_PY}
OVERLAY_DEEPSEEK_V2_PY=${OVERLAY_DEEPSEEK_V2_PY}
OVERLAY_DEEPGEMM_MOE_PY=${OVERLAY_DEEPGEMM_MOE_PY}
OVERLAY_MXFP8_ORACLE_PY=${OVERLAY_MXFP8_ORACLE_PY}
OVERLAY_MXFP4_ORACLE_PY=${OVERLAY_MXFP4_ORACLE_PY}
OVERLAY_OCP_MX_EMULATION_MOE_PY=${OVERLAY_OCP_MX_EMULATION_MOE_PY}
OVERLAY_NVFP4_ORACLE_PY=${OVERLAY_NVFP4_ORACLE_PY}
OVERLAY_NVFP4_EMULATION_MOE_PY=${OVERLAY_NVFP4_EMULATION_MOE_PY}
OVERLAY_MODELOPT_PY=${OVERLAY_MODELOPT_PY}
OVERLAY_GGUF_PY=${OVERLAY_GGUF_PY}
OVERLAY_GGUF_LOADER_INIT_PY=${OVERLAY_GGUF_LOADER_INIT_PY}
OVERLAY_GGUF_LOADER_PY=${OVERLAY_GGUF_LOADER_PY}
OVERLAY_WEIGHT_UTILS_PY=${OVERLAY_WEIGHT_UTILS_PY}
OVERLAY_GGUF_UTILS_PY=${OVERLAY_GGUF_UTILS_PY}
OVERLAY_QUANT_INIT_PY=${OVERLAY_QUANT_INIT_PY}
OVERLAY_LINEAR_PY=${OVERLAY_LINEAR_PY}
OVERLAY_GGUF_DEQUANT_EXT_SO=${OVERLAY_GGUF_DEQUANT_EXT_SO}
OVERLAY_VLLM_C_SO=${OVERLAY_VLLM_C_SO}
OVERLAY_VLLM_C_STABLE_SO=${OVERLAY_VLLM_C_STABLE_SO}
OVERLAY_QUTLASS_SO=${OVERLAY_QUTLASS_SO}
VLLM_GGUF_DEQUANT_AT_LOAD=${VLLM_GGUF_DEQUANT_AT_LOAD:-0}
VLLM_GGUF_DEQUANT_CPU_OFFLOAD_GB=${VLLM_GGUF_DEQUANT_CPU_OFFLOAD_GB:-0}
VLLM_GGUF_DEQUANT_UVA_OFFLOAD_GB=${VLLM_GGUF_DEQUANT_UVA_OFFLOAD_GB:-0}
GPU_DEVICES=${GPU_DEVICES}
TP=${TP}
DCP=${DCP}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB}
LOAD_FORMAT=${LOAD_FORMAT}
DTYPE=${DTYPE}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
QUANTIZATION=${QUANTIZATION}
MOE_BACKEND=${MOE_BACKEND}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
HF_OVERRIDES=${HF_OVERRIDES}
LLM_EXTRA_JSON=${LLM_EXTRA_JSON}
PROBE_RUNNER=${PROBE_RUNNER}
FALLBACK_QUANTIZATION=${FALLBACK_QUANTIZATION}
FALLBACK_LLM_EXTRA_JSON=${FALLBACK_LLM_EXTRA_JSON}
EOF

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG=${LOG}"

quant_args=()
if [[ -n "${QUANTIZATION}" ]]; then
  quant_args=(--quantization "${QUANTIZATION}")
fi

extra_mounts=()
if [[ -n "${OVERLAY_MODEL_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_MODEL_PY}:/opt/venv/lib/python3.12/site-packages/vllm/config/model.py:ro")
fi
if [[ -n "${OVERLAY_DEEPSEEK_V2_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_DEEPSEEK_V2_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py:ro")
fi
if [[ -n "${OVERLAY_DEEPGEMM_MOE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_DEEPGEMM_MOE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro")
fi
if [[ -n "${OVERLAY_MXFP8_ORACLE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_MXFP8_ORACLE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp8.py:ro")
fi
if [[ -n "${OVERLAY_MXFP4_ORACLE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_MXFP4_ORACLE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:ro")
fi
if [[ -n "${OVERLAY_OCP_MX_EMULATION_MOE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_OCP_MX_EMULATION_MOE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py:ro")
fi
if [[ -n "${OVERLAY_NVFP4_ORACLE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_NVFP4_ORACLE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:ro")
fi
if [[ -n "${OVERLAY_NVFP4_EMULATION_MOE_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_NVFP4_EMULATION_MOE_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py:ro")
fi
if [[ -n "${OVERLAY_MODELOPT_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_MODELOPT_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/modelopt.py:ro")
fi
if [[ -n "${OVERLAY_GGUF_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_GGUF_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/gguf.py:ro")
fi
if [[ -n "${OVERLAY_QUANT_INIT_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_QUANT_INIT_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py:ro")
fi
if [[ -n "${OVERLAY_LINEAR_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_LINEAR_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py:ro")
fi
if [[ -n "${OVERLAY_GGUF_DEQUANT_EXT_SO}" ]]; then
  extra_mounts+=(-v "${OVERLAY_GGUF_DEQUANT_EXT_SO}:/opt/venv/lib/python3.12/site-packages/vllm_gguf_dequant_ext.so:ro")
fi
if [[ -n "${OVERLAY_GGUF_LOADER_INIT_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_GGUF_LOADER_INIT_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/__init__.py:ro")
fi
if [[ -n "${OVERLAY_GGUF_LOADER_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_GGUF_LOADER_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/gguf_loader.py:ro")
fi
if [[ -n "${OVERLAY_WEIGHT_UTILS_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_WEIGHT_UTILS_PY}:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/weight_utils.py:ro")
fi
if [[ -n "${OVERLAY_GGUF_UTILS_PY}" ]]; then
  extra_mounts+=(-v "${OVERLAY_GGUF_UTILS_PY}:/opt/venv/lib/python3.12/site-packages/vllm/transformers_utils/gguf_utils.py:ro")
fi
if [[ -n "${OVERLAY_VLLM_C_SO}" ]]; then
  extra_mounts+=(-v "${OVERLAY_VLLM_C_SO}:/opt/venv/lib/python3.12/site-packages/vllm/_C.abi3.so:ro")
fi
if [[ -n "${OVERLAY_VLLM_C_STABLE_SO}" ]]; then
  extra_mounts+=(-v "${OVERLAY_VLLM_C_STABLE_SO}:/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so:ro")
fi
if [[ -n "${OVERLAY_QUTLASS_SO}" ]]; then
  extra_mounts+=(-v "${OVERLAY_QUTLASS_SO}:/opt/venv/lib/python3.12/site-packages/vllm/_qutlass_C.abi3.so:ro")
fi

docker run --rm --name "${CONTAINER_NAME}" \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host \
  --network=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/kld:/root/kld:rw \
  -v /root/models:/root/models:ro \
  -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
  -v /root/vllm/src:/root/vllm/src:ro \
  -v "${LOGITS_OVERLAY}:/overlay/vllm-release-kld:ro" \
  -v /cache:/cache:rw \
  -v /cache/kld_pydeps:/cache/kld_pydeps:ro \
  "${extra_mounts[@]}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_INNER}" \
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
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-0}" \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_DCP_A2A_MAX_TOKENS=64 \
  -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB \
  -e VLLM_PCIE_DMA_FP8=0 \
  -e B12X_PCIE_DMA_FP8=0 \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_W4A8_TINY_DECODE=1 \
  -e B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-0}" \
  -e B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-0}" \
  -e VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD="${VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD:-0}" \
  -e VLLM_NVFP4_EMULATION_WEIGHT_ONLY="${VLLM_NVFP4_EMULATION_WEIGHT_ONLY:-0}" \
  -e VLLM_GGUF_DEQUANT_AT_LOAD="${VLLM_GGUF_DEQUANT_AT_LOAD:-0}" \
  -e VLLM_GGUF_DEQUANT_IN_ITERATOR="${VLLM_GGUF_DEQUANT_IN_ITERATOR:-0}" \
  -e VLLM_GGUF_DEQUANT_EXT_SO=/opt/venv/lib/python3.12/site-packages/vllm_gguf_dequant_ext.so \
  -e VLLM_GGUF_DEQUANT_CPU_OFFLOAD_GB="${VLLM_GGUF_DEQUANT_CPU_OFFLOAD_GB:-0}" \
  -e VLLM_GGUF_DEQUANT_UVA_OFFLOAD_GB="${VLLM_GGUF_DEQUANT_UVA_OFFLOAD_GB:-0}" \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
SITE=\$(/opt/venv/bin/python - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith('site-packages')]
print(paths[0])
PY
)
for rel in \
  vllm/outputs.py \
  vllm/sampling_params.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/engine/__init__.py \
  vllm/v1/engine/logprobs.py \
  vllm/v1/engine/output_processor.py \
  vllm/v1/outputs.py \
  vllm/v1/worker/gpu/async_utils.py \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/v1/worker/gpu/sample/output.py \
  vllm/v1/worker/gpu/sample/prompt_logprob.py \
  vllm/v1/worker/gpu/sample/sampler.py \
  vllm/v1/worker/gpu/sample/states.py \
  vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
  vllm/v1/worker/gpu_input_batch.py \
  vllm/v1/worker/gpu_model_runner.py; do
  cp \"/overlay/vllm-release-kld/\${rel}\" \"\${SITE}/\${rel}\"
done
echo applied_logits_overlay=${LOGITS_OVERLAY}
/opt/venv/bin/python - <<'PY'
import inspect
from vllm import SamplingParams
print('return_prompt_logits_enabled', 'return_prompt_logits' in inspect.signature(SamplingParams).parameters, flush=True)
PY
if [[ '${PROBE_RUNNER}' == 'fallback' ]]; then
  /opt/venv/bin/python /root/kld/prefill_kld_fallback.py \
    --model '${MODEL}' \
    ${TOKENIZER:+--tokenizer '${TOKENIZER}'} \
    --reference-logits '${REF}' \
    --context-length 2048 \
    --stride 512 \
    --max-windows 1 \
    --tensor-parallel-size '${TP}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    --dtype '${DTYPE}' \
    --kv-cache-dtype fp8 \
    --load-format '${LOAD_FORMAT}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
    --max-num-seqs '${MAX_NUM_SEQS}' \
    --quantization '${FALLBACK_QUANTIZATION}' \
    --attention-backend '${ATTENTION_BACKEND}' \
    --hf-overrides '${HF_OVERRIDES}' \
    --llm-extra-json '${FALLBACK_LLM_EXTRA_JSON}' \
    --kld-chunk-rows 32
elif [[ '${PROBE_RUNNER}' == 'collect' ]]; then
  /opt/venv/bin/python /root/kld/collect_prefill_return_logits_ref.py \
    --model '${MODEL}' \
    --output-dir '${REF_OUTPUT}' \
    --context-length 2048 \
    --stride 512 \
    --max-windows 1 \
    --tensor-parallel-size '${TP}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    --dtype '${DTYPE}' \
    --kv-cache-dtype fp8 \
    --load-format '${LOAD_FORMAT}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
    --max-num-seqs '${MAX_NUM_SEQS}' \
    --quantization '${FALLBACK_QUANTIZATION}' \
    --attention-backend '${ATTENTION_BACKEND}' \
    --hf-overrides '${HF_OVERRIDES}' \
    --llm-extra-json '${FALLBACK_LLM_EXTRA_JSON}'
else
/opt/venv/bin/python /root/vllm/src/vllm/examples/offline_inference/score_mode_kld.py \
  --model '${MODEL}' \
  --reference-logits '${REF}' \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 \
  --stride 512 \
  --max-windows 1 \
  --tensor-parallel-size '${TP}' \
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
  --dtype '${DTYPE}' \
  --kv-cache-dtype fp8 \
  --load-format '${LOAD_FORMAT}' \
  --max-model-len '${MAX_MODEL_LEN}' \
  --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
  --max-num-seqs '${MAX_NUM_SEQS}' \
  --cpu-offload-gb '${CPU_OFFLOAD_GB}' \
  --attention-backend '${ATTENTION_BACKEND}' \
  --moe-backend '${MOE_BACKEND}' \
  --hf-overrides '${HF_OVERRIDES}' \
  --llm-extra-json '${LLM_EXTRA_JSON}' \
  --enforce-eager \
  --disable-custom-all-reduce \
  --trust-remote-code \
  ${quant_args[*]}
fi" >"${LOG}" 2>&1

python3 - "${LOG}" <<'PY'
import re
import sys

log = sys.argv[1]
text = open(log, errors="replace").read()
for pat in [
    r"return_prompt_logits_enabled .*",
    r"fallback_prefill_kld_done\s+(\{.*\})",
    r"Mean KLD:\s*([0-9.eE+-]+)",
    r"Total positions:\s*([0-9]+)",
    r"Time elapsed:\s*([0-9.]+) seconds",
]:
    m = re.search(pat, text)
    if m:
        print(m.group(0))
print("log=" + log)
PY
