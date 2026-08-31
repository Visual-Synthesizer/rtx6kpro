#!/usr/bin/env bash
set -euo pipefail

# Known-good GLM-5.2 NVFP4 DCP4 + MTP3 runtime from the 2026-06-19 PR #28
# sharded-draft evaluation. This is intentionally pinned to the historical
# image because later Dark Devotion/B12X rebases changed the DCP top-k path.
#
# Measured summary for this exact stack:
#   0k C1 92.4 tok/s, 0k C4 288.6 tok/s
#   64k C1 64.7 tok/s, 64k C4 207.9 tok/s
#   weighted sustained MTP acceptance 59.0%

IMAGE="${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-dcp4-mtp3-pr28-sharddraft-good-cu132-20260619}"
IMAGE_DIGEST="sha256:e886de591ce18dd079fa3de673d42b0550c62b37ae0cdd6f21454cf018f58401"
NAME="${NAME:-glm52-dcp4-mtp3-pr28-good}"
PORT="${PORT:-5530}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}"

TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-4}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-24}"

GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES="${HF_OVERRIDES:-$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "${GLM52_INDEX_TOPK_PATTERN}")}"
SPEC_CONFIG="${SPEC_CONFIG:-$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}' "${MODEL}" "${NUM_SPECULATIVE_TOKENS}")}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --init \
  --name "${NAME}" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/bench-results:/root/bench-results \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e VLLM_DCP_GLOBAL_TOPK=0 \
  -e VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY=0 \
  -e VLLM_DCP_TOPK_FORCE_DEEPGEMM=0 \
  -e MODEL="${MODEL}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e PORT="${PORT}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e DCP_SIZE="${DCP_SIZE}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  -e HF_OVERRIDES="${HF_OVERRIDES}" \
  -e SPEC_CONFIG="${SPEC_CONFIG}" \
  --entrypoint bash \
  "${IMAGE}" -lc '
set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
cd /
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend ag_rs \
  --dcp-kv-cache-interleave-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --kv-cache-dtype fp8 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --hf-overrides "${HF_OVERRIDES}" \
  --quantization modelopt_fp4 \
  --speculative-config "${SPEC_CONFIG}"
'

cat <<EOF
Started ${NAME} on port ${PORT}.
Image: ${IMAGE}
Pinned digest: ${IMAGE_DIGEST}
Logs: docker logs -f ${NAME}
EOF
