#!/usr/bin/env bash
# Serve a Kimi K3 checkpoint with the qualified hidden-state capture runtime
# used by the 1,024-context distribution-fidelity dataset.

set -euo pipefail

capture_image=${CAPTURE_IMAGE:-voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1}
capture_container=${CAPTURE_CONTAINER:-kimi-k3-fidelity-capture}
capture_port=${CAPTURE_PORT:-8001}
capture_source=${CAPTURE_VLLM_SOURCE:?Set CAPTURE_VLLM_SOURCE to a vLLM checkout at commit e77ee0612b9b7d117439920ef81bdbb162d09cd3}
capture_output=${CAPTURE_OUTPUT:?Set CAPTURE_OUTPUT to an empty or resumable capture directory}
capture_hf_cache=${CAPTURE_HF_CACHE:-/root/.cache/huggingface}
capture_jit_cache=${CAPTURE_JIT_CACHE:?Set CAPTURE_JIT_CACHE to a writable JIT-cache directory}
capture_model_relative=${CAPTURE_MODEL_RELATIVE:-hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}
capture_model_host=$capture_hf_cache/$capture_model_relative
capture_model_container=/root/.cache/huggingface/$capture_model_relative
capture_commit=e77ee0612b9b7d117439920ef81bdbb162d09cd3
capture_image_id=sha256:4be1d706e29cc5d53fc2891378ba185538d5a35e69793062a8f973f1886217f0

if [[ $(git -C "$capture_source" rev-parse HEAD) != "$capture_commit" ]]; then
  echo "CAPTURE_VLLM_SOURCE must be checked out at $capture_commit" >&2
  exit 1
fi
if [[ $(docker image inspect "$capture_image" --format '{{.Id}}') != "$capture_image_id" ]]; then
  echo "CAPTURE_IMAGE does not resolve to the qualified image ID $capture_image_id" >&2
  exit 1
fi
if docker container inspect "$capture_container" >/dev/null 2>&1; then
  echo "Container name already exists: $capture_container" >&2
  exit 1
fi
if [[ ! -f "$capture_model_host/model.safetensors.index.json" ]]; then
  echo "Kimi K3 checkpoint index is missing: $capture_model_host" >&2
  exit 1
fi

mkdir -p "$capture_output/capture-hidden"

docker run -d \
  --name "$capture_container" \
  --entrypoint /bin/bash \
  --gpus all --network host --ipc host --shm-size 32g \
  -v "$capture_jit_cache:/cache/jit" \
  -v "$capture_source/vllm/models/kimi_k3/nvidia/model.py:/opt/kimi-k3/vllm/vllm/models/kimi_k3/nvidia/model.py:ro" \
  -v "$capture_source/vllm/v1/worker/gpu_model_runner.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$capture_source/vllm/v1/worker/gpu/distribution_capture.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu/distribution_capture.py:ro" \
  -v "$capture_source/vllm/v1/worker/gpu/model_runner.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu/model_runner.py:ro" \
  -v "$capture_source/vllm/v1/worker/gpu/sample/prompt_logprob.py:/opt/kimi-k3/vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py:ro" \
  -v "$capture_output:/mnt/kld" \
  -v "$capture_hf_cache:/root/.cache/huggingface:ro" \
  -e OMP_NUM_THREADS=16 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e CUDA_MODULE_DATA_LOADING=LAZY \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK=0 \
  -e VLLM_PCIE_ONESHOT_SINGLE_CHANNEL=1 \
  -e INSTANTTENSOR_COPY=0 \
  -e VLLM_KLD_HIDDEN_CAPTURE_DIR=/mnt/kld/capture-hidden \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_KIMI_USE_B12X_PROJECTION_GATHER=1 \
  -e VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=2048 \
  -e INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.6 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_KIMI_SHARD_QKV_A=1 \
  -e VLLM_KIMI_FUSED_TOPK16=1 \
  -e INSTANTTENSOR_BUFFER_SIZE=536870912 \
  -e INSTANTTENSOR_BACKEND=AIO \
  -e VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PROMPT_LOGPROBS_CHUNK_SIZE=256 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_TUNER_PLUGIN=none \
  "$capture_image" \
  -lc "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; cd /opt/kimi-k3/vllm; exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve '$capture_model_container' --served-model-name Kimi-K3 --trust-remote-code --language-model-only --host 0.0.0.0 --port '$capture_port' --tensor-parallel-size 16 --decode-context-parallel-size 1 --load-format instanttensor --moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA --kda-prefill-backend triton --dtype bfloat16 --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 300000000 --max-model-len 4096 --max-num-batched-tokens 256 --max-num-seqs 1 --gpu-memory-utilization 0.982 --enable-chunked-prefill --no-enable-prefix-caching --compilation-config '{\"mode\":0,\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1]}' --generation-config vllm --seed 1 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice > /mnt/kld/server.log 2>&1"

echo "Server container: $capture_container"
echo "Server log: $capture_output/server.log"
echo "Health endpoint after model load: http://127.0.0.1:$capture_port/health"
