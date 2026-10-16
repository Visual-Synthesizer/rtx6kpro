#!/usr/bin/env bash
# GLM-5.3-Flash (local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, qad2-kda-w8a16-csf @ 96f27f8c) on two RTX PRO 6000:
# the exact vLLM command and environment that PRESET=glm53-tp2 runs in release-10152026, without the preset launcher.
# GPU-only prefix cache (the preset's default adds an LMCache RAM tier; that needs its own service and is left out here).
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/local-inference-lab/vllm:release-10152026}
NAME=${NAME:-glm53-plain}
GPUS=${GPUS:-0,1}
PORT=${PORT:-8000}
HF_CACHE=${HF_CACHE:-model-cache}      # docker volume or host directory with the Hugging Face cache
RUNTIME_CACHE=${RUNTIME_CACHE:-${NAME}-runtime}   # JIT/compile caches (optional, speeds up restarts)
MODEL=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD}
REVISION=${REVISION:-96f27f8c4e443be5bc486f8a978e3b9e0d3f7125}   # branch qad2-kda-w8a16-csf; a mirror of this revision needs its own commit here

docker run -d --name "$NAME" --init --gpus "\"device=$GPUS\"" \
  --network host --ipc host --shm-size 32g --stop-timeout 30 \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v "$HF_CACHE":/root/.cache/huggingface -v "$RUNTIME_CACHE":/cache -e XDG_CACHE_HOME=/cache \
  -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
  -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 \
  -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
  -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 \
  -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 \
  -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  -e B12X_MHC_PDL=1 \
  -e B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
  -e B12X_PCIE_ONESHOT_PDL=1 \
  -e B12X_PCIE_ONESHOT_THREADS=512 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -e B12X_W4A16_FP32_TOPK_WEIGHTS=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e GLOO_SOCKET_IFNAME=lo \
  -e HF_HOME=/root/.cache/huggingface \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e NCCL_BUFFSIZE=1048576 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_MIN_NCHANNELS=2 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_TUNER_PLUGIN=none \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e VLLM_B12X_MLA_CKV_GATHER=1 \
  -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_GLM53_DFLASH_ATTN=1 \
  -e VLLM_GLM53_EMBED_HOST=1 \
  -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 \
  -e VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill \
  -e VLLM_GLM53_L2_PREFETCH=1 \
  -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0 \
  -e VLLM_GLM53_MTP_DRAFT_HEAD=bf16 \
  -e VLLM_GLM53_ONLINE_DENSE_MXFP8=0 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048 \
  -e VLLM_LM_HEAD_A16=1 \
  -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e VLLM_MXFP8_LM_HEAD=0 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_DMA_MIN_BYTES=6MB \
  -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off \
  -e VLLM_PLUGINS='' \
  -e VLLM_SHARE_PYNCCL_COMMS=1 \
  -e VLLM_USE_FASTOKENS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
    --attention-backend B12X \
    --block-size 256 \
    --chat-template /opt/lil/runtime/templates/glm53-flash.jinja \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --cp-kv-cache-interleave-size 4 \
    --cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32 \
    --dcp-kv-cache-interleave-size 4 \
    --decode-context-parallel-size 2 \
    --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}' \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.985 \
    --host 0.0.0.0 \
    --kv-cache-dtype fp8 \
    --kv-cache-memory-bytes 7650410496 \
    --linear-backend b12x \
    --load-format instanttensor \
    --mamba-cache-mode align \
    --max-cudagraph-capture-size 32 \
    --max-model-len -1 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --max-parallel-prefills 1 \
    --moe-backend b12x \
    --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
    --pipeline-parallel-size 1 \
    --port "$PORT" \
    --prefill-compute-share 0.4 \
    --prefill-schedule-interval 1 \
    --quantization modelopt_mixed \
    --reasoning-parser glm45 \
    --recurrent-checkpoint-policy request_boundaries \
    --revision "$REVISION" \
    --served-model-name GLM-5.3-Flash \
    --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3,"revision":"'"$REVISION"'"}' \
    --tensor-parallel-size 2 \
    --tool-call-parser glm47
