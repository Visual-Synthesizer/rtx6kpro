# GLM-5.1 v4 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-25 on the local 8-GPU RTX PRO 6000 Blackwell host.

This page documents the current GLM-5.1 v4 mixed-precision profile. The key
change versus v3 is the native ModelOpt W4A16 decode path: the checkpoint-native
ModelOpt/NVFP4 weight layout stays resident, and B12X W4A16 decode consumes that
layout directly instead of keeping a second large prepacked W4A16 weight copy on
GPU.

Recommended default for this image: DCP4 + MTP enabled + mixed FP8 L42-62 main
checkpoint + W4A16 decode. This is the profile currently used for interactive
testing.

## Docker Compose

Set the two model snapshots:

```bash
export MODEL_DIR=/root/.cache/huggingface/hub/models--festr2--glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517/snapshots/e37f1787435d2b2c111a5f5eac924a556a06e257
export MTP_MODEL_DIR=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

Then start:

```bash
cat >compose.glm51-v4.yml <<'EOF'
services:
  glm51-v4:
    image: voipmonitor/vllm:glm51-v4-nativew4a16-modelopt-20260525
    container_name: glm51-v4
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: /bin/bash
    command: -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec /opt/vllm/scripts/run-glm51-vllm'
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      OMP_NUM_THREADS: "16"
      SAFETENSORS_FAST_GPU: "1"
      CUTE_DSL_ARCH: sm_120a
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_NCCL_SO_PATH: /opt/libnccl-local-inference.so.2.30.4
      LD_PRELOAD: /opt/libnccl-local-inference.so.2.30.4
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_CPP_AR_1STAGE_NCCL_CUTOFF: 56KB
      VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS: "0"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "0"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_B12X_MLA_SPEC_SERIAL_DECODE: "0"
      VLLM_MTP_RETURN_NORMALIZED_HIDDEN: "1"
      VLLM_SPEC_ACCEPT_THRESHOLD_ACC: "1.0"
      VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE: "1.0"
      VLLM_DISABLE_SHARED_EXPERTS_STREAM: "0"
      B12X_MOE_FORCE_A16: "0"
      VLLM_B12X_FORCE_MOE_A16: "1"
      PORT: "${PORT:-5317}"
      MODEL: /models/GLM-5.1-MIXED-FP8
      SERVED_MODEL_NAME: GLM-5
      TP_SIZE: "8"
      DCP_SIZE: "${DCP_SIZE:-4}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.85}"
      MAX_MODEL_LEN: "202752"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "64"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "64"
      KV_CACHE_DTYPE: fp8
      ATTENTION_BACKEND: B12X_MLA_SPARSE
      MOE_BACKEND: b12x
      GLM51_DISABLE_MTP: "${GLM51_DISABLE_MTP:-0}"
      SPEC_CONFIG: '{"model":"/models/GLM-5.1-NVFP4-MTP","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic","use_local_argmax_reduction":false}'
      HF_HOME: /root/.cache/huggingface
      HUGGINGFACE_HUB_CACHE: /root/.cache/huggingface/hub
      XDG_CACHE_HOME: /cache/jit
      CUDA_CACHE_PATH: /cache/jit
      VLLM_CACHE_DIR: /cache/jit/vllm
      TVM_FFI_CACHE_DIR: /cache/jit/tvm-ffi
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      VLLM_CACHE_ROOT: /root/.cache/vllm
      TRITON_CACHE_DIR: /root/.cache/triton
      TORCHINDUCTOR_CACHE_DIR: /root/.cache/torchinductor
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      CUTE_DSL_CACHE_DIR: /root/.cache/cutlass_dsl
    volumes:
      - ${MODEL_DIR:?set MODEL_DIR}:/models/GLM-5.1-MIXED-FP8:ro
      - ${MTP_MODEL_DIR:?set MTP_MODEL_DIR}:/models/GLM-5.1-NVFP4-MTP:ro
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/vllm-glm51-v4/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${HOME}/.cache/vllm-glm51-v4/jit:/cache/jit
      - ${HOME}/.cache/vllm-glm51-v4/triton:/root/.cache/triton
      - ${HOME}/.cache/vllm-glm51-v4/torchinductor:/root/.cache/torchinductor
      - ${HOME}/.cache/vllm-glm51-v4/vllm:/root/.cache/vllm
EOF

docker compose -f compose.glm51-v4.yml up -d
```

Useful variants:

```bash
# Same profile without MTP
GLM51_DISABLE_MTP=1 docker compose -f compose.glm51-v4.yml up -d

# Lower KV memory if startup is tight on a different host
GPU_MEMORY_UTILIZATION=0.825 docker compose -f compose.glm51-v4.yml up -d
```

Do not pass an empty `NCCL_GRAPH_FILE=` environment variable. With the newer
local-inference NCCL build, leave `NCCL_GRAPH_FILE` unset.

## Docker Run

Equivalent non-compose launch for the measured DCP4 + MTP profile:

```bash
docker rm -f glm51-v4 >/dev/null 2>&1 || true

docker run -d --gpus all --ipc=host --network host --privileged \
  --name glm51-v4 \
  --entrypoint /bin/bash \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e PORT=5317 \
  -e MODEL=/models/GLM-5.1-MIXED-FP8 \
  -e SERVED_MODEL_NAME=GLM-5 \
  -e TP_SIZE=8 \
  -e DCP_SIZE=4 \
  -e GPU_MEMORY_UTILIZATION=0.85 \
  -e MAX_MODEL_LEN=202752 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=64 \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=b12x \
  -e GLM51_DISABLE_MTP=0 \
  -e SPEC_CONFIG='{"model":"/models/GLM-5.1-NVFP4-MTP","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic","use_local_argmax_reduction":false}' \
  -v "${MODEL_DIR:?set MODEL_DIR}:/models/GLM-5.1-MIXED-FP8:ro" \
  -v "${MTP_MODEL_DIR:?set MTP_MODEL_DIR}:/models/GLM-5.1-NVFP4-MTP:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm-glm51-v4/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${HOME}/.cache/vllm-glm51-v4/jit:/cache/jit" \
  -v "${HOME}/.cache/vllm-glm51-v4/triton:/root/.cache/triton" \
  -v "${HOME}/.cache/vllm-glm51-v4/torchinductor:/root/.cache/torchinductor" \
  -v "${HOME}/.cache/vllm-glm51-v4/vllm:/root/.cache/vllm" \
  voipmonitor/vllm:glm51-v4-nativew4a16-modelopt-20260525 \
  -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec /opt/vllm/scripts/run-glm51-vllm'
```

## Image And Source State

Primary Docker tag:

```bash
voipmonitor/vllm:glm51-v4-nativew4a16-modelopt-20260525
```

Equivalent local build tags used during validation:

```bash
voipmonitor/vllm:glm51-nativew4a16-modelopt-20260524
voipmonitor/vllm:glm51-nativew4a16-vf7d79ac-b12x0a32ff3-20260524
```

Validated local image ID:

```text
sha256:aa0e526a827ed398c93ce57e70bf9c1618b236e9a276dedd2833a68a5c531fb2
```

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | `codex/current-a16-kld-noprepack-20260524` |
| vLLM commit | `f7d79acaa21c100211ce64b424d927cbe4e78136` |
| vLLM base | `https://github.com/local-inference-lab/vllm/tree/dev/b12x-integration` at `28be48fa944d8c58aeffd1ab1b56e27828c6b268` |
| B12X repo | `https://github.com/voipmonitor/b12x.git` |
| B12X branch | `codex/current-a16-kld-layout-20260524` |
| B12X commit | `0a32ff337682257b90b81aa1fbf1f6a79f63ba55` |
| B12X base | `https://github.com/lukealonso/b12x/tree/master` at `c1331e2179e7b14e32c01021b3732fde48e5e862` |
| CUDA | `13.2.1` |
| NCCL | `/opt/libnccl-local-inference.so.2.30.4` |

The running container's modified runtime files were hash-checked against these
git branches before KLD validation.

## Validation

Current serving profile:

| Setting | Value |
|---|---|
| Main model | `festr2/glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517` |
| Main snapshot | `e37f1787435d2b2c111a5f5eac924a556a06e257` |
| Draft model | `lukealonso/GLM-5.1-NVFP4-MTP` |
| Draft snapshot | `78b7fe365f3905b4e0261a85182fefdbd5137989` |
| TP / DCP | `8 / 4` |
| MTP | enabled, `num_speculative_tokens=3`, probabilistic draft |
| MoE | `MOE_BACKEND=b12x` |
| Decode A16 | `VLLM_B12X_FORCE_MOE_A16=1` |
| B12X global A16 | `B12X_MOE_FORCE_A16=0` |
| KV cache | `fp8` |
| GPU memory utilization | `0.85` |

Startup memory/KV from the measured run:

| Metric | Value |
|---|---:|
| Model loading memory | `72.86 GiB` |
| B12X joint arena | `2.21 GiB` |
| Available KV cache memory | `5.26 GiB` |
| GPU KV cache size | `362,496 tokens` |
| Max concurrency at `202,752` tokens | `1.79x` |

KLD smoke result from this exact image:

| Model | Reference | Windows | Context | Mean KLD |
|---|---|---:|---:|---:|
| mixed FP8PBWO L42-62 + W4A16 decode | BF16 vLLM WikiText | 1 | 2048 | `0.063471` |

KLD log:

```bash
/root/kld/mixedfp8_nativew4a16_modelopt_image_kld_vs_bf16_vllm_ctx2048_w1_20260525_002646.log
```

Historical quality comparison, not from this exact image:

| Model | Reference | Windows | Mean KLD |
|---|---|---:|---:|
| NVFP4 W4A16 | FP8 vLLM | 42 | `0.068724` |
| mixed FP8PBWO L42-62 W4A16 | FP8 vLLM | 42 | `0.049504` |

The historical W42 comparison says the mixed FP8PBWO L42-62 variant was better
by `0.019220` absolute KLD, roughly `28%` lower KLD than NVFP4 W4A16. A pure
NVFP4 A16 KLD run has not yet been remeasured on this exact v4 image ID.

## Operational Notes

- `VLLM_B12X_FORCE_MOE_A16=1` is the intended v4 A16 switch.
- `B12X_MOE_FORCE_A16=0` is intentional for this profile.
- The native ModelOpt path avoids the previous large resident W4A16 prepacked
  copy and keeps the serving profile within usable KV headroom.
- `VLLM_USE_V2_MODEL_RUNNER` is not part of this profile.
- Leave `NCCL_GRAPH_FILE` unset. Do not set it to an empty value.
- The MTP path is expected to be lossless at the target distribution level; the
  practical metric to watch is acceptance rate and end-to-end throughput.
