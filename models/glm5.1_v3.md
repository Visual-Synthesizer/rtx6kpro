# GLM-5.1 v3 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-23 on the local 8-GPU RTX PRO 6000 Blackwell host.

This is the current GLM-5.1 v3 recipe for the W4A16 decode fast path with the
native ModelOpt/NVFP4 checkpoint layout. The recommended production profile uses
B12X sparse MLA, B12X MoE with A16 forced for decode, B12X PCIe oneshot
allreduce, GLM MTP support, FP8 KV cache, CUDA 13.2, FlashInfer git, and patched
local-inference NCCL 2.30.4.

Recommended default: DCP1 + MTP enabled for maximum decode throughput. Use DCP4
when larger KV/cache headroom matters more than single-request speed.

## Docker Compose

Default profile: DCP1, MTP enabled, forced A16 MoE, B12X PCIe oneshot
allreduce, FP8 KV cache. `VLLM_USE_V2_MODEL_RUNNER` is explicitly disabled for
now; the V2 runner is not part of this stable profile.

Set the model snapshot once:

```bash
export MODEL_DIR=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

Then start:

```bash
cat >compose.glm51-v3.yml <<'EOF'
services:
  glm51-v3:
    image: voipmonitor/vllm:glm51-w4a16micro-modelopt-20260523
    container_name: glm51-v3
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: /bin/bash
    command: -lc 'exec /opt/vllm/scripts/run-glm51-vllm'
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
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "0"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_B12X_MLA_SPEC_SERIAL_DECODE: "0"
      VLLM_MTP_RETURN_NORMALIZED_HIDDEN: "1"
      VLLM_SPEC_ACCEPT_THRESHOLD_ACC: "1.0"
      VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE: "1.0"
      VLLM_USE_V2_MODEL_RUNNER: "0"
      B12X_MOE_FORCE_A16: "1"
      VLLM_B12X_FORCE_MOE_A16: "1"
      PORT: "${PORT:-5264}"
      MODEL: /models/GLM-5.1-NVFP4-MTP-NVFP4
      SERVED_MODEL_NAME: GLM-5
      TP_SIZE: "8"
      DCP_SIZE: "${DCP_SIZE:-1}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.855}"
      MAX_MODEL_LEN: "202752"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "64"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}"
      KV_CACHE_DTYPE: fp8
      ATTENTION_BACKEND: B12X_MLA_SPARSE
      MOE_BACKEND: b12x
      GLM51_DISABLE_MTP: "${GLM51_DISABLE_MTP:-0}"
      SPEC_CONFIG: '{"model":"/models/GLM-5.1-NVFP4-MTP-NVFP4","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic","use_local_argmax_reduction":false}'
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
      - ${MODEL_DIR:?set MODEL_DIR}:/models/GLM-5.1-NVFP4-MTP-NVFP4:ro
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/vllm-glm51-v3/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${HOME}/.cache/vllm-glm51-v3/jit:/cache/jit
      - ${HOME}/.cache/vllm-glm51-v3/triton:/root/.cache/triton
      - ${HOME}/.cache/vllm-glm51-v3/torchinductor:/root/.cache/torchinductor
      - ${HOME}/.cache/vllm-glm51-v3/vllm:/root/.cache/vllm
EOF

docker compose -f compose.glm51-v3.yml up -d
```

Useful variants:

```bash
# DCP4 + MTP, larger KV/cache headroom but lower decode throughput
DCP_SIZE=4 GPU_MEMORY_UTILIZATION=0.855 docker compose -f compose.glm51-v3.yml up -d

# DCP1 without MTP
GLM51_DISABLE_MTP=1 GPU_MEMORY_UTILIZATION=0.825 docker compose -f compose.glm51-v3.yml up -d

# DCP4 without MTP, larger KV headroom
DCP_SIZE=4 GLM51_DISABLE_MTP=1 GPU_MEMORY_UTILIZATION=0.825 docker compose -f compose.glm51-v3.yml up -d
```

## Docker Run

Equivalent non-compose launch for the default DCP1 + MTP profile:

```bash
docker rm -f glm51-v3 >/dev/null 2>&1 || true

docker run -d --gpus all --ipc=host --network host --privileged \
  --name glm51-v3 \
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
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e PORT=5264 \
  -e MODEL=/models/GLM-5.1-NVFP4-MTP-NVFP4 \
  -e SERVED_MODEL_NAME=GLM-5 \
  -e TP_SIZE=8 \
  -e DCP_SIZE=1 \
  -e GPU_MEMORY_UTILIZATION=0.855 \
  -e MAX_MODEL_LEN=202752 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=64 \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=b12x \
  -e GLM51_DISABLE_MTP=0 \
  -e SPEC_CONFIG='{"model":"/models/GLM-5.1-NVFP4-MTP-NVFP4","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic","use_local_argmax_reduction":false}' \
  -v "${MODEL_DIR:?set MODEL_DIR}:/models/GLM-5.1-NVFP4-MTP-NVFP4:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm-glm51-v3/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${HOME}/.cache/vllm-glm51-v3/jit:/cache/jit" \
  -v "${HOME}/.cache/vllm-glm51-v3/triton:/root/.cache/triton" \
  -v "${HOME}/.cache/vllm-glm51-v3/torchinductor:/root/.cache/torchinductor" \
  -v "${HOME}/.cache/vllm-glm51-v3/vllm:/root/.cache/vllm" \
  voipmonitor/vllm:glm51-w4a16micro-modelopt-20260523 \
  -lc 'exec /opt/vllm/scripts/run-glm51-vllm'
```

## Image And Source State

```bash
voipmonitor/vllm:glm51-w4a16micro-modelopt-20260523
```

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `dev/b12x-integration` |
| vLLM commit | `76fe76a9075aabc289993747c93babc930a8d70c` |
| B12X repo | `https://github.com/voipmonitor/b12x.git` |
| B12X branch | `codex/w4a16-micro-modelopt-native-20260523` |
| B12X commit | `6eb12201a40e4bca0d6bd240a9cdc6fab8c3d3bd` |
| B12X base commit | `20a279fa321d53f20d37e85e765f4ba2ced60ed3` |
| Patchset | `w4a16-micro-decode-fastpath+modelopt-native-shared-staging` |
| FlashInfer | git `9035311e975a6aeb2d229f5162e999dfb7c9a733` |
| NCCL | `/opt/libnccl-local-inference.so.2.30.4` |
| CUDA | `13.2.1` |
| PyTorch | `2.12.0+cu132` |

The image labels record these revisions and can be checked with:

```bash
docker image inspect voipmonitor/vllm:glm51-w4a16micro-modelopt-20260523 \
  --format '{{json .Config.Labels}}'
```

## Benchmark Results

Benchmark command shape:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5317 \
  --model GLM-5 \
  --concurrency 1,16 \
  --contexts 0 \
  --duration 20 \
  --skip-prefill \
  --kv-budget <reported_budget> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor
```

Recommended rows use:

```text
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
B12X_MOE_FORCE_A16=1
VLLM_B12X_FORCE_MOE_A16=1
VLLM_USE_V2_MODEL_RUNNER=0
MAX_CUDAGRAPH_CAPTURE_SIZE=64
```

| Profile | GPU memory utilization | KV budget used by benchmark | cc1 ctx0 | cc16 ctx0 |
|---|---:|---:|---:|---:|
| DCP1, MTP off | 0.825 | 217,855 | 55.5 tok/s | 411.9 tok/s |
| DCP1, MTP on | 0.825 | n/a | startup failed | startup failed |
| DCP1, MTP on | 0.855 | 245,056 | 95.2 tok/s | 613.0 tok/s |
| DCP4, MTP off | 0.825 | 456,192 | 46.2 tok/s | 316.1 tok/s |
| DCP4, MTP on | 0.855 | 570,112 | 69.7 tok/s | 426.3 tok/s |

Result directory on the measured host:

```bash
/root/bench-results/glm51-luke-b12x-a16-matrix-20260523
```

DCP1 + MTP at `GPU_MEMORY_UTILIZATION=0.825` failed before serving:

```text
To serve at least one request with max seq len 202752, 11.75 GiB KV cache is
needed, larger than available 11.36 GiB.
```

Use `GPU_MEMORY_UTILIZATION=0.855` for MTP profiles. Use `0.825` for non-MTP
profiles unless additional KV headroom is specifically needed and graph capture
has been revalidated.

Do not set an empty `NCCL_GRAPH_FILE=` with this NCCL stack. On this image it
can fail during `ncclCommInitRank` with `NCCL error: unhandled system error`
before model load or CUDA graph capture starts. Leaving the variable unset keeps
the DCP4 + MTP profile working through full decode graph capture.

### Known Bad: V2 Model Runner

Do not enable the V2 model runner for this profile yet:

```text
VLLM_USE_V2_MODEL_RUNNER=1
```

Reproduced on 2026-05-24 with `DCP4 + MTP on + A16 on + b12x allreduce`.
Startup reached KV cache initialization (`980,224` tokens) and then crashed
during `Capturing CUDA graphs (FULL)` around `31/33`.

Root stack:

```text
b12x/attention/mla/api.py:_final_lse_from_split_workspace
torch.logsumexp(chunk_lse, dim=-1, out=final_lse)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The later NCCL watchdog failure is secondary after the illegal memory access.

### Historical: CPP Allreduce, A16 On

Older v3 rows used the `cpp` PCIe allreduce backend. They are kept only as a
baseline; the recommended profile above uses `b12x`.

```text
VLLM_PCIE_ALLREDUCE_BACKEND=cpp
B12X_MOE_FORCE_A16=1
VLLM_B12X_FORCE_MOE_A16=1
```

| Profile | GPU memory utilization | KV budget used by benchmark | cc1 ctx0 | cc16 ctx0 |
|---|---:|---:|---:|---:|
| DCP1, MTP off | 0.865 | 71,088 | 55.4 tok/s | 416.7 tok/s |
| DCP1, MTP on | 0.855 | 61,296 | 92.0 tok/s | 606.5 tok/s |
| DCP4, MTP off | 0.865 | 284,416 | 46.5 tok/s | 316.7 tok/s |
| DCP4, MTP on | 0.855 | 245,184 | 71.2 tok/s | 418.6 tok/s |

Result directory on the measured host:

```bash
/root/bench-results/glm51-w4a16micro-modelopt-cc-20260523
```

### Historical: A16 Off Comparison

Measured with the same image and launch recipe, except:

```text
B12X_MOE_FORCE_A16=0
VLLM_B12X_FORCE_MOE_A16=0
```

| Profile | GPU memory utilization | KV budget used by benchmark | cc1 ctx0 | cc16 ctx0 |
|---|---:|---:|---:|---:|
| DCP1, MTP off | 0.865 | 92,112 | 57.0 tok/s | 409.5 tok/s |
| DCP1, MTP on | 0.855 | 85,248 | 94.1 tok/s | 541.0 tok/s |
| DCP4, MTP off | 0.865 | 380,160 | 46.9 tok/s | 311.9 tok/s |
| DCP4, MTP on | 0.855 | 1,324,288 | 69.7 tok/s | 382.6 tok/s |

DCP4 + MTP with A16 off was retested on 2026-05-24 after removing the empty
`NCCL_GRAPH_FILE=` environment variable. The corrected launch passed both mixed
prefill/decode graph capture and full decode graph capture before benchmarking.

Result directory on the measured host:

```bash
/root/bench-results/glm51-w4a16micro-modelopt-cc-a16off-20260523
/root/bench-results/glm51-v3-nographfile-dcp4-mtp-a16off-20260524
```
