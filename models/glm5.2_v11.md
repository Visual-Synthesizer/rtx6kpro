# GLM-5.2 v11 NVFP4 / FP8 / MXFP8 on Dark Devotion

Measured on 2026-06-18 on the local 16x RTX PRO 6000 Blackwell host.

This page records the GLM-5.2 v11 Docker/image state, BF16 teacher-forced
reference logits, KLD comparison for Luke NVFP4, official FP8, and a new
full-layer MXFP8 checkpoint built from Luke's NVFP4 checkpoint with BF16 MoE
weights requantized to MXFP8.

Status:

- DCP1 and DCP4 no-MTP start and decode coherently with B12X MLA sparse,
  B12X NvFp4 MoE, V2 model runner, CUDA graphs, and forced A16 MoE decode.
- DCP4 + MTP5 starts after the vLLM per-layer DCP LSE fix and passed a short
  coherence smoke test. A full DCP4+MTP speed sweep was not run yet.
- TP16 no-padding fix is included, but this page's served benchmark profile is
  TP8.

## Image

```text
voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618
```

Local image ID:

```text
sha256:4b54f000c8504710c78ea790f0b92d506bf8ffefc32cf3c1661c27ff82c251c8
```

Pinned digest:

```text
voipmonitor/vllm@sha256:fa12ec0b9152ef25a19959ad315468df879da40f34c5c275d28c9fb82880ef18
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA | `13.2` runtime stack |
| cuBLAS runtime | `13.4.1.2` symlinked into CUDA libdir |
| NCCL local library | `2.30.4`, compiled with CUDA 13.2 |
| PyTorch | `2.12.0+cu132` |
| FlashInfer | `0.6.12+cu132` prebuilt wheels in this image |
| DeepGEMM | `2.5.0` from PR324 source layer |
| vLLM branch | `codex/glm52-v11-tp16-padding-20260618` |
| vLLM commit | `a86f74e5054651e76c3d7db02df945908d8b9e24` |
| B12X branch | `lukealonso/b12x master` |
| B12X commit | `5b2e018d1c5228436a3ca23f67b17dab55c9cf65` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main` |
| Docker build fix | `9bfa9b4 fix: keep vLLM build dependency checks scoped` |

The image verifies:

```text
vllm 0.11.2.dev279+glm52.v11.darkdevotion.a86f74e.b12x5b2e018.cu132.20260618
b12x 0.23.0
rust_tool_parser /opt/venv/lib/python3.12/site-packages/vllm/_rust_tool_parser.abi3.so
vllm.config.virtual_tp._ATTENTION_HEAD_LOCAL_ALIGNMENT == 1
```

## Required vLLM changes

The clean vLLM branch contains two serving fixes on top of
`dev/dark-devotion@39ae3ed`:

| Commit | Purpose |
|---|---|
| `62cb7b3d4e9afcec2da1b10195e3dec4d75a3be9` | Avoid unnecessary B12X virtual-TP attention-head padding at TP16 by changing local head alignment from 8 to 1. |
| `a86f74e5054651e76c3d7db02df945908d8b9e24` | Check DCP LSE support per attention layer. This fixes DCP4+MTP, where backbone layers use DCP4 but the local MTP layer has `dcp_world_size=1`. |

KLD capture used additional local-only instrumentation for
`return_prompt_logits` and `return_sample_logits`. That instrumentation is not
part of the serving image and should not be included in the serving PR.

## Docker Build

Build recipe source:

```text
https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/Dockerfile.vllm-b12x-cu132
https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-vllm-b12x-cu132.sh
```

Build command used:

```bash
cd /root/vllm/blackwell-llm-docker

IMAGE=voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
BUILD_BASE_IMAGE=0 PUSH_BASE_IMAGE=0 MAX_JOBS=64 VLLM_MAX_JOBS=64 \
NVCC_THREADS=1 VLLM_NVCC_THREADS=1 \
VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
VLLM_REF=codex/glm52-v11-tp16-padding-20260618 \
VLLM_COMMIT=a86f74e5054651e76c3d7db02df945908d8b9e24 \
LAUNCHER_REPO=https://github.com/local-inference-lab/vllm.git \
LAUNCHER_REF=codex/glm52-v11-tp16-padding-20260618 \
LAUNCHER_COMMIT=a86f74e5054651e76c3d7db02df945908d8b9e24 \
B12X_REPO=https://github.com/lukealonso/b12x.git \
B12X_REF=master \
B12X_COMMIT=5b2e018d1c5228436a3ca23f67b17dab55c9cf65 \
FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git \
FLASHINFER_REF=refs/pull/3395/head \
FLASHINFER_COMMIT=b619f0c6508cc56df2d4717e90a6d46a2bac710e \
DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git \
DEEPGEMM_REF=refs/pull/324/head \
DEEPGEMM_COMMIT=9ca30487a6d1a484757f2d87f532c5f6707b9f25 \
CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git \
CUTLASS_REF=main \
CUTLASS_COMMIT=cf064d2e6bad2886238ac565b3b49007764f4939 \
VLLM_BUILD_VERSION=0.11.2.dev279+glm52.v11.darkdevotion.a86f74e.b12x5b2e018.cu132.20260618 \
./build-vllm-b12x-cu132.sh
```

Important build note: `FLASHINFER_REF` is passed for reproducibility of the
recipe, but this Dockerfile uses prebuilt `.tmp-flashinfer-wheels` when present.
This image reports `flashinfer 0.6.12+cu132`.

## Models

BF16 teacher source:

```text
/root/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/4d67f66cc64d3219133b767c253b2ad1425c6c88
```

Luke NVFP4 serving checkpoint:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

Official FP8 checkpoint:

```text
zai-org/GLM-5.2-FP8
```

New MXFP8 checkpoint:

```text
/root/kld/checkpoints/GLM-5.2-MXFP8-FULL-L3-77-FROM-LUKE-NVFP4-20260618
```

MXFP8 checkpoint size is about `737G`. Layers `3..77` use MXFP8 expert weights;
MTP layer `78` remains from Luke NVFP4.

## MXFP8 Checkpoint Build

Builder:

```text
/root/kld/tools/build_glm51_mxfp8_direct_from_bf16.py
```

Command:

```bash
PYTHONUNBUFFERED=1 /root/venvs/glm51-mxfp8-modelopt/bin/python \
  /root/kld/tools/build_glm51_mxfp8_direct_from_bf16.py \
  --base /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 \
  --bf16-source /root/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/4d67f66cc64d3219133b767c253b2ad1425c6c88 \
  --output /root/kld/checkpoints/GLM-5.2-MXFP8-FULL-L3-77-FROM-LUKE-NVFP4-20260618 \
  --layers 3-77 \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --force
```

Build log:

```text
/root/kld/glm52_mxfp8_build_20260618_logs/build.log
```

## KLD

BF16 reference logits:

```text
Prefill: /root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618/logits_0.safetensors
Decode:  /root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors
```

Decode teacher token ids:

```text
[5562, 13, 1096, 32619, 9929, 374, 1172, 1483, 979, 279, 29121, 1178, 6500, 374, 34813, 13, 576]
```

| Model | Runtime backend | Prefill KLD mean | Decode KL model\|\|bf16 | Decode KL bf16\|\|model | Decode JS mean | Decode token match |
|---|---|---:|---:|---:|---:|---:|
| Luke NVFP4 | B12X A16 | 0.067532 | 0.000009892 | 0.000013952 | 0.000002845 | 17/17 |
| Official FP8 | auto/FP8 | 0.079041 | 0.000085952 | 0.000045639 | 0.000011247 | 17/17 |
| New MXFP8 | `modelopt_mixed`, MARLIN MxFp8 MoE | 0.018720 | 0.000012423 | 0.000015746 | 0.000003378 | 17/17 |

KLD output directories:

```text
/root/kld/glm52_kld_nvfp4_20260618_0334/nvfp4_b12x_a16
/root/kld/glm52_kld_fp8_20260618_0346/fp8_auto
/root/kld/glm52_kld_mxfp8_20260618_0406/mxfp8_auto
```

## Runtime Profile

Local helper:

```text
/root/run-glm52-v11
```

Important defaults:

| Setting | Value |
|---|---|
| GPUs | `0,1,2,3,4,5,6,7` |
| Tensor parallel | `8` |
| DCP | set by `DCP_SIZE=1|4` |
| MTP | set by `MTP=0|1` |
| MTP tokens | default `3`, DCP4 smoke used `5` |
| MTP draft sampling | `probabilistic` |
| Max model len | `256000` |
| Max num seqs | `64` no-MTP; DCP4+MTP smoke used `16` |
| Max batched tokens | `8192` |
| CUDA graph cap | `64` no-MTP; DCP4+MTP smoke used `96` |
| KV cache dtype | `fp8` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| Quantization | `modelopt_fp4` |
| GPU memory utilization | `0.96` no-MTP; DCP4+MTP smoke used `0.94` |
| A16 force | `B12X_MOE_FORCE_A16=1` |
| A16 decode kernel flag | `B12X_W4A16_TC_DECODE=1` |
| GLM 5.2 IndexCache | `use_index_cache=true` with explicit `FFFSSS...` top-k pattern |

Important: unset empty NCCL graph variables before `vllm serve`:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

Example launches:

```bash
# DCP1, MTP off, force A16.
DCP_SIZE=1 MTP=0 GPU_MEMORY_UTILIZATION=0.96 PORT=5329 /root/run-glm52-v11

# DCP4, MTP off, force A16.
DCP_SIZE=4 MTP=0 GPU_MEMORY_UTILIZATION=0.96 PORT=5329 /root/run-glm52-v11

# DCP4, MTP5, force A16. Startup/smoke verified, full speed sweep pending.
DCP_SIZE=4 MTP=1 NUM_SPECULATIVE_TOKENS=5 \
  GPU_MEMORY_UTILIZATION=0.94 MAX_NUM_SEQS=16 MAX_CUDAGRAPH_CAPTURE_SIZE=96 \
  PORT=5329 /root/run-glm52-v11
```

Healthy startup logs include:

```text
Using V2 Model Runner
Using AttentionBackendEnum.B12X_MLA_SPARSE backend
Using 'B12X' NvFp4 MoE backend
B12X MoE force-A16 enabled: using quant_mode=w4a16
Using b12x PCIe oneshot allreduce backend
```

## Docker Compose

This compose is equivalent to `/root/run-glm52-v11` for the main DCP4 no-MTP
profile. Set `MTP=1` and `NUM_SPECULATIVE_TOKENS=5` for the DCP4+MTP smoke
profile.

```yaml
services:
  glm52-v11:
    image: ${IMAGE:-voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618}
    container_name: ${NAME:-glm52-v11}
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
    ulimits:
      memlock: -1
      stack: 67108864
      nofile:
        soft: 1048576
        hard: 1048576
    volumes:
      - /mnt:/mnt
      - /cache:/cache
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/bench-results:/root/bench-results
      - /root/vllm/artifacts:/root/vllm/artifacts
      - /root/kld/checkpoints:/root/kld/checkpoints:ro
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      OMP_NUM_THREADS: "16"
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      USES_B12X: "True"
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: "1"
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}
      PORT: ${PORT:-5329}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-4}
      MTP: ${MTP:-0}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.96}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-256000}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      NUM_SPECULATIVE_TOKENS: ${NUM_SPECULATIVE_TOKENS:-3}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}
      KV_CACHE_DTYPE: ${KV_CACHE_DTYPE:-fp8}
      ATTENTION_BACKEND: ${ATTENTION_BACKEND:-B12X_MLA_SPARSE}
      MOE_BACKEND: ${MOE_BACKEND:-b12x}
      QUANTIZATION: ${QUANTIZATION:-modelopt_fp4}
      GLM52_INDEX_TOPK_PATTERN: ${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}
    entrypoint:
      - bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        if [ -z "$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]; then
          if [ "$${MTP}" = "1" ]; then
            MAX_CUDAGRAPH_CAPTURE_SIZE=$$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
          else
            MAX_CUDAGRAPH_CAPTURE_SIZE="$${MAX_NUM_SEQS}"
          fi
        fi
        HF_OVERRIDES="$$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$${GLM52_INDEX_TOPK_PATTERN}")"
        SPEC_ARGS=()
        if [ "$${MTP}" = "1" ]; then
          SPEC_CONFIG="$$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","draft_sample_method":"probabilistic"}' "$${MODEL}" "$${NUM_SPECULATIVE_TOKENS}" "$${MOE_BACKEND}")"
          SPEC_ARGS=(--speculative-config "$${SPEC_CONFIG}")
        fi
        QUANTIZATION_ARGS=()
        if [ "$${QUANTIZATION}" != "none" ]; then
          QUANTIZATION_ARGS=(--quantization "$${QUANTIZATION}")
        fi
        exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --tensor-parallel-size "$${TP_SIZE}" \
          --pipeline-parallel-size 1 \
          --max-model-len "$${MAX_MODEL_LEN}" \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --async-scheduling \
          -cc.pass_config.fuse_allreduce_rms=True \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --attention-backend "$${ATTENTION_BACKEND}" \
          --moe-backend "$${MOE_BACKEND}" \
          --kv-cache-dtype "$${KV_CACHE_DTYPE}" \
          --tool-call-parser glm47 \
          --enable-auto-tool-choice \
          --reasoning-parser glm45 \
          --hf-overrides "$${HF_OVERRIDES}" \
          "$${QUANTIZATION_ARGS[@]}" \
          "$${SPEC_ARGS[@]}"
```

## Capacity

Startup capacities observed:

| Profile | KV cache tokens | Max concurrency for 256k | Notes |
|---|---:|---:|---|
| DCP1, MTP off | 582,400 | 2.27x | `gpu_memory_utilization=0.96`, graph cap 64 |
| DCP4, MTP off | 2,329,600 | 9.10x | `gpu_memory_utilization=0.96`, graph cap 64 |
| DCP4, MTP5 | 2,060,543 | 8.05x | `gpu_memory_utilization=0.94`, `max_num_seqs=16`, graph cap 96; smoke only |

## Speed

Benchmark commands intentionally omitted `--max-tokens`; current
`llm_decode_bench.py` default was `8192`.

Prefill:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 --contexts 8k,64k,128k --concurrency 1 --run-prefill
```

Decode:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 --contexts 0k --concurrency 1,2,4,8,16,32,64 --skip-prefill
```

Prefill tok/s:

| DCP | MTP | 8k | 64k | 128k | Notes |
|---:|---|---:|---:|---:|---|
| 1 | off | 4,733 | 4,470 | 4,140 | `/root/bench-results/glm52-v11-20260618/dcp1-nomtp/prefill.json` |
| 4 | off | 3,084 | 2,794 | 2,738 | `/root/bench-results/glm52-v11-20260618/dcp4-nomtp/prefill.json` |
| 4 | MTP5 | not run | not run | not run | Startup + short coherence smoke only |

Aggregate decode tok/s, ctx 0:

| DCP | MTP | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | off | 78.9 | 134.3 | 219.6 | 326.4 | 494.2 | 676.4 | 865.7 |
| 4 | off | 64.2 | 100.7 | 156.1 | 239.0 | 343.1 | 447.5 | 557.3 |
| 4 | MTP5 | smoke | smoke | smoke | smoke | smoke | smoke | smoke |

## DCP4 + MTP Fix Verification

Original DCP4+MTP startup failed because the global DCP compatibility check
treated the local MTP layer as if it also used DCP4:

```text
AssertionError: Decode Context Parallelism (DCP) requires attention implementations to return the softmax LSE during decode, but B12xMLASparseImpl does not.
```

The final image uses the per-layer `dcp_world_size` check. DCP4+MTP5 then
started with:

```text
KV cache: 2,060,543 tokens
Maximum concurrency for 256,000 tokens per request: 8.05x
Graph capture finished in 119s, took 1.23 GiB
```

Smoke command:

```bash
timeout 90 python3 /mnt/test.py --port 5329 -L -c 512
```

Result: 6 iterations before timeout, coherent output, no CJK watchdog hit.
Generation-only throughput was about `63-65 tok/s`.
