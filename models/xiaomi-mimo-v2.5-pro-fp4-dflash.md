# Xiaomi MiMo V2.5 Pro FP4-DFlash

Updated on 2026-06-10. This page documents the working vLLM launch paths
for `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` on RTX 6000 Pro Blackwell.

The model is currently run without MTP. The validated paths are:

| Variant | Port | GPUs | Image | Backend summary |
|---|---:|---|---|---|
| Black unified CUTLASS+B12X linear | `5332` | `0-7` | `voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132` | `TRITON_ATTN`, `flashinfer_cutlass` MoE, B12X FP8 linear |
| B12X functional | `5333` | `8-15` | `voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132` | `TRITON_ATTN`, B12X FP8 linear, B12X MXFP4 MoE, B12X PCIe all-reduce |
| Lucifer/CUTLASS legacy | standalone `5332` | `0-7` | `voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610` | `TRITON_ATTN`, `flashinfer_cutlass` MoE, DeepGEMM FP8 linear |

Important: none of the working black-benediction MiMo variants use B12X
attention. MiMo is a normal QKV/DiffKV/SWA/sinks model, not a GLM/DeepSeek MLA
model. The working black launches keep attention on `TRITON_ATTN`.

## Model

Hugging Face model:

```text
XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash
```

Local snapshot used for all launches:

```text
/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
```

Current black-benediction `/v1/models` reports `max_model_len=1048576`.

## Required vLLM Patches

Black unified branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/mimo-v25-pro-fp4-dflash-black-unified-20260610
340ee7ad5
```

This branch is based on `dev/black-benediction` commit
`bb6c5b7351fceb9d524e0d43b957415ffefcb981`. It can run both black-benediction
MiMo modes:

| File | Why it is needed |
|---|---|
| `vllm/model_executor/layers/attention/attention.py` | Passes `head_size_v` to the experimental `B12X_PAGED_ATTN` path. The production compose below does not select it. |
| `vllm/model_executor/models/mimo_v2.py` | Stops forcing the FA2 DiffKV backend. MiMo has Q/K head dim `192` and V head dim `128`; the patch pads V to `192` for Triton/FA cache layout and slices attention output back before `o_proj`. It keeps attention sinks enabled. |
| `vllm/model_executor/layers/quantization/fp8.py` | Reads `store_dtype=mxfp4`. For FlashInfer/DeepSeek-style backends it routes routed experts to `Mxfp4MoEMethod`; for B12X/GPT-OSS-style it keeps `GptOssMxfp4MoEMethod`. |
| `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | Adds FlashInfer CUTLASS MXFP4 conversion for native contiguous `[gate, up]` expert weights, swaps to CUTLASS `[up, gate]`, injects zero biases when missing, and interleaves MXFP8 scales. |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` | Applies GPT-OSS alpha/beta/clamp only to `model_type=gpt_oss`. MiMo uses standard SwiGLU; without this, black CUTLASS MoE produces CJK/gibberish. |

Older B12X-only branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/mimo-v25-pro-fp4-dflash-b12x-triton-20260610
dbdedbb1e68c792dbea40762c716d9af4f303f35
```

This branch is still valid for the B12X functional compose, but the unified
branch above is preferred because it also supports the black CUTLASS MoE mode.

Lucifer/CUTLASS legacy branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/mimo-v25-pro-fp4-dflash-lucifer-cutlass-20260610
755bd7a149f4d8852ca41b180468aa702ae19e2e
```

This branch is based on image revision `bbbb9fc3b32f99c1fa6f17cf5c497d3f89a86682`
and contains only the two files mounted by the validated Lucifer container:

| File | Why it is needed |
|---|---|
| `vllm/model_executor/layers/quantization/fp8.py` | Reads `quantization_config.store_dtype`; routes `fp8 + store_dtype=mxfp4` routed experts to `Mxfp4MoEMethod`, which lets `--kernel-config.moe_backend flashinfer_cutlass` select `FLASHINFER_CUTLASS_MXFP4_MXFP8`. |
| `vllm/model_executor/models/mimo_v2.py` | Stops forcing the FA2 DiffKV backend. MiMo has Q/K head dim `192` and V head dim `128`; the patch pads V to `192` for Triton/FA cache layout and slices the attention output back before `o_proj`. It keeps attention sinks enabled. |

Do not set `VLLM_MIMO_DISABLE_ATTENTION_SINKS=1` for real runs. Disabling sinks
made the model start, but produced incoherent output.

Do not run with `NCCL_GRAPH_FILE=` set to an empty string. The compose files
below unset NCCL graph variables before `exec vllm serve`.

## Black Unified CUTLASS+B12X Linear Compose

Image:

```text
voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Prepare the patch worktree:

```bash
git clone https://github.com/local-inference-lab/vllm.git /root/vllm-mimo-black-unified
cd /root/vllm-mimo-black-unified
git checkout 340ee7ad5
```

This is the preferred single-image CUTLASS mode. It uses FlashInfer CUTLASS for
MXFP4 MoE and B12X for dense FP8 linear. Pure black-benediction CUTLASS dense
linear is not validated here; `linear_backend=auto` fails in
`cutlass_gemm_caller ... Invalid status` on this image.

```yaml
services:
  mimo25-black-cutlass-unified:
    image: voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132
    container_name: mimo25-black-cutlass-unified
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
    ulimits:
      memlock: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/.cache/vllm-mimo25-black-cutlass-unified-b12xlinear:/cache
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/attention/attention.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/models/mimo_v2.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/quantization/fp8.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/fp8.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:ro
    environment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
      CUTE_DSL_ARCH: sm_120a
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      XDG_CACHE_HOME: /cache/jit
      VLLM_CACHE_DIR: /cache/jit/vllm
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
    entrypoint:
      - /bin/sh
      - -c
      - >-
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
        VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_USE_B12X_MOE
        B12X_MOE_FORCE_A16 VLLM_PCIE_ALLREDUCE_BACKEND
        VLLM_ENABLE_PCIE_ALLREDUCE VLLM_CPP_AR_1STAGE_NCCL_CUTOFF
        VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS;
        exec vllm serve "$@"
      - --
    command:
      - /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
      - --served-model-name
      - mimo-v25-pro-fp4-dflash-cutlass-black
      - --host
      - 0.0.0.0
      - --port
      - "5332"
      - --trust-remote-code
      - --kv-cache-dtype
      - fp8
      - --block-size
      - "64"
      - --tensor-parallel-size
      - "8"
      - --gpu-memory-utilization
      - "0.90"
      - --max-num-seqs
      - "64"
      - --max-num-batched-tokens
      - "8192"
      - --max-cudagraph-capture-size
      - "64"
      - --attention-backend
      - TRITON_ATTN
      - --kernel-config.moe_backend
      - flashinfer_cutlass
      - --kernel-config.linear_backend
      - b12x
      - --reasoning-parser
      - mimo
      - --enable-auto-tool-choice
      - --tool-call-parser
      - mimo
      - --compilation-config
      - '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}'
      - --async-scheduling
      - --no-scheduler-reserve-full-isl
      - --enable-chunked-prefill
      - --enable-prefix-caching
```

Expected log checks:

```text
Using max model len 1048576
GPU KV cache size: 1,875,650 tokens
AttentionBackendEnum.TRITON_ATTN
Selected B12xFp8BlockScaledMMKernel
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8'
Using contiguous w13 layout for FlashInfer Cutlass MXFP4 MoE
Using standard SwiGLU parameters for FlashInfer MXFP4 MoE (model_type=mimo_v2)
Graph capturing finished
```

Validation run:

```text
timeout 240 python3 /mnt/test.py --port 5332 -L
```

Result: 30 full iterations completed before timeout, all with `0` CJK
characters. Generation-only throughput was stable around `95 tok/s`.

## Lucifer/CUTLASS Compose

Image:

```text
voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610
voipmonitor/vllm@sha256:3b6806ca1a38352cb0b0549cc23ae8a88c5d54f6c0042080d0b35f647b44eafb
```

Source labels in the image:

| Component | Revision |
|---|---|
| vLLM image source | `local-inference-lab/vllm`, branch `lucifer`, commit `7c6bbf4c5a482e100af886c5b6eb4303746cc3ba` |
| Image overlay revision | `bbbb9fc3b32f99c1fa6f17cf5c497d3f89a86682` |
| FlashInfer | PR3395, `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| DeepGEMM | PR324, `aced12c2c8882a945c568ace9d4a7e5778aae410` |
| B12X installed package | PR11, `d90d89c8353adabb56cc84bd3924ef811ef8d877` |
| CUDA | `13.2.1` |
| cuBLAS | image label `13.4.1.2-1`; runtime env also carries NVIDIA base env `13.4.0.1-1` |
| cuDNN | image label `9.22.0.52-1` |
| NCCL | patched `2.30.4` |

Prepare the patch worktree:

```bash
git clone https://github.com/local-inference-lab/vllm.git /root/vllm-mimo-lucifer
cd /root/vllm-mimo-lucifer
git checkout 755bd7a149f4d8852ca41b180468aa702ae19e2e
```

The compose below matches the validated `mimo25-lucifer-cutlass` runtime
parameters. That container originally mounted the same two files from
`/root/vllm/worktrees/vllm-lucifer-glm51-mxfp4-20260610`; this reproducible
version points at the clean branch checkout.

```yaml
services:
  mimo25-lucifer-cutlass:
    image: voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610
    container_name: mimo25-lucifer-cutlass
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
    ulimits:
      memlock: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/.cache/vllm-mimo25-lucifer-cutlass:/cache
      - /root/vllm-mimo-lucifer/vllm/model_executor/models/mimo_v2.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro
      - /root/vllm-mimo-lucifer/vllm/model_executor/layers/quantization/fp8.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/fp8.py:ro
    environment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
      CUTE_DSL_ARCH: sm_120a
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      XDG_CACHE_HOME: /cache/jit
      VLLM_CACHE_DIR: /cache/jit/vllm
      VLLM_USE_V2_MODEL_RUNNER: "1"
    entrypoint:
      - /bin/sh
      - -c
      - >-
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
        VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_USE_B12X_MOE
        VLLM_USE_B12X_FP8_GEMM B12X_MOE_FORCE_A16
        VLLM_PCIE_ALLREDUCE_BACKEND VLLM_ENABLE_PCIE_ALLREDUCE
        VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS;
        exec vllm serve "$@"
      - --
    command:
      - /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
      - --served-model-name
      - mimo-v25-pro-fp4-dflash-cutlass
      - --host
      - 0.0.0.0
      - --port
      - "5332"
      - --trust-remote-code
      - --kv-cache-dtype
      - fp8
      - --block-size
      - "64"
      - --tensor-parallel-size
      - "8"
      - --gpu-memory-utilization
      - "0.90"
      - --max-num-seqs
      - "64"
      - --max-num-batched-tokens
      - "8192"
      - --max-cudagraph-capture-size
      - "64"
      - --attention-backend
      - TRITON_ATTN
      - --kernel-config.moe_backend
      - flashinfer_cutlass
      - --reasoning-parser
      - mimo
      - --enable-auto-tool-choice
      - --tool-call-parser
      - mimo
      - --compilation-config
      - '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}'
      - --async-scheduling
      - --no-scheduler-reserve-full-isl
      - --enable-chunked-prefill
      - --enable-prefix-caching
```

Expected log checks:

```text
Using max model len 1048576
GPU KV cache size: 2,213,124 tokens
AttentionBackendEnum.TRITON_ATTN
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8'
Graph capturing finished
```

Claude Code/gateway note: vLLM must be started with
`--enable-auto-tool-choice --tool-call-parser mimo` if Anthropic-compatible
clients send `tool_choice=auto`. If raw reasoning should be hidden from Claude
Code, set the gateway to send:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

## B12X Functional Compose

Image:

```text
voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Source labels in the image:

| Component | Revision |
|---|---|
| vLLM | `local-inference-lab/vllm`, branch `dev/black-benediction`, commit `bb6c5b7351fceb9d524e0d43b957415ffefcb981` |
| B12X | PR11, `d90d89c8353adabb56cc84bd3924ef811ef8d877` |
| FlashInfer | PR3395, `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| DeepGEMM | PR324, `aced12c2c8882a945c568ace9d4a7e5778aae410` |
| CUDA | `13.2.1` |
| cuBLAS | image label `13.4.1.2-1`; runtime env also carries NVIDIA base env `13.4.0.1-1` |
| cuDNN | image label `9.22.0.52-1` |
| NCCL | patched `2.30.4` |

Prepare the patch worktree:

```bash
git clone https://github.com/local-inference-lab/vllm.git /root/vllm-mimo-black-unified
cd /root/vllm-mimo-black-unified
git checkout 340ee7ad5
```

The compose below matches the live `mimo25-b12x-stable-triton` runtime
parameters. It uses B12X MoE/linear/all-reduce, but keeps MiMo attention on
`TRITON_ATTN`. It uses the same unified branch checkout as the CUTLASS mode.

```yaml
services:
  mimo25-b12x-stable-triton:
    image: voipmonitor/vllm:black-benediction-bb6c5b7-b12xd90d89c-cu132
    container_name: mimo25-b12x-stable-triton
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
    ulimits:
      memlock: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/.cache/vllm-mimo25-b12x-triton-stable:/cache
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/attention/attention.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/models/mimo_v2.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro
      - /root/vllm-mimo-black-unified/vllm/model_executor/layers/quantization/fp8.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/fp8.py:ro
    environment:
      CUDA_VISIBLE_DEVICES: "8,9,10,11,12,13,14,15"
      CUTE_DSL_ARCH: sm_120a
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      XDG_CACHE_HOME: /cache/jit
      VLLM_CACHE_DIR: /cache/jit/vllm
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      B12X_MOE_FORCE_A16: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
    entrypoint:
      - /bin/sh
      - -c
      - >-
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
        VLLM_B12X_MLA_EXTEND_MAX_CHUNKS;
        exec vllm serve "$@"
      - --
    command:
      - /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
      - --served-model-name
      - mimo-v25-pro-fp4-dflash
      - --host
      - 0.0.0.0
      - --port
      - "5333"
      - --trust-remote-code
      - --kv-cache-dtype
      - fp8
      - --block-size
      - "64"
      - --tensor-parallel-size
      - "8"
      - --gpu-memory-utilization
      - "0.90"
      - --max-num-seqs
      - "64"
      - --max-num-batched-tokens
      - "8192"
      - --max-cudagraph-capture-size
      - "64"
      - --attention-backend
      - TRITON_ATTN
      - --kernel-config.moe_backend
      - b12x
      - --kernel-config.linear_backend
      - b12x
      - --compilation-config
      - '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}'
      - --async-scheduling
      - --no-scheduler-reserve-full-isl
      - --enable-chunked-prefill
      - --enable-prefix-caching
```

Expected log checks:

```text
Using max model len 1048576
GPU KV cache size: 1,993,210 tokens
AttentionBackendEnum.TRITON_ATTN
Selected B12xFp8BlockScaledMMKernel
Using 'B12X' Mxfp4 MoE backend
Using b12x PCIe oneshot allreduce backend
Graph capturing finished
```

The B12X PCIe oneshot path has a `65536` byte max size in this image. Logs show
small decode tensors accepted, for example `(4, 6144) bf16`, while large prefill
tensors fall back to `PYNCCL`.

## Smoke Checks

Health/model metadata:

```bash
curl -fsS http://127.0.0.1:5332/v1/models
curl -fsS http://127.0.0.1:5333/v1/models
```

Coherence smoke:

```bash
python3 /mnt/test.py --port 5332 -L
python3 /mnt/test.py --port 5333 -L
```

Backend log grep:

```bash
docker logs mimo25-black-cutlass-unified 2>&1 | grep -E "TRITON_ATTN|FLASHINFER_CUTLASS|standard SwiGLU|B12xFp8|GPU KV cache|Graph capturing"
docker logs mimo25-b12x-stable-triton 2>&1 | grep -E "TRITON_ATTN|B12X|GPU KV cache|Graph capturing"
```

## Known Bad Paths

| Path | Result |
|---|---|
| Disable MiMo attention sinks with `VLLM_MIMO_DISABLE_ATTENTION_SINKS=1` | Server can start, but generation becomes incoherent. |
| Stock FA2 DiffKV forced by old `mimo_v2.py` | Fails because MiMo attention sinks require a backend that supports sinks. |
| B12X MLA sparse attention for MiMo | Not applicable; MiMo is not an MLA model. |
| Black-benediction CUTLASS MoE without the standard-SwiGLU patch | Starts, but eventually produces CJK/gibberish because GPT-OSS alpha/beta/clamp is incorrectly applied to `model_type=mimo_v2`. |
| Black-benediction pure CUTLASS dense linear | `linear_backend=auto` fails in `cutlass_gemm_caller ... Invalid status`; use `--kernel-config.linear_backend b12x` in the unified black recipe. |
| Any compose without `--enable-auto-tool-choice --tool-call-parser mimo` behind Claude Code | Anthropic-compatible requests with `tool_choice=auto` fail with vLLM 400. |
