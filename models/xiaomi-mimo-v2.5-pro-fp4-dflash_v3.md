# Xiaomi MiMo V2.5 Pro FP4-DFlash v3

This page documents the Fathomless Firmament validation for
`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` on RTX 6000 Pro Blackwell.

The published FF v15 image needs the vLLM fix from PR #85 for MiMo DFlash. Until
that PR is merged into `dev/fathomless-firmament` and the image is rebuilt, run
the image with the one-file bind mount shown below.

## Image

```text
voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709
voipmonitor/vllm@sha256:2dbc40a1fd104168226f46eb31f14301967a37aca95fed71fd23ff4f74b10698
```

| Component | Revision |
|---|---|
| vLLM base | `dev/fathomless-firmament @ 4cf20be86` |
| MiMo DFlash fix | PR #85, `293de7cda` |
| B12X | `90172a504e96d246e07cb1ebad3b291532445560` |
| CUDA | 13.2 |

PR:

```text
https://github.com/local-inference-lab/vllm/pull/85
```

## Model

```text
XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash
/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
```

Do not pass `--max-model-len` for the standard launch. The model reports
`max_model_len=1048576`.

## Fix

FF rebased `compute_kv_seq_mask()` to the signature:

```text
(query_abs_pos, seq_offset, seq_idx, seq_len, ...)
```

but `kernel_unified_attention()` still called it as:

```text
(query_abs_pos, seq_offset, seq_len, seq_idx, ...)
```

That swaps sequence index and sequence length. For MiMo DFlash / per-sequence
causal metadata, the draft attention mask then reads the wrong per-sequence
causal flag and uses the wrong length. The target path can still produce
coherent output, but DFlash draft acceptance collapses.

Expected fast markers:

```text
Using AttentionBackendEnum.TRITON_ATTN backend.
kernel_unified_attention
```

Bad slow marker:

```text
kernel_unified_attention_diffkv
```

## Docker Run

```bash
IMAGE=voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709
MODEL=/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
PATCHED_VLLM=/root/vllm/worktrees/vllm-ff-mimo-dflash-seqmask-20260709
CACHE=/root/.cache/vllm-mimo25-dflash-v3

docker rm -f mimo25-dflash-v3 2>/dev/null || true
mkdir -p "$CACHE"

docker run -d --name mimo25-dflash-v3 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/cache \
  -v "$PATCHED_VLLM/vllm/v1/attention/ops/triton_unified_attention.py":/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro \
  -e MODEL_PATH="$MODEL" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  "$IMAGE" \
  /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_USE_B12X_MOE B12X_MOE_FORCE_A16 VLLM_USE_B12X_FP8_GEMM VLLM_PCIE_ALLREDUCE_BACKEND VLLM_ENABLE_PCIE_ALLREDUCE VLLM_DISABLED_KERNELS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS; exec vllm serve "$MODEL_PATH" --served-model-name mimo-v25-pro-fp4-dflash --host 0.0.0.0 --port 8000 --trust-remote-code --kv-cache-dtype fp8 --block-size 64 --tensor-parallel-size 8 --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 16384 --max-cudagraph-capture-size 128 --attention-backend TRITON_ATTN --kernel-config.moe_backend flashinfer_cutlass --kernel-config.linear_backend b12x --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"custom_ops\":[\"all\"]}" --async-scheduling --no-scheduler-reserve-full-isl --enable-chunked-prefill --enable-prefix-caching --speculative-config "{\"model\":\"$MODEL_PATH/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":7}"'
```

## Bench Commands

```bash
python3 /mnt/test.py --port 8000 -c 0 --max-tokens 256 --quiet --json-summary /tmp/mimo_smoke.json

python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8000 \
  --model mimo-v25-pro-fp4-dflash \
  --concurrency 1 \
  --contexts 0 \
  --duration 30 \
  --max-tokens 8192 \
  --skip-prefill \
  --output /root/bench-results/mimo-v3/decode_cc1.json \
  --display-mode plain

python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8000 \
  --model mimo-v25-pro-fp4-dflash \
  --prefill-only \
  --standalone-prefill \
  --prefill-contexts 8k,64k \
  --prefill-duration 10 \
  --output /root/bench-results/mimo-v3/prefill_8k_64k.json \
  --display-mode plain
```

## Validation

Measured on GPUs 0-7 only. GPUs 8-15 were left running the existing GLM
container and were not used by these MiMo servers.

| Runtime | Decode cc1 tok/s | Prefill 8k tok/s | Prefill 64k tok/s | Notes |
|---|---:|---:|---:|---|
| FF v15 + PR #85 overlay | 145.74 | 8,394 | 6,080 | DFlash acceptance restored; CJK 0 |
| Eldritch v2 control rerun | 146.26 | - | - | Same machine state, same bench client |

Outlier note: an earlier Eldritch control run measured `159.65 tok/s`, and an
earlier FF + old DFlash bundle run measured `157.69 tok/s`. Re-running the same
old paths later in the same session did not reproduce those numbers (`146.26`
for Eldritch v2 and `133.14` for the full-old FF bundle), so the v3 comparison
uses the reproducible same-state control.

Result files:

```text
/root/bench-results/mimo-v3-ff-argfix-20260709T080216Z/decode_cc1.json
/root/bench-results/mimo-v3-ff-argfix-20260709T080216Z/prefill_8k_64k.json
/root/bench-results/mimo-v2-control-rerun-20260709T092609Z/decode_cc1.json
```

