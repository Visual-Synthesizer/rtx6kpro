# Qwen3.5 Smaller Variants (27B, 35B, 122B) on RTX PRO 6000 Blackwell

## Table of Contents

- [Overview](#overview)
- [Available Checkpoints](#available-checkpoints)
- [Hardware Requirements](#hardware-requirements)
- [Benchmark Results](#benchmark-results)
- [Launch Commands -- vLLM](#launch-commands----vllm)
- [Launch Commands -- llama.cpp](#launch-commands----llamacpp)
- [Launch Commands -- SGLang](#launch-commands----sglang)
- [MTP / Speculative Decoding](#mtp--speculative-decoding)
- [DFlash Speculative Decoding](#dflash-speculative-decoding)
- [Quality Comparisons](#quality-comparisons)
- [Known Issues](#known-issues)
- [Engine Compatibility Matrix](#engine-compatibility-matrix)

---

## Overview

The Qwen3.5 family includes several smaller models that share the same hybrid GDN/full attention architecture as the flagship 397B but at reduced parameter counts. These models are attractive for single-GPU or 2-GPU deployments.

| Model | Total Params | Active Params | Architecture |
| --- | --- | --- | --- |
| Qwen3.5-27B | 27B | 27B (dense) | Hybrid GDN/full attention |
| Qwen3.5-35B-A3B | 35B | 3B | MoE + hybrid GDN/full attention |
| Qwen3.5-122B-A10B | 122B | 10B | MoE + hybrid GDN/full attention |

Key finding from community testing: the 122B model shows "very little difference" from the 397B in benchmarks despite being >50% smaller.

---

## Available Checkpoints

### 27B (Dense)

| Checkpoint | Quantization | MTP | Multimodal | Notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-27B-FP8` | FP8 | Yes | Yes | Official FP8, best MTP support |
| `Sehyo/Qwen3.5-27B-NVFP4` | NVFP4 | Yes | Yes | Full-featured, compressed-tensors |
| `osoleve/Qwen3.5-27B-NVFP4-MTP` | NVFP4 | Yes (bf16 weights) | No | MTP weights restored in bf16 |
| `Kbenkhaled/Qwen3.5-27B-NVFP4` | NVFP4 | No | No | Minimal quant, no extras |

### 35B-A3B (MoE)

| Checkpoint | Quantization | MTP | Multimodal | Notes |
| --- | --- | --- | --- | --- |
| `Sehyo/Qwen3.5-35B-A3B-NVFP4` | NVFP4 | Yes | Yes | Full-featured |

### 122B-A10B (MoE)

| Checkpoint | Quantization | MTP | Multimodal | Notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-122B-A10B` | Official FP8/BF16 | Yes | Yes | Official release |
| `Sehyo/Qwen3.5-122B-A10B-NVFP4` | NVFP4 | Yes | Yes | Best tested, 312k downloads |
| `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | NVFP4 | **No** | Yes | MTP weights missing despite card claims |

### Sehyo NVFP4 Series (full family)

All Sehyo checkpoints include multimodal + MTP support:

- https://huggingface.co/Sehyo/Qwen3.5-27B-NVFP4
- https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4
- https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4

**Important:** Sehyo checkpoints (llm-compressor) have `kv_cache_scheme: null` -- no calibrated FP8 KV scales. They default to bf16 KV cache at runtime, using 2x more memory than calibrated FP8 KV cache. You can force `--kv-cache-dtype fp8` for dynamic quantization to reclaim VRAM -- quality impact is minimal for agentic workloads.

### RedHatAI checkpoint warning

`RedHatAI/Qwen3.5-122B-A10B-NVFP4` model card claims MTP support but **MTP weights are not included** in the safetensors files (verified: only `lm_head` + `model` prefixes present, no `mtp` prefix). Running with `--speculative-config` results in 0% acceptance rate. The checkpoint also requires a tokenizer patch (`TokenizersBackend` -> `Qwen2TokenizerFast`) to load on current vLLM/transformers. See [discussion #1](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4/discussions/1).

---

## Hardware Requirements

| Model | NVFP4 | FP8 / BF16 |
| --- | --- | --- |
| 27B | **1x RTX PRO 6000** (96 GB) | 1x RTX PRO 6000 |
| 35B-A3B | 1x RTX PRO 6000 | 1x RTX PRO 6000 |
| 122B-A10B | **2x RTX PRO 6000** | 2-4x RTX PRO 6000 |

---

## Benchmark Results

All benchmarks on 2x RTX PRO 6000 Blackwell Workstation Edition, TRX40 platform, PCIe Gen4 (no NVLink).

### Generation Speed Summary

| Model | Engine | Checkpoint | GPU(s) | Spec Decode | Gen tok/s | MTP Accept |
| --- | --- | --- | --- | --- | --- | --- |
| 397B-A17B | llama.cpp | Unsloth Q3_K_XL GGUF | 2x | None | **70** | n/a |
| 122B-A10B | vLLM 0.19.0 | Sehyo NVFP4 (enforce-eager) | 2x | None | 11 | n/a |
| 122B-A10B | vLLM 0.19.0 | Sehyo NVFP4 (CUDA graphs) | 2x | None | **115** | n/a |
| 122B-A10B | vLLM nightly | Sehyo NVFP4 | 2x | MTP=2 | **100-117** | 85-96% |
| 27B | vLLM nightly | Official FP8 | 1x | MTP=2 | **63** | 82% |
| 27B | vLLM 0.19.0 | Sehyo NVFP4 | 1x | None | **48** | n/a |
| 27B | vLLM nightly | Official FP8 | 2x | MTP=2 | 70 | 73% |

### Key Findings

**CUDA graphs are critical for MoE models.** Removing `--enforce-eager` from the 122B config produced a **10x speedup** (11 -> 115 tok/s). This is the single most impactful optimization discovered. The `--enforce-eager` flag is included in many example configs online and silently costs an order of magnitude on MoE architectures.

**MTP on vLLM 0.19.0 is broken for all Qwen3.5.** Both `"method":"mtp"` and `"method":"qwen3_next_mtp"` produce 0% acceptance rates on the 0.19.0 release, regardless of checkpoint (Sehyo NVFP4, official FP8). This is caused by PR #34552. The vLLM nightly (0.19.1rc1+) fixes this.

**MTP acceptance rate does not translate to proportional speedup on TP=2 over PCIe.** Despite 91%+ acceptance, MTP on the 122B only matches or slightly underperforms the non-MTP baseline due to cross-GPU allreduce overhead per speculative step. MTP is more beneficial on single-GPU configs (27B).

**TP=2 provides minimal benefit for dense 27B models.** The 27B with MTP on TP=2 (70 tok/s) barely exceeds TP=1 (63 tok/s). PCIe communication overhead almost entirely negates the benefit of splitting a dense model across two cards.

### Prompt Throughput

| Model | Engine | Prompt tok/s |
| --- | --- | --- |
| 122B-A10B | vLLM | 3,300-16,400 |
| 27B | vLLM | 15,800 |

### Prefix Cache Performance

Prefix caching (`--enable-prefix-caching`) consistently reaches 40-60% hit rates during multi-turn agent conversations, providing significant prompt processing speedups on repeated system prompts and tool definitions.

---

## Launch Commands -- vLLM

### 122B-A10B NVFP4 -- Recommended (2x GPUs, no MTP)

Best overall performance config. Requires vLLM 0.19.0+.

```bash
#!/usr/bin/env bash
set -euo pipefail
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export VLLM_LOG_STATS_INTERVAL=1
export SAFETENSORS_FAST_GPU=1

vllm serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name Qwen3.5-122B \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --quantization compressed-tensors \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --mm-processor-cache-type shm \
  --host 0.0.0.0 \
  --port 8000
```

**Notes:**
- Do NOT use `--enforce-eager` -- this costs 10x performance on MoE models.
- `--quantization compressed-tensors` is correct for all Sehyo/llm-compressor checkpoints (not `modelopt`).
- `--kv-cache-dtype fp8` forces dynamic FP8 KV quantization since Sehyo checkpoints lack calibrated scales.
- `VLLM_SLEEP_WHEN_IDLE=1` is not recognized in vLLM 0.19.0+.

### 122B-A10B NVFP4 with MTP (2x GPUs, nightly required)

Requires vLLM nightly (0.19.1rc1+). Achieves 100-117 tok/s with 85-96% acceptance.

```bash
#!/usr/bin/env bash
set -euo pipefail
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export VLLM_LOG_STATS_INTERVAL=1
export SAFETENSORS_FAST_GPU=1

vllm serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name Qwen3.5-122B \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8 \
  --quantization compressed-tensors \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"rejection_sample_method":"probabilistic"}' \
  --host 0.0.0.0 \
  --port 8000
```

**Notes:**
- `--disable-custom-all-reduce` is required on PCIe (non-NVLink) topologies with the nightly to avoid CUDA illegal memory access during graph capture.
- `--max-model-len 32768` and `--gpu-memory-utilization 0.90` are needed to fit the MTP drafter (loads as unquantized bf16 MoE layer).
- MTP on TP=2 over PCIe does not consistently beat the non-MTP baseline due to allreduce overhead.

### 27B FP8 with MTP (single GPU, nightly required)

Best single-GPU config. Leaves GPU 1 free for ComfyUI/FLUX.

```bash
#!/usr/bin/env bash
set -euo pipefail
export VLLM_LOG_STATS_INTERVAL=1
export SAFETENSORS_FAST_GPU=1

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --served-model-name Qwen3.5-27B \
  --gpu-memory-utilization 0.95 \
  --max-model-len 128000 \
  --max-num-seqs 512 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"rejection_sample_method":"probabilistic"}' \
  --host 0.0.0.0 \
  --port 8001
```

**Notes:**
- Use the official `Qwen/Qwen3.5-27B-FP8` checkpoint, NOT Sehyo NVFP4. MTP on Sehyo NVFP4 produces 0% acceptance even on the nightly.
- `--max-num-seqs 512` prevents "max_num_seqs exceeds available Mamba cache blocks" error from the hybrid GDN architecture.
- `--attention-backend TRITON_ATTN` as used in community configs reporting best MTP performance.

### 27B NVFP4 without MTP (single GPU, vLLM 0.19.0)

Simpler config, no nightly required. 48 tok/s baseline.

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOG_STATS_INTERVAL=1
export SAFETENSORS_FAST_GPU=1

vllm serve Sehyo/Qwen3.5-27B-NVFP4 \
  --served-model-name Qwen3.5-27B \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128000 \
  --quantization compressed-tensors \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --mm-processor-cache-type shm \
  --host 0.0.0.0 \
  --port 8001
```

---

## Launch Commands -- llama.cpp

### 397B-A17B GGUF (2x GPUs)

70 tok/s generation with Unsloth Q3_K_XL (3.61 BPW, 166 GB). Near-original quality (< 1 point accuracy drop on 750-prompt benchmarks).

```bash
llama-server \
  --model Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 61 \
  --port 8080
```

**Notes:**
- `BLACKWELL_NATIVE_FP4 = 1` is auto-detected on Blackwell GPUs.
- Fused GDN (Gated Delta Net) kernels are auto-enabled.
- Speculative decoding is not currently supported for the hybrid GDN architecture in llama.cpp.
- Uses both GPUs automatically via layer offloading.

---

## Launch Commands -- SGLang

**WARNING:** SGLang does not currently work with Sehyo NVFP4 checkpoints for Qwen3.5. The `compressed-tensors` quantization loader fails to map GDN `linear_attn` weight names, resulting in NaN output (`"matched_stop":"NaN happened"`). The `modelopt_fp4` quantization flag is rejected because the checkpoint metadata says `compressed-tensors`.

The SGLang commands below are included for reference but have **not been successfully validated** on this hardware with Sehyo checkpoints.

### 27B NVFP4 (single GPU) -- BROKEN

```
python -m sglang.launch_server \
  --model-path Sehyo/Qwen3.5-27B-NVFP4 \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --quantization compressed-tensors \
  --attention-backend triton \
  --context-length 128000 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sleep-on-idle
```

### 122B-A10B NVFP4 (2x GPUs) -- UNTESTED

```
python -m sglang.launch_server \
  --model-path Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --quantization compressed-tensors \
  --attention-backend triton \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sleep-on-idle
```

**Note:** Remove `--moe-runner-backend` and `--fp4-gemm-backend` flags for the 27B (dense, not MoE). These flags cause assertion errors with `compressed-tensors` quantization.

---

## MTP / Speculative Decoding

### vLLM MTP Status

MTP on vLLM requires careful checkpoint + version matching:

| Checkpoint | vLLM 0.19.0 | vLLM Nightly | Acceptance Rate |
| --- | --- | --- | --- |
| Sehyo 122B NVFP4 | **0% (broken)** | **85-96%** | Working on nightly only |
| Sehyo 27B NVFP4 | **0% (broken)** | **0% (broken)** | Broken on both |
| Official 27B FP8 | **0% (broken)** | **82%** | Working on nightly only |
| RedHatAI 122B NVFP4 | n/a | **0%** | MTP weights missing from checkpoint |

**Root cause:** PR [#34552](https://github.com/vllm-project/vllm/pull/34552) broke MTP for Qwen3.5 in vLLM. The nightly fixes this for most checkpoint formats, but Sehyo NVFP4 27B remains broken (likely a `compressed-tensors` weight mapping issue specific to the dense architecture).

### vLLM MTP config

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"rejection_sample_method":"probabilistic"}'
```

**Note:** `"method":"qwen3_next_mtp"` is deprecated in the nightly and auto-redirects to `"mtp"`.

### SGLang MTP config (for future reference)

```
--speculative-algo NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

### Checkpoint MTP weight compatibility

MTP weights are stored in `extra_weights.safetensors` with prefix `mtp.*`. Verify presence with:

```python
from safetensors import safe_open
import glob
for f in glob.glob('model_dir/*.safetensors'):
    with safe_open(f, framework='pt') as st:
        mtp_keys = [k for k in st.keys() if 'mtp' in k.lower()]
        if mtp_keys:
            print(f'{f}: {len(mtp_keys)} MTP keys found')
```

---

## DFlash Speculative Decoding

[DFlash](https://github.com/z-lab/dflash) uses block diffusion to draft an entire block of tokens in a single parallel forward pass, achieving up to 6x lossless speedup. Community reports on 2x RTX PRO 6000:

| Config | Gen tok/s |
| --- | --- |
| 27B FP8 + DFlash 15 tokens (1x GPU) | ~116 |
| 27B FP8 + DFlash 15 tokens (2x GPU) | ~160 |
| 27B FP8 + MTP (1x GPU) | ~80 |
| 27B FP8 no spec decode | ~46 |

**vLLM command:**

```
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

**Status:** Requires access to the gated `z-lab/Qwen3.5-27B-DFlash` model. No 122B DFlash drafter model exists yet.

---

## Quality Comparisons

### Qwen3.5 vs competitors (community vibes-based assessment)

- **vs MiniMax M2.5**: M2.5 better for coding/agentic tasks. Qwen better for general reasoning, long context (2x better), instruction following, vision.
- **vs GLM-4.7**: Qwen on par overall. GLM slightly better for some coding tasks. Qwen "70% cases more detailed" in RAG pipelines.
- **vs Claude Opus 4.6 / Codex 5.3**: "Giant gap" -- Opus/Codex far superior for coding. Qwen good as "workhorse / code writer" with Opus as architect.
- **122B vs 397B**: "very little difference" in benchmarks.

---

## Known Issues

### enforce-eager costs 10x on MoE models

The `--enforce-eager` flag disables CUDA graph capture and costs approximately 10x generation speed on MoE models (11 vs 115 tok/s on the 122B). Many example configs online include this flag. Remove it unless you are debugging OOM issues.

### MTP broken on vLLM 0.19.0

All Qwen3.5 MTP is broken on vLLM 0.19.0 (0% acceptance rate). Use the nightly. See issues [#36031](https://github.com/vllm-project/vllm/issues/36031), [#36331](https://github.com/vllm-project/vllm/issues/36331), [#36872](https://github.com/vllm-project/vllm/issues/36872).

### SGLang + Sehyo compressed-tensors = NaN

SGLang's `compressed-tensors` loader does not correctly map GDN `linear_attn` weight names for Qwen3.5. Hundreds of `Parameter not found in params_dict` warnings are emitted during loading, and inference produces NaN. The `modelopt_fp4` flag is rejected because of a quantization config mismatch.

### RedHatAI checkpoint missing MTP weights

`RedHatAI/Qwen3.5-122B-A10B-NVFP4` does not include MTP weights despite the model card showing MTP launch commands. Additionally requires tokenizer patch (`TokenizersBackend` -> `Qwen2TokenizerFast`).

### Sehyo KV cache

Sehyo checkpoints default to bf16 KV cache (no calibrated FP8 scales). Forcing `--kv-cache-dtype fp8` enables dynamic quantization, reducing KV VRAM by ~2x at minimal quality cost for agentic workloads.

### Kbenkhaled checkpoint limitations

The `Kbenkhaled/Qwen3.5-27B-NVFP4` checkpoint has no multimodal support and no MTP weights.

### Tool calling issues

When MTP + thinking mode are both enabled, tool calls may output XML instead of JSON. See PR #35936 for the fix.

### Hybrid GDN architecture quirks

- vLLM's memory profiler may calculate `num_gpu_blocks=0` for hybrid GDN models, then override to a working value. This is cosmetic.
- Prefix caching in Mamba cache "align" mode is experimental. Works in practice but may have edge cases.
- `max_num_seqs` must not exceed available Mamba cache blocks (typically ~950 on a single 96GB GPU at 0.95 utilization).

---

## Engine Compatibility Matrix

| Feature | vLLM 0.19.0 | vLLM Nightly | SGLang | llama.cpp |
| --- | --- | --- | --- | --- |
| Sehyo NVFP4 inference | Yes | Yes | **No (NaN)** | n/a (GGUF only) |
| Official FP8 inference | Yes | Yes | Untested | n/a |
| GGUF inference | n/a | n/a | n/a | Yes |
| MTP (Sehyo 122B) | **Broken** | **Working** | n/a | Not supported |
| MTP (Official 27B FP8) | **Broken** | **Working** | Untested | Not supported |
| MTP (Sehyo 27B NVFP4) | **Broken** | **Broken** | n/a | Not supported |
| DFlash | No | In progress | Supported | No |
| CUDA graphs | Yes | Yes | Yes | n/a |
| Prefix caching | Yes | Yes | Yes (radix) | Yes (prompt cache) |
| Tool calling | Yes | Yes | Yes | Yes |
| Multimodal | Yes | Yes | **Broken** | Yes (mmproj) |
