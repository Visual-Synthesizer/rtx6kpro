# Kimi-K2.6 FP8 Draft Full CUDA Graph Crash Fix

Date: 2026-05-18

## Summary

Kimi-K2.6 DCP8 with `festr2/kimi-k2.6-eagle3-mla-fp8` draft, FlashInfer autotune enabled, and `FULL_AND_PIECEWISE` CUDA graph capture crashed during startup in the old `voipmonitor/vllm:glm-kimi-20260517` image.

The failing path was the FlashInfer dense FP8 GEMM `backend="auto"` wrapper used by vLLM for `FlashInferFP8ScaledMMLinearKernel`. The crash had two observed forms:

- `IndexError: list index out of range` from FlashInfer autotuner before graph capture completed.
- CUDA illegal memory access during full CUDA graph capture/registration.

Disabling all FlashInfer autotune avoided the crash, but that was too broad. Disabling `FlashInferFP8ScaledMMLinearKernel` also avoided the crash by falling back to Cutlass, but that changed kernel selection globally for the FP8 draft.

## Fix

vLLM commit:

```text
5de88dbd6ca3486bd1dbe247fc08f96cdc7b2d2c kernel: pin FlashInfer fp8 GEMM backend
```

GitHub:

```text
https://github.com/voipmonitor/vllm/commit/5de88dbd6ca3486bd1dbe247fc08f96cdc7b2d2c
```

The patch keeps `FlashInferFP8ScaledMMLinearKernel` enabled, but pins the vLLM FlashInfer dense FP8 GEMM wrapper from `backend="auto"` to `backend="cublas"`. Other FlashInfer autotune users remain enabled.

## Docker Image

```text
voipmonitor/vllm:glm-kimi-20260518
```

Digest:

```text
sha256:af6d4fb63440aa150e9909b6d5fc97617e614962d6380022fb425096e33d211f
```

Important labels:

```text
voipmonitor.vllm.git_sha=5de88dbd6ca3486bd1dbe247fc08f96cdc7b2d2c
voipmonitor.flashinfer_fp8_gemm_backend=cublas
voipmonitor.flashinfer_autotune.default=O1_O2_O3
voipmonitor.b12x.git_sha=c929144c7689668b07ca65af10ceadf1c745165d
```

## Verification

Reproducer:

```bash
FP8='{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8"}'
IMAGE=voipmonitor/vllm:glm-kimi-20260518 \
MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
/tmp/kimi_fullgraph_repro.sh fp8_image_20260518 "$FP8"
```

Result:

```text
RESULT fp8_image_20260518 READY 200s
```

Startup log confirmed:

```text
Selected FlashInferFP8ScaledMMLinearKernel for Fp8LinearMethod
GPU KV cache size: 1,823,872 tokens
Capturing CUDA graphs (decode, FULL): 100%
Registering 492 cuda graph addresses
```

Smoke decode on the patched file before building the image, DCP8 MTP with `festr2` FP8 draft:

```text
ctx0 cc1:   89.5 tok/s
ctx128k cc1: 64.3 tok/s
```

This was a startup/crash fix verification, not a replacement for the full v5 benchmark sweep.
