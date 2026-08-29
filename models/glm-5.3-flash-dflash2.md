# GLM-5.3-Flash NVIDIA 4-bit floating-point (NVFP4) with DFlash2 Microscaling 8-bit floating-point (MXFP8)

This runbook serves the `local-inference-lab/GLM-5.3-Flash-NVFP4` target with
the serialized `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8`
Diffusion/Block-Diffusion Flash draft path (DFlash), version 2 (DFlash2), on
four NVIDIA RTX PRO 6000 Blackwell GPUs. The vLLM inference engine uses Tensor
Parallelism (TP), with four ranks (TP4), and Decode Context Parallelism (DCP),
also with four ranks (DCP4). The target key-value (KV) cache is
sequence-sharded, while every DCP rank keeps a complete DFlash2 cache. The B12X
kernel/backend stack supplies the target and draft compute kernels described
below.

## Status

| Field | Value |
|---|---|
| Runtime status | qualified |
| Hardware | four RTX PRO 6000 Blackwell GPUs |
| Target quantization | ModelOpt NVIDIA 4-bit floating-point (NVFP4); B12X 4-bit-weight, 4-bit-activation (W4A4) routed experts |
| Draft quantization | serialized ModelOpt Microscaling 8-bit floating-point (MXFP8) weights |
| Target KV cache | 8-bit floating-point (FP8) compressed Multi-head Latent Attention (MLA) |
| Draft KV cache | automatic dtype, resolving to Brain Floating Point 16-bit (BF16) for this checkpoint |
| Decode graphs | target and DFlash2 full decode graphs through concurrency 16 |
| Prefill graphs | unsupported by the GLM Gated Delta Network backend; prefill is eager |
| Source review | vLLM pull requests 509, 510, 511, 513, 515, 516, 517, 519, and 520 are open; human review is required before merge |

The qualified artifact is a source-locked review snapshot. Its behavior does
not imply that the open pull requests are present in
`dev/jovian-judgement` until they are merged.

## Docker artifact

```text
voipmonitor/vllm:glm53-flash-nvfp4-dflash2-mxfp8-dcp4-pagealign-vllmbfd30c0d-b12xd56c1163-cu133-torch213-20260829-r2
voipmonitor/vllm@sha256:9a0ce5badb50ac93647bf517573ac5578dee0dc8beccd58e699f44767b14495f
```

The local image ID used for qualification is
`sha256:2897d7a659453f256af37c26fa8aa933885af7cd28c8bda25c85912f8801c94c`.
The `local-inference.runtime.source-lock.sha256` Open Container Initiative
(OCI) label is
`7d92e36e7e3d22fa4632f7b91bfdc0612a7ca267c7690d062f1c8cebe0aec26b`.

## Source contract

| Component | Revision or review |
|---|---|
| vLLM base | `dev/jovian-judgement@766acf0e218a075432e6c45755cd561ab765ec2d` |
| vLLM integration commit | `bfd30c0db01846a1de2e6a47d33aac3fb970b759` |
| vLLM integration tree | `559049e2e214e3091d08138dab7ffe95cb10fd11` |
| Hybrid KV page packing | [vLLM #509](https://github.com/local-inference-lab/vllm/pull/509) at `3ac643b8b93b21041bcc3768727302515d53548d` |
| Unbounded hybrid cache anchor | [vLLM #510](https://github.com/local-inference-lab/vllm/pull/510) at `8b83d9bb66c611a8a004481c88c91f9bc767213e` |
| Exact small-graph profiling | [vLLM #511](https://github.com/local-inference-lab/vllm/pull/511) at `905ac2b74f988e61932b64680b03dbfedf021f19` |
| Replicated DFlash cache under DCP | [vLLM #513](https://github.com/local-inference-lab/vllm/pull/513) at `e5e7bf99182833c6ce25042c29252bbb4107539c` |
| Replicated/sharded cache-group isolation | [vLLM #519](https://github.com/local-inference-lab/vllm/pull/519) at `00c463493591031d1d90be73994cf5f92c6e350e` |
| Replicated draft page alignment | [vLLM #520](https://github.com/local-inference-lab/vllm/pull/520) at `fd47007a6adc95f1a947d80dc2fd50745e22bdcc` |
| B12X | `d56c1163b6e019d828ed24f135c2efd05fdca6ea` |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Draft checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8@b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c` |
| CUDA / PyTorch | Compute Unified Device Architecture (CUDA) 13.3 / PyTorch 2.13 development snapshot |

The integration also contains the full compressed-KV prefill stack from vLLM
pull requests [515](https://github.com/local-inference-lab/vllm/pull/515),
[516](https://github.com/local-inference-lab/vllm/pull/516), and
[517](https://github.com/local-inference-lab/vllm/pull/517). Original Git
authors are retained in the composed source history.

## Cache invariant

The target sparse Multi-head Latent Attention (MLA) cache is sequence-sharded
across four DCP ranks. Its B12X C4 prefill indexer gathers the complete
compressed KV state before candidate ranking.

The DFlash2 attention layer owns TP-local heads and has no cross-rank
log-sum-exp reduction. Every DCP rank must therefore retain the complete draft
cache and execute draft attention as a local DCP1 operation. Target cache
groups use an effective context-parallel size of 4; the draft group uses 1.

The planner keeps replicated draft groups separate from sharded target groups.
It aligns the draft sliding-window page with the greatest common target
physical page only when the draft block divides that alignment and the larger
natural draft page does not increase the shared BlockPool stride. Otherwise it
preserves the attention backend's native draft block.

For the pinned checkpoints the effective cache-group context-parallel sizes are
`(4, 4, 4, 4, 4, 1)`, and all six physical block sizes are 2304 tokens. The
alignment does not increase the scheduler least common multiple.

## Runtime backends

| Operation | Selected backend |
|---|---|
| Target sparse MLA attention | B12X |
| Target C4 indexer, prefill and decode | B12X |
| Target routed experts | B12X NVFP4 W4A4 |
| Target linear layers | B12X |
| Tensor-parallel all-reduce | B12X Peripheral Component Interconnect Express (PCIe) first, PyNCCL outside the B12X dispatch range |
| Draft MXFP8 linear layers | B12X MXFP8 General Matrix Multiply (GEMM) |
| Draft fused context K/V projection | B12X MXFP8 GEMM |
| Replicated draft local attention | FlashAttention 2 |
| Sampling | FlashInfer |

DeepGEMM and TileLang are installed dependencies, but neither is selected for
the target or draft hot paths in this profile.

## Start the server

The image entrypoint pins both Hugging Face repositories and revisions. The
command uses named Docker volumes and does not require checkpoint paths or
source bind mounts.

```bash
IMAGE=voipmonitor/vllm:glm53-flash-nvfp4-dflash2-mxfp8-dcp4-pagealign-vllmbfd30c0d-b12xd56c1163-cu133-torch213-20260829-r2
GPU_DEVICES=0,1,2,3

docker run -d \
  --name glm53-dflash2-dcp4 \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v glm53-dflash2-cache:/cache \
  -v glm53-huggingface-cache:/root/.cache/huggingface \
  -e PORT=5001 \
  -e TP=4 \
  -e DCP=4 \
  -e CP_KV_CACHE_INTERLEAVE_SIZE=4 \
  -e SPECULATOR=dflash \
  -e NUM_SPECULATIVE_TOKENS=7 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
  -e CUDAGRAPH_MODE=FULL \
  -e GPU_MEMORY_UTILIZATION=0.95 \
  -e DCP_CKV_GATHER=1 \
  -e DFLASH_KV_CACHE_DTYPE=auto \
  -e DFLASH_ATTENTION_BACKEND=FLASH_ATTN \
  -e B12X_PCIE_ALLREDUCE=1 \
  -e GLM53_KDA_DECODE_BACKEND=auto \
  "$IMAGE"
```

The requested `CUDAGRAPH_MODE=FULL` resolves to `FULL_DECODE_ONLY` because the
GLM Gated Delta Network backend supports uniform-batch decode graphs but not
full prefill graphs. This is the qualified dispatch, not an eager-decode
workaround.

## Verify startup

```bash
docker logs glm53-dflash2-dcp4 2>&1 | grep -E \
  'B12X PCIe|B12xMxfp8|Aligned DCP-replicated|KV cache group CP geometry|GPU KV cache size|Graph capturing finished|Application startup complete'

curl -fsS http://127.0.0.1:5001/v1/models

curl -fsS http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":64}'
```

Expected startup evidence includes:

```text
Using B12X PCIe all-reduce
Using 'B12X' NvFp4 MoE backend
Using B12xMxfp8LinearKernel for MXFP8 GEMM
Using B12xMxfp8LinearKernel for the fused DFlash context K/V projection
Aligned DCP-replicated sliding-window cache blocks to the sharded target's 2304-token physical block
effective_group_cp=(4, 4, 4, 4, 4, 1)
block_sizes=(2304, 2304, 2304, 2304, 2304, 2304)
```

## Qualification evidence

All measurements used physical GPUs 4, 5, 6, and 7 on the qualification host,
TP4, DCP4, target FP8 KV, automatic BF16 draft KV, seven speculative tokens,
and B12X PCIe all-reduce.

### Correctness

- 113 focused tests passed for the composed DFlash DCP and full compressed-KV
  source tree.
- 96 KV-cache utility and DFlash DCP tests passed after page alignment.
- A 25,210-token prompt followed by 3,072 generated tokens reproduced warmed
  output SHA-256
  `acf8a33a8123fa87a857b83dcb089deb45f11d69af24010765ed183a92758637`
  under both 1152- and 2304-token draft pages.
- The first request after an empty Just-in-Time compile (JIT) cache did not reproduce the long-output
  hash. Cross-cold-start deterministic identity is unsupported; warmed-path
  parity is qualified.

### Cache capacity

At an explicit 12 GiB KV budget, both geometries allocated 906 physical blocks:

| Replicated draft block | Group-aware engine KV tokens | Change |
|---:|---:|---:|
| 1152-token control | 5,688,681 | baseline |
| 2304-token planner alignment | 5,974,904 | +286,223 (+5.03%) |

A production-style start at `GPU_MEMORY_UTILIZATION=0.95` reported 15.87 GiB
available for KV and 7,900,591 group-aware engine KV tokens. That is 7.53
concurrent requests at the configured 1,048,576-token maximum length.

### Decode and prefill

Raw speculative throughput changes with accepted draft length. Engine steps
per second isolate target-forward and scheduler speed from that
prompt-dependent acceptance. CC1 means concurrency one; CC16 means concurrency
16.

| Draft page geometry | CC1 output | CC1 engine rate | CC16 output | CC16 engine rate |
|---|---:|---:|---:|---:|
| 1152-token control, three-run mean | 127.9 tok/s | 62.041 steps/s | 802.4 tok/s | 330.026 steps/s |
| 2304-token alignment, three-run mean | 139.7 tok/s | 62.055 steps/s | 783.0 tok/s | 328.749 steps/s |
| Engine-rate change | — | +0.02% | — | -0.39% |

The aligned runtime measured 10,637, 10,100, and 10,708 prompt tok/s for a
32,321-token standalone prefill, a mean of 10,482 prompt tok/s. The isolated
full compressed-KV target qualification measured 11,929 prompt tok/s versus
8,876 prompt tok/s for ordinary DCP at 32k, a 34.4% improvement under its
matched target-only A/B configuration.

## Limitations

- The source-change pull requests listed in the source contract remain open.
  Build from the pinned image digest until the source stack is reviewed and
  merged.
- Prefill is eager because the GLM Gated Delta Network backend does not support
  full prefill graph capture. Target and draft decode remain graph-captured.
- The mixed-cache Prometheus token estimate is not a group-aware capacity
  oracle. Use the engine's `GPU KV cache size` line or a fixed
  `--kv-cache-memory` A/B.
- Output tok/s alone is not a reliable DFlash regression signal. Record
  accepted length and engine steps per second with every decode comparison.
