# Eldritch Enlightenment vLLM Docker

This page documents the reproducible Docker build used by the GLM-5.2 v13,
DS4 Flash v6, Kimi 2.7, and MiMo validation work.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629
voipmonitor/vllm@sha256:2ddd49e4ee162ea79a0f2e537aec3e18918bef168e3d684cfa5332151c7fbe2e
```

The image is a clean Docker build. It does not require runtime bind-mount
overlays for vLLM or B12X sources.

## Build Source

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 60fa373
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629 \
BUILD_BASE_IMAGE=0 \
./build-eldritch-enlightenment-head66-pr20-cu132.sh
```

The build helper is:

```text
build-eldritch-enlightenment-head66-pr20-cu132.sh
```

It is a clean source build from pinned vLLM, B12X, FlashInfer, and DeepGEMM
commits. It inherits the current CUDA 13.2 system/build bases and does not use
a runtime source overlay or `VLLM_PATCH_URL`.

## Component Pins

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/eldritch-head66-b12xmla-20260629` |
| vLLM commit | `8722ac7f8427919ed67bfe9c5e47b3cc30dfbf2e` |
| B12X repo | `https://github.com/lukealonso/b12x.git` |
| B12X branch | `master` |
| B12X commit | `15cd38ce3f10ee5cb7db1179cbc7c88fd15e37b7` |
| B12X merged PRs needed here | `#14`, `#16`, `#17`, and `#20` are in this `master` commit |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| CUDA / cuBLAS | CUDA `13.2.1`; `libcublas13-cuda-13=13.4.1.2-1`; the base image also carries legacy `libcublas-13-2=13.4.0.1-1` |
| cuDNN / NCCL | cuDNN `9.22.0.52-1`, local NCCL `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| FlashInfer cubin wheel | disabled in this build (`FLASHINFER_BUILD_CUBIN=0`) |

## vLLM Patch Stack

The release branch is based on `dev/eldritch-enlightenment` and includes the
validated follow-up fixes below. The final row is PR #64, the only extra vLLM
patch added for the 2026-06-29 head66 image:

| Commit | Purpose |
|---|---|
| `1e8d565` | Avoid sparse-indexer host sync on the fast path. |
| `5d35206` | Pass K/V strides and scales to Triton MLA, needed for Kimi-style page layouts. |
| `0ec1381` | Keep DFlash target `lm_head` sharing, needed by MiMo/Kimi DFlash runs. |
| `fcc6141` | Make speculative warmup prompts DCP shard-safe; fixes DCP4 no-MTP and DCP8 MTP3 graph warmup. |
| `dc5bb1` | Include DCP communicator in CUDA graph capture context; fixes Kimi DCP4+DFlash illegal memory access under graph capture. |
| `9c9d23e` | Make DFlash DCP KV metadata prefix-safe. |
| `a81d072` | Preserve DFlash prefix cache under DCP by aligning effective target/draft prefix boundaries. |
| `bfaa36b` | Keep Eagle3 draft DCP opt-in so Kimi DCP4+Eagle3 does not inherit an invalid target DCP layout. |
| `905d6a5` | Shard native MTP draft under DCP by default; fixes GLM/DS native MTP `topk_scores_buffer` startup failures without requiring env overrides. |
| `67e95e7` | Integrate the GLM sparse-indexer fused prefill kernel cherry-picked from upstream vLLM PR #46862. |
| `56fb5d8` | Add DCP decode/LSE support to `FLASHINFER_MLA_SPARSE_SM120` and fix mixed prefill/decode warmup for DCP. |
| `8722ac7` | Restore minimal GLM/DSA virtual attention-head padding and pad/slice B12X sparse MLA heads locally for TP6 head66. |

The underlying fullstack branch includes the DS4 Flash SM120/CUTLASS runtime
fixes, GLM DCP global top-k and sharded draft KV, structural tool-call fixes,
Kimi parser/spec-decode fixes, MiMo DFlash hooks, and selected upstream vLLM
performance/bugfix PRs.

## Verification

Inside the image:

```text
torch 2.12.0+cu132
flashinfer 0.6.13+cu132
deep_gemm 2.5.0
vllm 0.11.2.dev279+eldritch.enlightenment.8722ac7.b12x15cd38c.fi25dd814.cu132.20260629
b12x 0.23.0
Rust tool parser present
```

The runtime image config must not contain an empty `NCCL_GRAPH_FILE` or
`NCCL_GRAPH_DUMP_FILE`. Launch scripts should still defensively unset both:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

Validated clean-image smokes:

| Mode | Result |
|---|---|
| GLM-5.2 NVFP4 TP6/DCP6/MTP3 | B12X attention, A16 MoE, correct 78-char top-k pattern, KV cache `1,068,888`, coherent short and 30k Sieve output, `0` CJK, about `79-84 tok/s` short and `75-79 tok/s` at 30k. |
| DeepSeek-V4-Flash TP2/B12X/MTP-off | B12X attention/MoE/linear, KV cache `992,634`, coherent short Sieve output, `0` CJK, about `132-133 tok/s`. |
| DeepSeek-V4-Flash TP2/B12X/MTP-off, MBT8192 | Clean image with B12X PR #20 and no overlays passed the 64k prefill repro: `prompt_tokens=64021`, `completion_tokens=128`, `finish=length`, no CUDA/runtime errors. |
| GLM-5.2 NVFP4 TP8/DCP2/MTP3 defaults | Clean image with no `VLLM_DCP_GLOBAL_TOPK` or `VLLM_DCP_SHARD_DRAFT` env overrides started successfully, KV cache `1,289,728`, coherent `/mnt/test.py -L`, `0` CJK, about `96-98 tok/s` generation-only in smoke. |
| Kimi-K2.7-Code TP8/DCP1/DFlash7 | Target `TRITON_MLA`, draft `TRITON_ATTN`, KV cache `369,429`, coherent short Sieve output, `0` CJK, about `200-310 tok/s` generation-only in debug smoke. |
| MiMo V2.5 Pro FP4-DFlash TP8/DCP1/DFlash7 | `TRITON_ATTN`, `FLASHINFER_CUTLASS_MXFP4_MXFP8` MoE, KV cache `1,363,603`, coherent short Sieve output, `0` CJK, about `250-290 tok/s` generation-only in debug smoke. |
