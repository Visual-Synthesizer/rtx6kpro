# Glossary And Acronym Guide

This wiki uses a lot of model-serving shorthand. When writing or editing pages,
expand important acronyms on first use, then use the short form afterwards.

Example:

```text
Decode Context Parallelism (DCP) is enabled with `--decode-context-parallel-size`.
DCP4 means four decode-context-parallel ranks.
```

Do not expand acronyms inside command-line flags, Docker tags, environment
variables, file paths, JSON, or raw logs.

## Core Runtime Terms

| Acronym | Meaning | Notes |
|---|---|---|
| AOT | Ahead-of-Time compile | Used by vLLM/Inductor/TileLang compile paths. |
| B12X | B12X kernel/backend stack | Local optimized backend stack used for sparse MLA, MoE, dense GEMM, and PCIe all-reduce. |
| CC | Concurrency | Usually request concurrency in `llm_decode_bench.py` tables. |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU programming/runtime stack. |
| CUDA Graph | CUDA Graph capture/replay | vLLM uses graph capture sizes to reduce per-token launch overhead. |
| DCP | Decode Context Parallelism | Splits decode context work across ranks. |
| DFlash | Diffusion/Block-Diffusion Flash draft path | Speculative decoding style used by Kimi/MiMo pages. |
| DS4 | DeepSeek-V4-Flash | Local shorthand for `deepseek-ai/DeepSeek-V4-Flash`. |
| DSpark | DeepSeek DSpark speculative decoding | Native DeepSeek-V4-Flash DSpark checkpoint/method. |
| E2E | End-to-End | Full server/client throughput, not isolated kernel timing. |
| JIT | Just-in-Time compile | Runtime compilation, common for Triton/TileLang/CuTe kernels. |
| MBT | Max Batched Tokens | Usually `--max-num-batched-tokens`. |
| MLA | Multi-head Latent Attention | Attention architecture used by DeepSeek/GLM/Kimi-family models. |
| MoE | Mixture of Experts | Routed expert MLP layers. |
| MTP | Multi-Token Prediction | Speculative decoding using model MTP heads. |
| PR | Pull Request | GitHub change request. |
| TP | Tensor Parallelism | Splits model tensors across GPUs. |
| TTFT | Time To First Token | Latency until first generated token. |
| TPOT | Time Per Output Token | Decode latency metric. |
| VRAM | Video RAM | GPU memory. |

## Numeric And Quantization Terms

| Acronym | Meaning | Notes |
|---|---|---|
| A8 | 8-bit activation path | In these pages, usually FP8/MXFP8 activation serving for FP4 MoE kernels. |
| A16 | 16-bit activation path | Usually BF16 activation serving for FP4 MoE kernels. |
| BF16 | Brain Floating Point 16-bit | 16-bit floating-point format. |
| FP4 | 4-bit floating-point | General 4-bit floating-point family. |
| FP8 | 8-bit floating-point | General 8-bit floating-point family. |
| INT4 | 4-bit integer quantization | Weight quantization format. |
| INT8 | 8-bit integer quantization | Weight or activation quantization format. |
| KLD | Kullback-Leibler Divergence | Used here to compare BF16 reference logits to quantized/runtime logits. |
| MXFP4 | Microscaling FP4 | FP4 format with microscaling. |
| MXFP8 | Microscaling FP8 | FP8 format with microscaling. |
| NVFP4 | NVIDIA FP4 | NVIDIA FP4 quantization format. |
| QAT | Quantization-Aware Training | Training/fine-tuning with quantization in the loop. |
| W4A8 | 4-bit weights, 8-bit activations | Common MoE kernel mode. |
| W4A16 | 4-bit weights, 16-bit activations | Common MoE kernel mode. |

## Communication And Kernel Terms

| Acronym | Meaning | Notes |
|---|---|---|
| A2A | All-to-All | Collective communication pattern. |
| AG | All-Gather | Collective communication primitive. |
| AG+RS | All-Gather plus Reduce-Scatter | Hybrid DCP communication policy in some GLM pages. |
| CuTe | CUDA Templates | NVIDIA/CUTLASS template library used by some kernels. |
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | NVIDIA CUDA kernel template library. |
| GEMM | General Matrix Multiply | Core dense matrix multiplication operation. |
| NCCL | NVIDIA Collective Communications Library | GPU collective communication library. |
| P2P | Peer-to-Peer | Direct GPU-to-GPU communication path, commonly controlled through NCCL P2P settings. |
| PCIe | Peripheral Component Interconnect Express | Host/GPU interconnect used on RTX PRO 6000 systems. |
| RDMA | Remote Direct Memory Access | Direct memory-transfer mechanism; appears in cache/transfer notes. |
| RMS | Root Mean Square | Often appears in RMSNorm / fused RMS all-reduce contexts. |
| RS | Reduce-Scatter | Collective communication primitive. |
| SM100 | Streaming Multiprocessor 100 | NVIDIA Blackwell datacenter architecture class. |
| SM120 | Streaming Multiprocessor 120 | RTX PRO 6000 Blackwell / GB202 architecture class. |
| SWA | Sliding Window Attention | Attention mode for long-context models and draft paths. |
| TMA | Tensor Memory Accelerator | NVIDIA GPU memory movement feature. |

## Project And Tool Names

| Acronym | Meaning | Notes |
|---|---|---|
| HF | Hugging Face | Model hosting and download/cache paths. |
| LMCache | LLM KV-cache storage/reuse project | Appears in external cache experiments. |
| SGLang | Structured Generation Language runtime | Alternative inference runtime. |
| vLLM | vLLM inference engine | Primary serving runtime in current pages. |
| xgrammar | Structured-output grammar backend | Used by vLLM for constrained/tool-call output. |

## Local Test Shorthand

| Acronym | Meaning | Notes |
|---|---|---|
| CJK | Chinese/Japanese/Korean watchdog marker | Local historical shorthand in `/mnt/test.py`; often used as a corruption/repetition smoke signal. |
| cc1 / cc64 | Concurrency 1 / concurrency 64 | Shorthand used in benchmark tables and discussion. |
| ctx0 / 100k | Context length 0 / 100k tokens | Benchmark prompt-context shorthand. |
