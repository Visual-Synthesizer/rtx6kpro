# RTX PRO 6000 Blackwell LLM Wiki

Field notes for serving frontier MoE models on NVIDIA RTX PRO 6000 Blackwell
PCIe systems without NVLink. The wiki focuses on reproducible Docker images,
exact vLLM/SGLang launch recipes, B12X/FlashInfer/DeepGEMM backends, DCP,
MTP, DSpark, DFlash, KLD checks, and throughput validation.

> Community workbench for the RTX PRO 6000 / SM120 Discord:
> https://discord.gg/X54jjmcxWJ

## Start Here

| Goal | Read |
|---|---|
| Run the current GLM-5.2 stack | [GLM-5.2 v15](models/glm5.2_v15.md) |
| Run DeepSeek-V4-Flash standard / MTP | [DS4 Flash v6](models/ds4-flash-v6.md); standard MTP rows are also in [DS4 DSpark v9](models/ds4dspark-v9.md) and [v10](models/ds4dspark-v10.md) |
| Run DeepSeek-V4-Flash DSpark | [DS4 DSpark v10](models/ds4dspark-v10.md), with [v9](models/ds4dspark-v9.md) as the full DSpark plus standard-MTP reference |
| Run Kimi-K2.7-Code with DFlash | [Kimi-K2.7-Code v3](models/kimi-k27-code_v3.md) |
| Run Xiaomi MiMo V2.5 Pro FP4-DFlash | [MiMo FP4-DFlash v3](models/xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md) |
| Rebuild the shared Eldritch image | [Eldritch Enlightenment Docker](models/eldritch-enlightenment-docker.md) |
| Debug topology / PCIe bandwidth | [Topology](hardware/topology.md), [PCIe bandwidth](hardware/pcie-bandwidth.md), [GPU configs](hardware/gpu-configs.md) |
| Compare quality and quantization | [GLM-5.2 KLD](benchmarks/glm52-kld-evaluation.md), [KLD evaluation](benchmarks/kld-evaluation.md), [NVFP4 comparison](benchmarks/nvfp4-quantization-comparison.md) |

## Current Model Runbooks

| Model | Current page | Runtime focus | Notes |
|---|---|---|---|
| GLM-5.2 NVFP4 / online MXFP8 | [glm5.2_v15.md](models/glm5.2_v15.md) | vLLM Fathomless + B12X | TP8 DCP1/2/4/8, TP6 notes, MTP0/MTP3, KLD, prefill/decode sweeps. |
| GLM-5.2 FP8 + MXFP4 experts | [glm5.2_mxfp4.md](models/glm5.2_mxfp4.md) | vLLM + B12X MXFP4 expert path | Native MXFP4 routed experts, public checkpoint, A8 serving path. |
| DeepSeek-V4-Flash | [ds4-flash-v6.md](models/ds4-flash-v6.md) | Eldritch image, B12X and Lucifer variants | B12X, FlashInfer/CUTLASS, DeepGEMM default, MTP token sweeps, prefill tables. |
| DeepSeek-V4-Flash standard MTP / DSpark | [ds4dspark-v10.md](models/ds4dspark-v10.md), [ds4dspark-v9.md](models/ds4dspark-v9.md) | Fathomless / Eldritch images | Standard-checkpoint MTP0/MTP2/MTP3 rows plus DSpark checkpoint validation. |
| Kimi-K2.7-Code | [kimi-k27-code_v3.md](models/kimi-k27-code_v3.md) | vLLM V2 + DFlash | DCP4, Kimi parser/tool-call runtime, DFlash7 validation. |
| Xiaomi MiMo V2.5 Pro FP4-DFlash | [xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md](models/xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md) | vLLM V2 + DFlash | FP4-DFlash checkpoint, seq-mask fix, expected fast/slow backend markers. |
| GLM-5.1 | [glm5.1_v10.md](models/glm5.1_v10.md), [glm-5.1-mxfp4.md](models/glm-5.1-mxfp4.md) | Historical vLLM/B12X GLM work | Older DCP, MXFP4, KLD, and checkpoint-conversion notes. |
| Kimi-K2.6 | [kimi-k26-v9.md](models/kimi-k26-v9.md), [kimi-k26.md](models/kimi-k26.md) | Historical Kimi MLA/Eagle/DFlash work | Kept for regression comparisons and parser/spec-decode history. |
| Qwen / MiniMax / older Kimi | [Qwen3.5-397B](models/qwen35-397b.md), [Qwen3.5-27B/122B](models/qwen35-27b.md), [MiniMax M2.5](models/minimax-m25.md), [Kimi-K2.5](models/kimi-k25.md) | Legacy recipes | Useful for topology, quantization, and older engine comparisons. |

Older versioned pages are intentionally kept. Prefer the highest version number
for a model family unless a page explicitly says it is a reduced validation or a
historical reference.

## Docker And Release Lines

| Line | Page | Use |
|---|---|---|
| Fathomless Firmament | [GLM-5.2 v15 image section](models/glm5.2_v15.md#image-and-model), [DS4 DSpark v10 image section](models/ds4dspark-v10.md#image) | Current July 2026 GLM/DS4 validation line. |
| Eldritch Enlightenment | [eldritch-enlightenment-docker.md](models/eldritch-enlightenment-docker.md) | June 2026 shared GLM/DS4/Kimi/MiMo fullstack baseline. |
| General image notes | [Docker Images](optimization/docker-images.md) | Image naming, build conventions, and reusable operational notes. |

Build docs are part of the model pages because each release line pins a different
vLLM branch, B12X commit, FlashInfer commit, DeepGEMM commit, CUDA stack, and
runtime wrapper.

## Benchmarks And Quality

| Area | Page |
|---|---|
| Consolidated throughput | [Benchmark Results](benchmarks/results.md) |
| vLLM vs SGLang throughput | [Inference throughput](benchmarks/inference-throughput/README.md) |
| GLM-5.2 KLD and quant quality | [GLM-5.2 KLD Evaluation](benchmarks/glm52-kld-evaluation.md) |
| General KLD methodology | [KLD Evaluation](benchmarks/kld-evaluation.md) |
| MTP quality checks | [MTP Quality Evaluation](benchmarks/mtp-quality-evaluation.md) |
| NVFP4 quantization comparison | [NVFP4 Quantization Comparison](benchmarks/nvfp4-quantization-comparison.md) |

KLD is a regression and quantization-sanity tool, not a complete quality metric.
Use it together with long-context decode, coding probes, acceptance-rate checks,
and task-level benchmarks.

## Optimization Topics

| Topic | Page |
|---|---|
| PCIe oneshot all-reduce | [pcie-oneshot-allreduce.md](optimization/pcie-oneshot-allreduce.md) |
| NCCL tuning and graph XML issues | [nccl-tuning.md](optimization/nccl-tuning.md) |
| Speculative decoding | [speculative-decoding.md](optimization/speculative-decoding.md) |
| NVFP4 quantization | [nvfp4-quantization.md](optimization/nvfp4-quantization.md) |
| Hybrid NVFP4 assembly | [hybrid-nvfp4-assembly.md](optimization/hybrid-nvfp4-assembly.md) |
| B12X FP8 / DeepGEMM comparison | [b12x-dense-fp8-gemm-vs-deepgemm.md](optimization/b12x-dense-fp8-gemm-vs-deepgemm.md) |
| B12X W4A8 tiny-decode work | [b12x-w4a8mx-tiny-decode-kernel.md](optimization/b12x-w4a8mx-tiny-decode-kernel.md) |
| DSpark upstream consolidation | [dspark-upstream-consolidation.md](optimization/dspark-upstream-consolidation.md) |
| I/O tuning | [io-tuning.md](optimization/io-tuning.md) |

## Hardware And Topology

All modern measurements are on NVIDIA RTX PRO 6000 Blackwell / GB202 / SM120:

- 96 GB GDDR7 per GPU.
- PCIe 5.0 x16, no NVLink.
- 4-GPU, 8-GPU, and 16-GPU PCIe switch systems.
- AMD EPYC Turin/Genoa hosts are the most common community targets.

Key pages:

- [SM120 vs SM100 Architecture](hardware/sm120-vs-sm100-architecture.md)
- [PCIe Topology](hardware/topology.md)
- [PCIe Bandwidth](hardware/pcie-bandwidth.md)
- [GPU Configurations](hardware/gpu-configs.md)
- [ASUS ESC8000A-E13P + Broadcom Switches](hardware/asus-esc8000a-e13p-broadcom-switches.md)
- [ASRockRack + EPYC Turin + 4x c-payne, 16 GPU](hardware/asrockrack-turin-cpayne-16gpu.md)
- [ASRock WRX90 + 4x c-payne, 16 GPU](hardware/wrx90-cpayne-16gpu-4switch.md)
- [Blackwell power limit sweep](hardware/blackwell-power-limit-sweep.md)

## Inference Engines

| Engine | Page | Current role |
|---|---|---|
| vLLM | [vllm.md](inference-engines/vllm.md) | Primary runtime for GLM-5.2, DS4 Flash, Kimi 2.7, MiMo DFlash. |
| FlashInfer | [flashinfer.md](inference-engines/flashinfer.md) | SM120 sparse MLA, CUTLASS MoE, sampler, and kernel integration notes. |
| SGLang | [sglang.md](inference-engines/sglang.md) | Historical and alternate runtime notes, especially for older GLM/MiMo paths. |

## Common Operational Rules

- Do not launch with `NCCL_GRAPH_FILE=` set to an empty string. Unset it if no
  real XML graph file is used.
- Reuse cache directories while debugging; otherwise TileLang/Triton/CuTe
  rebuilds dominate iteration time.
- For quick smoke tests, use small `MAX_NUM_SEQS` and graph caps. For published
  tables, use the graph sizes documented in the model page.
- For DFlash and DSpark, confirm the backend markers and acceptance rates before
  trusting throughput numbers.
- For GLM-5.2, keep the exact `index_topk_pattern` and DCP policy from the
  relevant runbook; a truncated pattern can silently degrade output.

## Troubleshooting

- [Common Issues](troubleshooting/common-issues.md)
- [DS4-Flash empty `content` from unclosed `<think>`](models/ds4f-empty-think/README.md)
- [ASUS ESC8000A-E13P PEX890xx bug report](troubleshooting/asus-esc8000a-e13p-pex890xx-bug-report.md)
- [Daily summaries](daily-summaries/) for chronological context and regression history

## Contributing

Open a PR with exact commands, Docker image tags, model snapshot IDs, GPU layout,
backend choices, and raw benchmark artifacts where possible. For performance
claims, include both the launch config and the client command so results can be
reproduced on another PCIe-only Blackwell host.

Maintained from community Discord experiments through July 2026.
