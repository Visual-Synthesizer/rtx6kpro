# DeepSeek-V4-Flash-0731 on Infernal Invocation r18

**Status: qualified.** This page specifies TP2/DCP1 target-only and fixed
probabilistic DSpark K5 serving for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell GPUs. The
qualified profile uses FULL CUDA graphs for target decode, DSpark proposal,
and DFlash context-KV execution. Structured-output and persistent-KV gates
were repeated against the immutable image identified below.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18` |
| Registry digest | `sha256:414ec7d0d28358cfd8af0697f330f5c8acbb80e4dc4e5ba69c9fd5b5855ea804` |
| Image ID | `sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM integration tree | `f0fa1cefc1865d316c2478525f550e7646addc40` |
| B12X integration tree | `75787c7a7431b3bea414d2ebf5f2b8671b23eb33` |
| LMCache integration tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Runtime receipt | [`infernal-invocation-r18-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r18-remote-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, LMCache 0.5.2+glm52dcp.5, XGrammar 0.2.5, InstantTensor 0.1.9 |

## Start The Server

Download the committed Compose profile and start fixed probabilistic K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-infernal-invocation-cu133-r18.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r18.yml up -d
```

The profile defaults to TP2/DCP1, B12X W4A8, FP8 compressed MLA KV,
InstantTensor `BUFFERED`, prefix caching, fixed probabilistic DSpark K5, and a
release-scoped JIT cache. Native vLLM KV offload and LMCache are disabled
unless explicitly selected.

Use this scheduler envelope for a 1,048,576-token model limit:

```bash
MAX_MODEL_LEN=1048576 \
MAX_NUM_SEQS=8 \
MAX_NUM_BATCHED_TOKENS=4096 \
GRAPH=auto \
GPU_MEMORY_UTILIZATION=0.975 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r18.yml up -d
```

`GRAPH=auto` derives the graph cap from scheduler-reachable verifier rows. A
fixed K5 profile with eight request slots requires
`8 * (1 + 5) = 48` all-decode graph rows.

## Serving Contract

| Component | Behavior |
|---|---|
| Target and draft quantization | `deepseek_v4_fp8` |
| Attention | B12X compressed MLA |
| MoE and linear layers | B12X W4A8 |
| Inactive MoE routes | Negative and out-of-range routes are masked before expert-table access |
| KV cache | FP8 compressed MLA plus FP32 sliding-window compressor state |
| Speculative decoding | Fixed probabilistic DSpark K5 |
| Structured output | XGrammar 0.2.5 with speculative-prefix validation before scheduler commit |
| Target and verifier decode | FULL CUDA graphs for captured all-decode rows |
| DSpark proposal | FULL CUDA graphs for captured verifier rows |
| Context-KV decode | FULL DFlash CUDA graphs |
| Prefill | PIECEWISE or uncaptured model path |
| Model loading | InstantTensor `BUFFERED` |

The runtime sizes C128A metadata from physical graph capacity, restricts
sparse top-k processing to active rows, updates sparse metadata on the GPU,
and resets MRV2 logits-processing state for each request. Failed external-KV
loads restore heterogeneous cache-group block tables independently.

Target-only, fixed K7, and confidence-controlled K7 are implemented but are
outside the physical-GPU receipt. Select target-only serving with
`MODE=dspark-mtp0`. Fixed and confidence-controlled K7 use
`DSPARK_TOKENS=7` with `DSPARK_DEPTH_MODE=fixed` or `dynamic`.

## Native KV Offload

Native vLLM offload can use a pinned CPU tier and a persistent filesystem
tier:

```bash
KV_OFFLOADING_SIZE=40 \
NATIVE_L2_GB=512 \
NATIVE_L2_PATH=/cache/native-kv/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r18.yml up -d
```

The CPU tier must cover the intended transfer concurrency. The filesystem
path must be persistent and unique per checkpoint revision and cache geometry.
Native offload and LMCache must not be enabled together because they have
independent ownership and transfer protocols.

The restart gate used a 2 GiB CPU tier and a 64 GiB filesystem tier. It
published 695 complete objects, restarted the complete vLLM process, found all
695 objects, read 607,357,440 bytes from the filesystem tier, and reproduced
the exact concurrent outputs `17` and `29`. The store contained
1,394,892,800 bytes and no temporary publication files after replay.

## LMCache

Select LMCache independently:

```bash
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=512 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r18.yml up -d
```

Every service port requires a distinct LMCache endpoint and L2 path. The
restart gate used a 2 GiB L1 tier and a 64 GiB filesystem tier. A 24,349-token
request stored 94 chunks. After a complete LMCache and vLLM process restart,
the first request restored all 94 chunks and reported 24,064 cached tokens.
The completion SHA-256 remained
`9d3dcd6699ae22a7d31ca493d8dbee975121aad91e8db31d7674a950e1beba37`.

## Qualification Evidence

Two direct-root-port RTX PRO 6000 Blackwell GPUs ran TP2/DCP1 with
`MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=4096`, graph cap 192, and
`MAX_MODEL_LEN=262144`.

| Gate | Result |
|---|---|
| Docker and source suites | Passed; 17 focused Python tests passed and `pytest 8.4.1` remains installed |
| FULL CUDA graphs | Target, DSpark, and DFlash context-KV capture passed |
| C1 decode | 164.46 aggregate tok/s; 165.13 active-user tok/s; 64.40 target steps/s |
| Infernal Invocation r16 control | 172.93 aggregate tok/s; 64.03 target steps/s |
| Strict tool modes | Required, named, and automatic tool choice passed in buffered and streaming modes |
| Strict tool soak | 160/160 valid at concurrency 8; 22,908 completion tokens; 0 failures |
| Strict tool latency | 1.301 s median, 3.158 s p95, 4.393 s maximum |
| Native filesystem restart | 695/695 objects found; exact outputs preserved; no temporary files |
| LMCache process restart | 94/94 chunks and 24,064 prompt tokens restored; exact completion preserved |
| Runtime health | Server healthy after every gate |

The r18 target-step rate is 0.58 percent above the r16 control. Aggregate
emitted-token throughput differs because the probabilistic draft accepted
2.55 tokens per target step in r18 and 2.70 in the r16 control. Engine steps
per second are therefore the acceptance-normalized runtime comparison.

## Source Contract

The image contains no private source patch. The exact vLLM and B12X merge
queues are maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).
The official `vllm-project/vllm/main` branch was audited at
`8f4a7f45c53ab52b17023d3ca804e477daa36a23`; commits outside the Infernal
Invocation source line changed only ROCm CI files and did not supply a DS4 or
GLM runtime dependency.

## Qualification Limits

- **Qualified:** TP2/DCP1 fixed probabilistic DSpark K5, B12X W4A8, FP8
  compressed MLA KV, FULL target/draft/context-KV graphs, strict tools at
  concurrency 8, native filesystem replay, and LMCache disk replay across
  complete process restart.
- **Implemented:** target-only serving, fixed K7, confidence-controlled K7,
  LMCache, native KV offload, and a 1,048,576-token launcher envelope.
- **Unsupported by this receipt:** DCP greater than one, TP other than two,
  K7 runtime performance, full-context 1,048,576-token execution, and model
  quality evaluation.
- Performance measurements use direct-root-port GPUs and are not directly
  comparable with switched-PCIe results.
- Persist the release-scoped `/cache` mount because the first request for an
  uncovered shape may compile B12X kernels.
