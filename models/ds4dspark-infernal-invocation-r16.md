# DeepSeek-V4-Flash-0731 on Infernal Invocation r16

**Status: qualified.** This page specifies target-only and fixed probabilistic
DSpark K5 serving for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000
Blackwell GPUs. The runtime masks inactive MoE routes in every qualified B12X
decode kernel and validates speculative structured output before scheduler
state is committed.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm5beffc4-b12xa4a0bc8-fi1ac6942-cu133-torch213-20260817-r16` |
| Registry digest | `sha256:ff9d4f2402ed88b1ae7ca3a6886c80a64d72993f1a593380c8cb6f193437567d` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM integration tree | `5beffc48f7cd9d4ade076e4b6d1f117ac8e79d4a` |
| B12X integration tree | `a4a0bc8a8f5e56dbef85f9b46b0d74f6e8edb491` |
| LMCache integration tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Runtime receipt | [`infernal-invocation-r16-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16-remote-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, LMCache 0.5.2+glm52dcp.5, XGrammar 0.2.5 |

## Start The Server

Download the committed Compose profile and start TP2/DCP1 fixed probabilistic
K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-infernal-invocation-cu133-r16.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r16.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, prefix caching, fixed probabilistic DSpark K5, and release-scoped
JIT storage. Native vLLM KV offload and LMCache are disabled unless explicitly
selected.

The 1,048,576-token launch profile is:

```bash
MAX_MODEL_LEN=1048576 \
MAX_NUM_SEQS=8 \
MAX_NUM_BATCHED_TOKENS=4096 \
GRAPH=auto \
GPU_MEMORY_UTILIZATION=0.975 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r16.yml up -d
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
| Structured output | XGrammar 0.2.5 with speculative-prefix validation before commit |
| Target and verifier decode | FULL CUDA graphs for captured all-decode rows |
| DSpark proposal | FULL CUDA graphs for captured verifier rows |
| Context-KV decode | FULL DFlash CUDA graphs |
| Prefill | PIECEWISE or uncaptured model path |
| Model loading | InstantTensor `BUFFERED` |

Target-only, fixed K7, and confidence-controlled K7 are implemented but are
outside the physical-GPU receipt. Select them with `MODE=dspark-mtp0`,
`DSPARK_TOKENS=7`, and `DSPARK_DEPTH_MODE=dynamic`, respectively.

## Native KV Offload

Native vLLM offload can use a pinned CPU tier and a persistent filesystem tier:

```bash
KV_OFFLOADING_SIZE=40 \
NATIVE_L2_GB=512 \
NATIVE_L2_PATH=/cache/native-kv/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r16.yml up -d
```

The CPU tier must be large enough for the intended transfer concurrency. The
filesystem path must be persistent and unique per model revision and cache
geometry. Native offload and LMCache must not be enabled together because they
have independent ownership and transfer protocols.

The native restart gate used a 2 GiB CPU tier and a 64 GiB filesystem tier. It
published 695 complete objects, restarted the engine, found all 695 objects,
read 1,394,892,800 bytes from the filesystem tier, and reproduced exact
outputs. No temporary objects remained after replay.

## LMCache

LMCache is selected independently:

```bash
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=512 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r16.yml up -d
```

`LMCACHE_TRANSFER_MODE` accepts `auto`, `lmcache_driven`, or `engine_driven`.
Use the default `auto` unless a deployment has a separately qualified transfer
policy. Every service port requires a distinct LMCache endpoint and L2 path.

The LMCache restart gate used a 2 GiB L1 tier and a 64 GiB disk tier. A
24,349-token request populated a 4,360,044,928-byte L2 store. After explicit L1
eviction and local-prefix reset, LMCache restored 24,064 prefix tokens. A
complete vLLM and LMCache process restart restored the same 24,064 tokens on
the first request and again after another L1 eviction. Both TP ranks loaded
the same chunk count.

## Qualification Evidence

Two direct-root-port RTX PRO 6000 Blackwell GPUs ran TP2/DCP1 with
`MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=4096`, graph cap 192, and
`MAX_MODEL_LEN=262144`.

| Gate | Result |
|---|---|
| Docker release suites | Passed |
| FULL CUDA graphs | Target, DSpark, and DFlash context-KV capture passed |
| C1 decode | 172.93 aggregate tok/s; 178.49 active per-user tok/s; 64.03 target steps/s |
| Gilded Gnosis r27 control | 173.18 aggregate tok/s; 62.96 target steps/s |
| Strict tool soak | 160/160 valid at concurrency 8; 23,285 completion tokens; 0 failures |
| Strict tool latency | 1.150 s median, 2.901 s p95, 3.669 s maximum |
| Native filesystem restart | 695/695 objects found; 1,394,892,800 bytes read; exact outputs preserved |
| LMCache disk restart | 24,064/24,349 prompt tokens restored after L1 eviction and process restart |
| Runtime health | No engine errors and server healthy after every gate |

The r16 aggregate C1 rate differed from the Gilded Gnosis r27 control by
-0.14 percent. Acceptance-normalized target execution improved by 1.70 percent.
Aggregate DSpark throughput remains trajectory-dependent, so target steps/s is
the runtime-speed control for unmatched generated trajectories.

## Qualification Limits

- **Qualified:** TP2/DCP1 fixed probabilistic DSpark K5, B12X W4A8, FP8
  compressed MLA KV, FULL target/draft/context-KV graphs, strict tools at
  concurrency 8, native filesystem replay, and LMCache disk replay across
  process restart.
- **Implemented:** target-only serving, fixed and dynamic K7, LMCache, and the
  1,048,576-token launcher envelope.
- **Unsupported by this receipt:** DCP greater than one, TP other than two,
  full-context 1,048,576-token execution, and model-quality evaluation.
- Performance measurements use direct-root-port GPUs and are not directly
  comparable with switched-PCIe results.
- Persist the release-scoped `/cache` mount because the first context-dependent
  request may compile B12X kernels.
