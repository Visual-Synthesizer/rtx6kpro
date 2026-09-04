# DeepSeek-V4-Flash Jovian Judgement r4

This DeepSeek-V4-Flash (DS4) serving profile combines the vLLM inference engine
(vLLM), B12X kernel/backend stack (B12X), DeepSeek DSpark speculative decoding
(DSpark), and LLM KV-cache storage/reuse project (LMCache) on NVIDIA Streaming
Multiprocessor 120 (SM120) GPUs. It uses Tensor Parallelism (TP), Decode
Context Parallelism (DCP), Multi-head Latent Attention (MLA), 8-bit
floating-point (FP8), 4-bit weights, 8-bit activations (W4A8), Compute
Unified Device Architecture (CUDA), NVIDIA Collective Communications Library
(NCCL), CUDA Templates for Linear Algebra Subroutines (CUTLASS), and Peripheral
Component Interconnect Express (PCIe).

**Status: qualified for TP2/DCP1 fixed probabilistic DSpark serving.** The text
checkpoint uses K5. The Vision checkpoint uses K3 and supports text and
multi-image requests. GPU KV storage is the default; LMCache host storage is
opt-in. Native vLLM filesystem KV offload is unsupported by this serving
specification.

## TL;DR

Start `deepseek-ai/DeepSeek-V4-Flash-0731` with fixed DSpark K5 on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r4.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r4.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r4.yml up -d
```

Start `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` with fixed DSpark K3:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r4.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r4.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r4.yml up -d
```

Both profiles use GPU KV unless `LMCACHE_MODE=ram` or `LMCACHE_MODE=disk` is
set explicitly.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllm075caea-b12xd0c4487-fi803c466-cu133-torch213-20260904-r4` |
| Registry digest | `sha256:74f74eff0fe36b74334a5aaead7fa65c5a14fc4bd415bd26877415d5277e970c` |
| Image ID | `sha256:c2d49a67c624082a6ed94dba2bf2c37d8596976f8d9e6602ffd8635592820fb1` |
| Image size | 34,614,573,173 bytes |
| Docker source used by the image | `local-inference-lab/blackwell-llm-docker@572c13490b6efbb20f5bb2563309095eacf91810` |
| Runtime receipts | [text direct-LMCache 1M](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r4-direct-lmcache-1m.json), [Vision text/image smoke](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r4-vision-smoke.json) |
| Source merge contract | [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| vLLM base | `dev/jovian-judgement@a50ebee1d2460d22386b54e79f46236376e2b486` |
| vLLM integration tree | `075caea7bf032731baeca337a9ee882ab8967c20` |
| B12X base | `master@9ae41c5cb9935d740456479954b0089f80bd2ef2` |
| B12X integration tree | `d0c4487adefa7dacd661d194fd4234ef61830796` |
| LMCache integration tree | `eb4c227f68a4e1c45d6b8edf6b4934e18f6d1f8b` |
| FlashInfer tree | `803c4664f4771ddc418f20a57f752469a237a825` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, B12X 1.3.0, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, XGrammar 0.2.5, LMCache 0.5.2+glm52dcp.5 |

Image labels record every source base, pull-request head, integration tree,
generated patch digest, and dependency revision. The runtime contains compiled
installed packages and no source mount or source overlay.

## Text Serving Contract

| Setting | Value |
|---|---:|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| `MODE` | `dspark` |
| `DSPARK_TOKENS` | `5` |
| Draft sampling | fixed probabilistic |
| `BACKEND` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` |
| `MAX_NUM_SEQS` | `8` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| `MAX_MODEL_LEN` | `1048576` |
| GPU-only memory utilization | `0.975` |
| Direct-LMCache memory utilization | `0.965` |
| CUDA graphs | `FULL_AND_PIECEWISE`, cap 48 |
| KV format | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` |

Select target-only execution with `MODE=dspark-mtp0`. Fixed K5 remains the
qualified speculative profile for the 0731 checkpoint.

## Vision Serving Contract

| Setting | Value |
|---|---:|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| `MODE` | `dspark` |
| `DSPARK_TOKENS` | `3` |
| Draft sampling | fixed probabilistic |
| `BACKEND` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` |
| `MAX_NUM_SEQS` | `4` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| GPU-only model length / memory utilization | `1048576` / `0.975` |
| Direct-LMCache model length / memory utilization | `900000` / `0.951` |
| CUDA graphs | `FULL_AND_PIECEWISE`, cap 16 |
| KV format | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` |

The Vision checkpoint contains three draft layers, so K3 is its deepest
supported DSpark mode. The target consumes and verifies image embeddings. The
drafter proposes from text-only inputs because it has no external multimodal
embedding interface; this can reduce draft acceptance without changing target
verification.

## LMCache Memory Contract

Enable host RAM storage for the text checkpoint:

```bash
LMCACHE_MODE=ram \
LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r4.yml up -d
```

Enable RAM plus filesystem persistence:

```bash
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=256 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r4.yml up -d
```

The direct-LMCache server initializes CUDA before it accepts cache
registrations. An isolated server used 550 MiB on GPU 0 at startup. After TP2
KV registration it used 708 MiB on each GPU. The increase covers CUDA IPC
mappings, streams, block-ID storage, and transfer staging resources.
`LMCACHE_L1_GB` controls host RAM and does not reduce this GPU footprint.

For text TP2 fixed K5 at a 1,048,576-token model length, the launcher defaults
to `GPU_MEMORY_UTILIZATION=0.965` when direct LMCache is active. Values above
0.965 are rejected before model loading unless
`LMCACHE_ALLOW_UNQUALIFIED_MEMORY_PROFILE=1` is set.

For Vision direct LMCache, the launcher defaults to a 900,000-token model
length and memory utilization 0.951. Engine-driven LMCache does not establish a
standalone CUDA context and retains the GPU-only profile.

Each serving instance requires a distinct LMCache HTTP port and filesystem
path. Do not enable native vLLM filesystem KV offload together with LMCache.

## Text Direct-LMCache Qualification

Two RTX PRO 6000 Blackwell GPUs connected through a PCIe switch ran the exact
registry artifact with TP2/DCP1, fixed probabilistic K5, MNS8, MNB4096, and
memory utilization 0.965.

| Gate | Result |
|---|---|
| GPU KV capacity | 1,181,020 tokens |
| Uncached request | 1,000,000 prompt tokens; HTTP 200; 176.56 seconds |
| Cache replay | 999,424 prompt tokens restored; HTTP 200; 1.42 seconds |
| Output parity | Same generated token on miss and replay |
| Minimum free memory | 425 MiB on GPU 0; 431 MiB on GPU 1 |
| Runtime health | vLLM and LMCache healthy after both requests |

The rejected 0.975 profile created a 1,338,517-token KV pool but retained only
47-51 MiB of free memory. Its first long request required a 128 MiB temporary
allocation and failed. The 0.965 contract preserves enough runtime reserve
while retaining more than one million GPU KV tokens.

## Vision Qualification

The exact r4 artifact loaded target and K3 draft weights, captured PIECEWISE
target, FULL target, and FULL DSpark graphs, and created a 1,291,930-token GPU
KV pool.

| Gate | Result |
|---|---|
| Text request | HTTP 200; exact response `r4 vision text passed` |
| Two-image request | HTTP 200; identified carrots with edible roots and corn with edible kernels |
| Service health | API remained healthy with no runtime error |

The Vision memory envelope was qualified with registry artifact
`voipmonitor/vllm:jovian-judgement-vllmd6f9e77-b12x283a63e-fi803c466-cu133-torch213-20260904-r3`.
That artifact used vLLM integration tree `d6f9e777` and B12X integration tree
`283a63ee`.

| Memory gate | Result |
|---|---|
| GPU-only mixed pressure | An uncached 810k-token text request completed while ten 2048x2048 images were processed; minimum free memory 325/327 MiB |
| Direct-LMCache profile | 907,467 GPU KV tokens at the 900,000-token contract |
| Direct-LMCache replay | 47,872 of 48,092 prompt tokens restored; 10.73-second miss and 0.37-second replay |
| Direct-LMCache mixed pressure | An uncached 808,598-token text request plus ten 2048x2048 images returned HTTP 200; minimum free memory 2,239/2,249 MiB |

The r4 registry artifact passed the graph-capture and text/two-image smoke
listed above. A complete repeat of the mixed long-context Vision pressure
workload was not performed on the r4 artifact.

## Source Composition

**Implemented, review pending:** the vLLM tree applies
[vLLM #628](https://github.com/local-inference-lab/vllm/pull/628),
[vLLM #630](https://github.com/local-inference-lab/vllm/pull/630), and
[vLLM #634](https://github.com/local-inference-lab/vllm/pull/634) to
`dev/jovian-judgement`. These pull requests register scheduler-reachable
B12X graph rows, make explicit NCCL selection authoritative, provide the
Vision architecture and incremental loader, and enforce the DS4 direct-LMCache
memory contracts.

**Implemented, review pending:** the B12X tree applies
[B12X #246](https://github.com/local-inference-lab/b12x/pull/246),
[B12X #302](https://github.com/local-inference-lab/b12x/pull/302),
[B12X #301](https://github.com/local-inference-lab/b12x/pull/301), and
[B12X #306](https://github.com/local-inference-lab/b12x/pull/306) to
`master`. These pull requests provide generation-safe TP2 graph peer-push, a
valid W4A8 profiling oracle, sparse top-k-512 dual-cache prefill, and the Vision
checkpoint's `rms_norm_eps=1e-20` specialization.

**Implemented, review pending:**
[LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44) transfers
interleaved 64-head cache pages with their physical dimension-zero stride.

**Implemented:** FlashInfer tree
`803c4664f4771ddc418f20a57f752469a237a825` supplies SM120 sparse-MLA
top-k-512 fallback support.

## Tokenizer Diagnostic

The string `)Skip` is vocabulary token 83480 in the Vision checkpoint. It is an
ordinary model token, not a special token, parser marker, or runtime-injected
value. A report observed 12 occurrences in 220,000 generated tokens, but no
request-level reproducer is available. Diagnosing a recurrence requires the
exact request JSON, sampling parameters, and surrounding generated token IDs.

## Qualification Limits

- **Qualified:** TP2/DCP1 text fixed K5 and Vision fixed K3, B12X W4A8, FP8
  compressed MLA KV, target and draft CUDA graphs, InstantTensor loading,
  one-million-token text direct-LMCache miss/replay, and Vision text/image
  inference.
- **Implemented:** target-only execution, LMCache RAM storage, LMCache
  filesystem persistence, and engine-driven LMCache transfer.
- **Unsupported:** native vLLM filesystem KV offload, Vision speculative depth
  above K3, and text fixed depths above the checkpoint-supported contract.
- **Not qualified by these receipts:** TP other than two, DCP greater than one,
  task-level model quality, memory envelopes for GPUs with less than 96 GiB,
  or the mixed long-context Vision pressure workload on the exact r4 artifact.
- Reuse the release-scoped `/cache` mount. Uncovered kernel shapes can compile
  during the first request.
