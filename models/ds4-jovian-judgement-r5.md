# DeepSeek-V4-Flash Jovian Judgement r5

This serving specification covers `deepseek-ai/DeepSeek-V4-Flash-0731` and
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on two NVIDIA SM120 GPUs. The stack
uses vLLM, B12X, DeepSeek DSpark speculative decoding, FP8 compressed MLA KV,
and optional LMCache host storage.

**Status: qualified.** Text serving uses TP2/DCP1 with fixed probabilistic
DSpark K5. Vision serving uses TP2/DCP1 with fixed probabilistic DSpark K3.
GPU KV storage is the default. Enabling LMCache selects asynchronous
engine-driven shared-memory transfer unless the operator requests another
transfer mode explicitly.

## TL;DR

Start the text checkpoint on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r5.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r5.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r5.yml up -d
```

Enable an in-memory external KV tier:

```bash
LMCACHE_MODE=ram \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r5.yml up -d
```

Enable RAM staging plus persistent filesystem storage:

```bash
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=256 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r5.yml up -d
```

Start the Vision checkpoint:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r5.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r5.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r5.yml up -d
```

The Compose profiles reference a prebuilt image and contain no `build`
section.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllm08e1c7d-b12xd0c4487-fi803c466-cu133-torch213-20260904-r5` |
| Registry digest | `sha256:3c3831dfd6f103c8542ab80c7f1dccbb2110f88493cab1ef6d42fcf6c4d710bf` |
| Image ID | `sha256:8d3a16b1973ceb574bb4dd6aab31156b128b941b37fafbcab3a0b6eb6eda9a05` |
| Image size | 34,615,336,283 bytes |
| Docker source used by the image | `local-inference-lab/blackwell-llm-docker@238cda9af370b207d187ac9ccd654185c08f8885` |
| Validation receipt | [engine-driven LMCache qualification](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/jovian-judgement-ds4-r5-engine-driven-lmcache.json) |
| Source merge contract | [rtx6kpro issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| vLLM base | `dev/jovian-judgement@a50ebee1d2460d22386b54e79f46236376e2b486` |
| vLLM integration tree | `08e1c7de14c4754facf53d26f0888f649a478ff4` |
| B12X base | `master@9ae41c5cb9935d740456479954b0089f80bd2ef2` |
| B12X integration tree | `d0c4487adefa7dacd661d194fd4234ef61830796` |
| LMCache base | `release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481` |
| LMCache integration tree | `62144a7f113d45fc27470d665b0b58a9f45679ee` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, B12X 1.3.0, FlashInfer 0.6.18, LMCache 0.5.2+glm52dcp.6, XGrammar 0.2.5 |

Image labels record each source base, pull-request head, integration tree,
generated patch digest, and dependency revision. The runtime contains compiled
installed packages and no source mount.

## Serving Profiles

| Setting | Text | Vision |
|---|---:|---:|
| Checkpoint | `DeepSeek-V4-Flash-0731` | `DeepSeek-V4-Flash-Vision-Exp` |
| DSpark depth | fixed K5 | fixed K3 |
| Draft sampling | probabilistic | probabilistic |
| `BACKEND` | `b12x-a8-dglin` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | `2` / `1` |
| `MAX_NUM_SEQS` | `8` | `4` |
| CUDA graph cap | `48` | `16` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` |
| `MAX_MODEL_LEN` | `1048576` | `1048576` |
| Engine-driven LMCache memory utilization | `0.970` | `0.970` |
| KV format | FP8 compressed MLA | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |

B12X serves compressed sparse attention and W4A8 routed experts. DGLIN serves
FP8 dense projections. Target, DSpark proposal, and DFlash context-KV decode
paths use `FULL_AND_PIECEWISE` CUDA graphs for scheduler-reachable row counts.

## Engine-Driven LMCache

DeepSeek V4 registers eight KV groups with different physical page shapes.
LMCache PR [#44](https://github.com/local-inference-lab/LMCache/pull/44)
preserves each group's layout and gathers group pages on a dedicated CUDA copy
stream. Shared-memory slots are reserved by the standalone cache process, but
the vLLM workers perform device-to-host and host-to-device transfers. The
standalone process therefore owns no CUDA context.

The model forward thread receives an unresolved store future immediately.
Scheduler preemption cannot recycle source blocks until every submitted gather
event completes. Shutdown and explicit flush operations drain both gather and
commit work. The pickle transport remains compatible but is not qualified as a
high-throughput DeepSeek V4 path.

Select the direct LMCache transfer implementation explicitly when comparison
or compatibility testing requires it:

```bash
LMCACHE_MODE=ram \
LMCACHE_TRANSFER_MODE=lmcache_driven \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r5.yml up -d
```

The direct cache process creates CUDA contexts and transfer resources outside
vLLM's memory profiler. Its qualified text profile uses memory utilization
0.965. Engine-driven transfer uses 0.970 for the one-million-token TP2 text
contract and does not consume a separate 512-708 MiB CUDA context per GPU.

## Transfer Comparison

The following measurements use the same two PCIe-switch-connected RTX PRO
6000 Blackwell GPUs, TP2/DCP1, fixed probabilistic K5, MNS8, MNB4096, a
48-row graph cap, FP8 KV, and warmed compiled kernels.

| Transfer | 32k cold | 128k cold | 1M cold | 1M replay |
|---|---:|---:|---:|---:|
| LMCache disabled | 2.372 s | 9.836 s | 165.061 s | n/a |
| LMCache-driven direct | not measured | not measured | 176.559 s | 1.417 s |
| Engine-driven pickle | not measured | not measured | 250.003 s | 1.507 s |
| Engine-driven SHM with synchronous hybrid gather | 2.833 s | 13.479 s | 183.566 s | 1.280 s |
| Engine-driven SHM with asynchronous hybrid gather | 2.393 s | 9.973 s | 165.660 s | 1.189 s |

The asynchronous SHM path adds 0.36% to one-million-token cold latency versus
LMCache-disabled execution in this controlled comparison. It avoids the
33.7% cold-latency penalty of engine-driven pickle and the 6.2% penalty of
direct transfer.

The exact r5 image repeated the steady-state one-million-token comparison with
different deterministic prompt token IDs:

| Mode | Cold | Replay | Cached prompt tokens |
|---|---:|---:|---:|
| LMCache disabled | 165.174 s | n/a | 0 |
| Engine-driven SHM | 166.808 s | 1.148 s | 999,424 |

The exact-image cold overhead is 0.99%. Cold and replay requests produced the
same generated token. The engine-driven profile created a 1,252,661-token GPU
KV pool and retained at least 681 MiB free per rank during the one-million-token
request.

A first request for uncovered compiled shapes took 177.520 seconds and
triggered four inference-time compilations. Reuse the release-scoped `/cache`
mount; first-use compilation is not steady-state ingest performance.

## Capacity Pressure

An engine-driven pressure test constrained the GPU KV pool to 162,026 tokens
and replayed two distinct 130,816-token prompts concurrently. Both outputs
matched their cold results. The 8 GiB L1 staging tier could not hold both full
restores simultaneously, so one request restored 106,496 tokens while the
other restored 130,560. An isolated replay after slot release restored 130,560
tokens.

LMCache exposes this condition through
`lmcache_mp_l1_allocation_failure_chunks_total`. A staging allocation failure
returns the largest available exact prefix; vLLM recomputes the suffix. Increase
`LMCACHE_L1_GB` when simultaneous long-prefix restores must achieve full hit
coverage.

## Vision Qualification

The exact r5 image loaded the Vision target and fixed-K3 draft through
InstantTensor, captured target and draft CUDA graphs, registered all eight
hybrid KV groups, and created a 1,228,734-token GPU KV pool.

| Gate | Result |
|---|---|
| Text prefix | 4,188 prompt tokens; 4,096 cacheable tokens; cold and replay output matched |
| Image prefix | 2,115 prompt tokens including 115 image tokens; 2,048 cacheable tokens |
| Image answer | Both executions identified the supplied red left half and green right half |
| Runtime health | API and LMCache remained healthy after text and image requests |

The Vision checkpoint contains three draft layers, so K3 is its deepest
supported DSpark mode. The target consumes and verifies image embeddings. The
drafter proposes from text-only inputs because it has no external multimodal
embedding interface; this can reduce acceptance without changing target
verification semantics.

## Source Composition

**Implemented, review pending:** vLLM pull requests
[#628](https://github.com/local-inference-lab/vllm/pull/628),
[#630](https://github.com/local-inference-lab/vllm/pull/630), and
[#634](https://github.com/local-inference-lab/vllm/pull/634) are composed onto
`dev/jovian-judgement`. They register B12X graph rows, make explicit NCCL
selection authoritative, provide DeepSeek V4 Vision support, and select the
qualified LMCache memory envelope.

**Implemented, review pending:** B12X pull requests
[#246](https://github.com/local-inference-lab/b12x/pull/246),
[#301](https://github.com/local-inference-lab/b12x/pull/301),
[#302](https://github.com/local-inference-lab/b12x/pull/302), and
[#306](https://github.com/local-inference-lab/b12x/pull/306) are composed onto
`b12x/master`. They provide generation-safe TP2 graph communication, valid
W4A8 profiling, sparse top-k-512 dual-cache prefill, and the Vision
checkpoint's normalization specialization.

**Implemented, review pending:** LMCache pull request
[#44](https://github.com/local-inference-lab/LMCache/pull/44) is composed onto
`release/v0.5.2-glm52-dcp-base`. It provides valid 64-head interleaved page
transfer and asynchronous multi-group engine-driven stores.

## Qualification Limits

- **Qualified:** TP2/DCP1 text fixed K5, Vision fixed K3, B12X W4A8, FP8
  compressed MLA KV, InstantTensor loading, engine-driven SHM cache transfer,
  one-million-token text miss/replay, and Vision text/image inference.
- **Implemented:** target-only execution, LMCache RAM storage, LMCache
  filesystem persistence, and explicit direct transfer.
- **Unsupported:** native vLLM filesystem KV offload and Vision speculative
  depth above K3.
- **Not qualified by this receipt:** TP other than two, DCP greater than one,
  GPUs with less than 96 GiB, or task-level model quality.

