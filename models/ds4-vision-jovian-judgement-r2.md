# DeepSeek-V4-Flash Vision Jovian Judgement r2

**Status: qualified for TP2/DCP1 target-only and fixed probabilistic DSpark K3
serving on NVIDIA SM120.** This specification covers text and image inference
with `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. GPU KV storage is the default;
LMCache RAM storage is supported as an explicit option. Native vLLM filesystem
KV offload is outside this serving contract.

## TL;DR

Download the committed Compose profile, pull the prebuilt image, and start it
on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r2.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r2.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r2.yml up -d
```

The default profile uses fixed probabilistic DSpark K3. Select target-only
serving with:

```bash
MODE=dspark-mtp0 \
docker compose -f docker-compose-ds4-vision-jovian-judgement-r2.yml up -d
```

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllma4c2eee-b12x283a63e-fi803c466-cu133-torch213-20260904-r2` |
| Registry digest | `sha256:868850f68b9a711d43623fab16df9c1ee146089c84529cb78ba5c0831863e67e` |
| Image ID | `sha256:6ad4a9c32a052a9d17b148e05e0b9336769107046b9c7908de02d566af81451e` |
| Docker source | `local-inference-lab/blackwell-llm-docker@3a4bc8fcbc845160ac583232c31177aece26270a` |
| Source merge contract | [`rtx6kpro` issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| Model revision | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| vLLM base | `dev/jovian-judgement@c085b910ebd4a8c89c2c4085cbf17ccaf15a384c` |
| vLLM integration tree | `a4c2eeebfb165ef63848d1f0a9e90e994d1ca16a` |
| B12X base | `master@1a7e3ec286b0ff0b7c2aabee22dce08daab7e011` |
| B12X integration tree | `283a63ee552d38e6a2ffa8a9ec2859ddcb227201` |
| FlashInfer tree | `803c4664f4771ddc418f20a57f752469a237a825` |
| LMCache integration tree | `eb4c227f68a4e1c45d6b8edf6b4934e18f6d1f8b` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, XGrammar 0.2.5, LMCache 0.5.2+glm52dcp.5 |

Image labels record every base revision, pull-request head, integration tree,
generated patch digest, and dependency revision. The runtime contains compiled
installed packages and no source overlay.

## Serving Contract

| Setting | Fixed K3 default | Target-only override |
|---|---:|---:|
| `MODE` | `dspark` | `dspark-mtp0` |
| `DSPARK_TOKENS` | `3` | not applicable |
| `BACKEND` | `b12x-a8-dglin` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | `2` / `1` |
| `MAX_NUM_SEQS` | `4` | `4` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` |
| `MAX_MODEL_LEN` | `1048576` | `1048576` |
| `GPU_MEMORY_UTILIZATION` | `0.975` | `0.975` |
| CUDA graphs | `FULL_AND_PIECEWISE`, cap 16 | `FULL_AND_PIECEWISE`, cap 16 |
| KV format | FP8 compressed MLA | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |
| Host KV storage | disabled | disabled |

The checkpoint contains three next-token draft layers. K3 is therefore the
deepest checkpoint-supported DSpark mode; K5 and K7 are unsupported for this
checkpoint. The target receives and verifies image embeddings. The DSpark
drafter receives text-only draft inputs for image requests because it has no
external multimodal embedding interface. This can reduce image-request draft
acceptance, but target verification remains authoritative.

## LMCache RAM Storage

Enable the qualified external RAM cache with:

```bash
LMCACHE_MODE=ram \
LMCACHE_L1_GB=8 \
docker compose -f docker-compose-ds4-vision-jovian-judgement-r2.yml up -d
```

When `LMCACHE_TRANSFER_MODE=auto` selects direct CUDA transfer and the operator
does not set `GPU_MEMORY_UTILIZATION`, the launcher uses `0.96`. The reduced
envelope reserves the CUDA IPC context memory needed after model loading.
Explicit operator values remain authoritative. `LMCACHE_TRANSFER_MODE=engine_driven`
retains the normal `0.975` default.

The RAM-cache qualification used a 48,092-token deterministic prompt. The
uncached request completed in 10.718 seconds. Replays restored 47,872 prompt
tokens, completed in 0.515-0.552 seconds, and returned the same user-visible
answer hash. LMCache metrics recorded 143,872 hit tokens and 566 L1 reads over
the complete replay sequence. A two-image carrots-and-corn request also stored,
replayed, and returned the correct image interpretation.

## Capacity And Throughput

Two RTX PRO 6000 Blackwell GPUs connected through one PCIe switch ran TP2/DCP1
with the exact registry artifact. Decode measurements used a 30-second CC1
window, context zero, temperature zero, and a 4,096-token output ceiling.

| Mode | GPU KV tokens | CC1 tok/s | Target steps/s | Mean accepted length |
|---|---:|---:|---:|---:|
| Target-only | 2,215,662 | 157.3 | 157.3 | 1.00 |
| Fixed probabilistic DSpark K3 | 1,331,761 | 222.1 | 96.9 | 2.29 |
| Fixed K3 plus LMCache direct/auto | 1,131,336 | not measured | not measured | not measured |

Emitted-token throughput depends on the generated trajectory's DSpark
acceptance. Target steps per second is the acceptance-normalized backend
metric. The table is a release qualification, not a model-quality comparison.

## Source Composition

**Implemented, review pending:** the vLLM integration applies these pull
requests to `dev/jovian-judgement`:

1. [vLLM #628](https://github.com/local-inference-lab/vllm/pull/628)
   registers scheduler-reachable speculative row counts before B12X graph
   warmup.
2. [vLLM #630](https://github.com/local-inference-lab/vllm/pull/630) makes
   explicit NCCL mode clear inherited B12X all-reduce selectors.
3. [vLLM #634](https://github.com/local-inference-lab/vllm/pull/634) provides
   the Vision architecture, incremental checkpoint loading, multimodal
   preprocessing, image-aware sparse attention, DSpark integration, and
   Vision-aware launcher policy.

**Implemented, review pending:** the B12X integration applies these pull
requests to `master`:

1. [B12X #246](https://github.com/local-inference-lab/b12x/pull/246) makes TP2
   graph peer-push generation-safe and binds prepared graph shapes.
2. [B12X #301](https://github.com/local-inference-lab/b12x/pull/301) supports
   FP8 DeepSeek V4 dual-cache prefill with sparse top-k 512.
3. [B12X #302](https://github.com/local-inference-lab/b12x/pull/302) supplies a
   valid W4A8 routed-expert profiling oracle.
4. [B12X #306](https://github.com/local-inference-lab/b12x/pull/306) supports
   the Vision checkpoint's `rms_norm_eps=1e-20` mHC contract.

**Implemented, review pending:**
[LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44) transfers
interleaved 64-head cache pages with their physical stride. The pinned LMCache
base includes the other multiprocess and heterogeneous-cache integration
changes recorded in the image labels.

**Implemented, review pending:** FlashInfer tree
`803c4664f4771ddc418f20a57f752469a237a825` supports the SM120 sparse-MLA
top-k-512 fallback contracts used by TP2 and TP1 Vision serving. The source is
published at `voipmonitor/flashinfer`; no `local-inference-lab/flashinfer`
repository exists for a pull request.

## Qualification Evidence

| Gate | Conditions and result |
|---|---|
| Source composition | Static release assertions passed for source locks, package versions, image labels, launcher policy, and Compose defaults |
| Weight loading | InstantTensor `BUFFERED` loaded all 48 checkpoint shards without materializing the complete 157 GiB state dictionary |
| Target-only graphs | PIECEWISE and FULL target capture completed |
| Fixed-K3 graphs | PIECEWISE target, FULL target, and FULL DSpark capture completed |
| Text inference | Target-only and fixed-K3 requests returned coherent requested answers |
| Image inference | Target-only and fixed-K3 official carrots-and-corn requests identified both images and their edible parts |
| LMCache | Direct/auto RAM storage restored 47,872 prompt tokens and preserved text and image output semantics |
| Runtime health | Every profile remained healthy after graph capture and request replay |

## Qualification Limits

- **Qualified:** TP2/DCP1 target-only and fixed probabilistic DSpark K3, B12X
  W8A8, FP8 compressed MLA KV, text and multi-image requests, target and draft
  CUDA graphs, InstantTensor loading, and LMCache RAM replay.
- **Implemented:** required tool parsing inherited from the DS4 launcher and
  LMCache engine-driven transfer.
- **Unsupported:** native vLLM filesystem KV offload and speculative depths
  greater than three for this checkpoint.
- **Not qualified by this receipt:** TP1, TP greater than two, DCP greater than
  one, a complete 1,048,576-token request, LMCache filesystem persistence, and
  task-level model-quality evaluation.
- Reuse the release-scoped `/cache` mount. An uncovered B12X or FlashInfer shape
  can otherwise compile during the first request.
