# DeepSeek-V4-Flash Vision Jovian Judgement r3

**Status: qualified for TP2/DCP1 target-only and fixed probabilistic DSpark K3
serving on NVIDIA SM120.** This specification covers text and image inference
with `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. GPU KV storage is the default;
LMCache RAM storage is supported as an explicit option. Native vLLM filesystem
KV offload is outside this serving contract.

## TL;DR

Download the committed Compose profile, pull the prebuilt image, and start it
on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-vision-jovian-judgement-r3.yml
docker compose -f docker-compose-ds4-vision-jovian-judgement-r3.yml pull
docker compose -f docker-compose-ds4-vision-jovian-judgement-r3.yml up -d
```

The default profile uses fixed probabilistic DSpark K3 and GPU KV storage.
Enable the qualified LMCache RAM profile with:

```bash
LMCACHE_MODE=ram \
docker compose -f docker-compose-ds4-vision-jovian-judgement-r3.yml up -d
```

Select target-only serving with:

```bash
MODE=dspark-mtp0 \
docker compose -f docker-compose-ds4-vision-jovian-judgement-r3.yml up -d
```

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllmd6f9e77-b12x283a63e-fi803c466-cu133-torch213-20260904-r3` |
| Registry digest | `sha256:57a5437690a51657e0f04dbd6d8adc92e38f7092ceefca7b6624ac5c816c28b6` |
| Image ID | `sha256:f423e9c1d19e78e95c965c7eeeab2c8443d776f442f50c2e995b1d68e8e899d0` |
| Docker source | `local-inference-lab/blackwell-llm-docker@50043a95a3f702aa8b520c5512aa2f4dacfe26c3` |
| Source merge contract | [`rtx6kpro` issue #95](https://github.com/local-inference-lab/rtx6kpro/issues/95) |
| Model revision | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| vLLM base | `dev/jovian-judgement@a50ebee1d2460d22386b54e79f46236376e2b486` |
| vLLM integration tree | `d6f9e777bdf23304ace1ce3b311935390009a149` |
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

The checkpoint contains three next-token draft layers. K3 is the deepest
checkpoint-supported DSpark mode; K5 and K7 are unsupported for this
checkpoint. The target receives and verifies image embeddings. The DSpark
drafter receives text-only draft inputs for image requests because it has no
external multimodal embedding interface. This can reduce image-request draft
acceptance, but target verification remains authoritative.

## LMCache RAM Profile

Direct LMCache transfer uses these launcher defaults on 96 GiB TP2 GPUs:

| Setting | Value |
|---|---:|
| `LMCACHE_MODE` | `ram` |
| `MAX_MODEL_LEN` | `900000` |
| `GPU_MEMORY_UTILIZATION` | `0.951` |
| `LMCACHE_L1_GB` | `8` |
| `LMCACHE_TRANSFER_MODE` | `auto` |

LMCache allocates direct CUDA transfer buffers after vLLM sizes the GPU KV
pool. The 900,000-token profile reserves memory for those buffers and for
concurrent image preprocessing. A clean-start test at memory utilization
0.950 failed admission because 6.41 GiB of KV storage was required when 6.37
GiB was available. The qualified 0.951 profile created 907,467 GPU KV tokens.

Set both `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` to override this profile.
The launcher rejects a direct-LMCache Vision length above 900,000 without an
explicit memory-utilization value. Engine-driven transfer retains the
GPU-only 1,048,576-token and 0.975 defaults.

## Capacity And Throughput

Two RTX PRO 6000 Blackwell GPUs connected through one PCIe switch ran TP2/DCP1.
Decode used vLLM integration tree
`2841848bcddb79391abb8fb275e9fd9991ffb43d`, a 30-second warmed CC1 window,
context zero, temperature zero, and a 4,096-token output ceiling. The released
tree `d6f9e777bdf23304ace1ce3b311935390009a149` differs only in launcher memory
defaults and tests; model execution code is identical.

| Mode | GPU KV tokens | CC1 tok/s | Target steps/s | Mean accepted length |
|---|---:|---:|---:|---:|
| Fixed probabilistic DSpark K3, GPU KV | 1,331,761 | 220.7 | 97.4 | 2.27 |
| Fixed probabilistic DSpark K3, direct LMCache | 907,467 | not measured | not measured | not measured |

The r2 fixed-K3 control produced 222.1 tok/s and 96.9 target steps/s under the
same hardware and benchmark contract. The r3 result differs by -0.6% emitted
throughput and +0.5% target-step throughput, which is within run variance.
Emitted-token throughput depends on the generated trajectory's DSpark
acceptance; target steps per second is the acceptance-normalized backend
metric.

## LMCache Qualification

| Gate | Result |
|---|---|
| Cache miss | 48,092 prompt tokens, 10.73 seconds |
| Cache replay | 47,872 prompt tokens restored, 0.37 seconds after storage completed |
| Long text plus images | An uncached 808,598-token text request overlapped ten 2048x2048 images; both returned HTTP 200 |
| Memory reserve | Minimum free memory during the overlapping requests was 2,239/2,249 MiB |
| Service health | vLLM and LMCache health endpoints remained healthy after replay and overlapping requests |

The GPU-only 1,048,576-token control also completed an uncached 810,098-token
text request overlapping ten 2048x2048 images. Its minimum free memory was
325/327 MiB. Direct LMCache at the GPU-only 0.975 memory envelope reproduced
the reported allocation failure; the failure is specific to combining the
GPU-only memory budget with LMCache's deferred transfer allocations.

## Source Composition

**Implemented, review pending:** the vLLM integration applies
[vLLM #628](https://github.com/local-inference-lab/vllm/pull/628),
[vLLM #630](https://github.com/local-inference-lab/vllm/pull/630), and
[vLLM #634](https://github.com/local-inference-lab/vllm/pull/634) to
`dev/jovian-judgement`. These changes bind scheduler-reachable B12X graph
shapes, make explicit NCCL selection authoritative, and provide the Vision
architecture, incremental loader, multimodal preprocessing, image-aware
sparse attention, DSpark integration, and Vision-aware memory policy.

**Implemented, review pending:** the B12X integration applies
[B12X #246](https://github.com/local-inference-lab/b12x/pull/246),
[B12X #301](https://github.com/local-inference-lab/b12x/pull/301),
[B12X #302](https://github.com/local-inference-lab/b12x/pull/302), and
[B12X #306](https://github.com/local-inference-lab/b12x/pull/306) to `master`.
These changes provide generation-safe TP2 graph peer-push, sparse top-k-512
dual-cache prefill, a valid W4A8 routed-expert profiling oracle, and the
checkpoint's `rms_norm_eps=1e-20` specialization.

**Implemented, review pending:**
[LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44) transfers
interleaved 64-head cache pages with their physical stride. The pinned LMCache
base includes the multiprocess and heterogeneous-cache changes recorded in the
image labels.

**Implemented:** FlashInfer tree
`803c4664f4771ddc418f20a57f752469a237a825` supplies SM120 sparse-MLA
top-k-512 fallback support. The source is published in `voipmonitor/flashinfer`;
no `local-inference-lab/flashinfer` repository exists for a pull request.

## Tokenizer Diagnostic

The string `)Skip` is vocabulary token 83480 in this checkpoint. It is an
ordinary model token, not a special token, control token, parser marker, or
runtime-injected value. One report observed the token 12 times in 220,000
generated tokens, but no request-level reproducer is available. The runtime
does not suppress valid vocabulary tokens without a correctness contract. A
reproducible report must include the exact request JSON, sampling parameters,
and surrounding generated token sequence.

## Qualification Limits

- **Qualified:** TP2/DCP1 target-only and fixed probabilistic DSpark K3, B12X
  W8A8, FP8 compressed MLA KV, text and multi-image requests, target and draft
  CUDA graphs, InstantTensor loading, LMCache RAM replay, and concurrent long
  text plus image pressure.
- **Implemented:** required tool parsing inherited from the DS4 launcher and
  LMCache engine-driven transfer.
- **Unsupported:** native vLLM filesystem KV offload and speculative depths
  greater than three for this checkpoint.
- **Not qualified by this receipt:** TP1, TP greater than two, DCP greater than
  one, LMCache filesystem persistence, and task-level model-quality evaluation.
- Reuse the release-scoped `/cache` mount. An uncovered B12X or FlashInfer
  shape can otherwise compile during the first request.
