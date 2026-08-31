# GLM-5.2 v18: Gilded Gnosis Fast DCP

v18 is the clean, unified GLM-5.2 and DS4 image built from the consolidated
`dev/gilded-gnosis` line. It is the cumulative successor to
[v15](glm5.2_v15.md), [v16](glm5.2_v16.md), and [v17](glm5.2_v17.md).
It keeps their checkpoint formats, online quantization, TP6 virtual sharding,
InstantTensor loader, hybrid DCP communication, and DS4 launcher. It adds:

- transient full-CKV gather and local-query execution for TP8/DCP4 and
  TP8/DCP8 sparse-MLA prefill;
- cross-layer CKV prefetch with ping-pong workspaces;
- exact TP4 Grid188 decode for the mixed NVFP4/NF3 checkpoint;
- NVFP4 MLA KV cache support on the consolidated GG source;
- CUDA graph lifetime fixes for DCP A2A buffers;
- same-repository MTP revision inheritance, including local checkpoints;
- an immutable-image regression campaign covering all maintained checkpoint
  and online-quant modes at `F8_DMA=0`.

No source directory, wheel, or runtime patch is mounted into the release
container.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v18-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718
Docker manifest: sha256:1a6c388b76dee43969760ca700ddaf222dc133f5d603a2e32124fcccdfd9c15e
Local image ID: sha256:a1202a5b9f712910306b684de05c4bbea05c37609c0eb0e9394fd7d980fa49ac
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `5b116d1a27` after the consolidation audit fix |
| vLLM release source | [`build/gilded-gnosis-v18-final-20260718`](https://github.com/local-inference-lab/vllm/tree/264bce1da81e27d638e7cf265b4cbd125d023c38) @ `264bce1da81e27d638e7cf265b4cbd125d023c38` |
| B12X | current `lukealonso/b12x` base `e71a090` plus [Grid188 PR #36](https://github.com/lukealonso/b12x/pull/36), release commit `bc85ef36192cb6e444d42ba7be86e1e125cca98a` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA | PyTorch `2.12.0+cu132`, CUDA `13.2.1` |
| Build repository | `local-inference-lab/blackwell-llm-docker` main @ `7f3cbc6` |

The canonical build script is
[`build-gilded-gnosis-v18-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/7f3cbc6/build-gilded-gnosis-v18-cu132.sh).
It checks every source pin, exercises the NVFP4 MLA CUDA writer, validates the
NF3 Grid188 mapping, and dry-runs all GLM and DS4 launcher modes before an
optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 7f3cbc6
PUSH_IMAGE=1 ./build-gilded-gnosis-v18-cu132.sh
```

## Source Audit

The GG consolidation was compared against the final FF/v17 source by commit
ancestry and content. [PR #112](https://github.com/local-inference-lab/vllm/pull/112)
restored the consolidated runtime invariants and is merged in the GG base.
The release branch then applies only public, reviewable changes:

| Change | Review |
|---|---|
| DSpark hardening and optional experiments | [vLLM PR #109](https://github.com/local-inference-lab/vllm/pull/109) |
| SM120 PCIe serving stack | [upstream vLLM PR #47979](https://github.com/vllm-project/vllm/pull/47979) |
| TP8 full-CKV DCP prefill | [vLLM PR #111](https://github.com/local-inference-lab/vllm/pull/111) |
| NF3 Grid188 integration | [vLLM PR #113](https://github.com/local-inference-lab/vllm/pull/113), [B12X PR #36](https://github.com/lukealonso/b12x/pull/36) |
| Environment-only DS4 helper | [vLLM PR #114](https://github.com/local-inference-lab/vllm/pull/114) |
| NVFP4 MLA KV cache | [vLLM PR #115](https://github.com/local-inference-lab/vllm/pull/115) |
| B12X scratch-format guard | [vLLM PR #116](https://github.com/local-inference-lab/vllm/pull/116) |
| DCP A2A CUDA graph buffer lifetime | [vLLM PR #117](https://github.com/local-inference-lab/vllm/pull/117) |
| MTP target-revision inheritance | [vLLM PR #118](https://github.com/local-inference-lab/vllm/pull/118) |

PR #109 is deliberately not part of canonical GG yet, but it is included in
this immutable v18 image. There are no private source edits hidden in a Docker
overlay.

## Start The Server

The image contains `/usr/local/bin/serve-gilded-gnosis.sh`. The host Compose
file selects a model family and exposes only deployment choices; backend
flags, the 78-character sparse-indexer pattern, graph sizing, InstantTensor,
NCCL, and DCP fast-path gates remain owned by the image helper.

Use the maintained
[`docker-compose-glm52-v18.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/7f3cbc6/examples/docker-compose-glm52-v18.yml):

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

# Highest-accuracy standard mode: Luke NVFP4, A16, DCP1, MTP off.
MOE_MODE=a16 MTP=0 DCP=1 \
  docker compose -f examples/docker-compose-glm52-v18.yml up -d
```

The helper calculates `GRAPH=4*MAX_NUM_SEQS` when `GRAPH` is not supplied.
InstantTensor and its page-cache-aware buffered backend are defaults:

```text
LOAD_FORMAT=instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

The release validation requires both of these log lines before benchmarking:

```text
Loading safetensors using InstantTensor loader
vLLM is using nccl==2.30.4
```

### Checkpoint Modes

| Checkpoint | `QUANTIZATION` | `MOE_MODE` | `ONLINE_QUANT` |
|---|---|---|---|
| `lukealonso/GLM-5.2-NVFP4` | `modelopt_fp4` | `a4` or `a16` | `none` or `mxfp8` |
| `festr2/GLM-5.2-BF16-AMDMXFP4experts` | `mxfp4` | `force-a8-experimental` | `none`, `mxfp8`, or `fp8` |
| `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` | `nvfp4_nf3_hybrid` | `a16` | `nf3-mxfp8` |

For Luke NVFP4, A4 means checkpoint-native NVFP4 activations. A16 keeps the
same 4-bit expert weights but uses BF16 expert activations. Force-A8 applies
to the AMD MXFP4-experts checkpoint, not to Luke NVFP4.

Online MXFP8 converts only eligible BF16 dense linears:

```json
{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}
```

Existing NVFP4 or MXFP4 expert tensors remain in their checkpoint format.
Shared experts are not converted by the generic `linear` rule. The detailed
layer and accuracy rationale remains in [v15](glm5.2_v15.md#online-fp8mxfp8-conversion).

### Fast DCP

For TP8/DCP4 and TP8/DCP8, the v18 helper enables:

```text
VLLM_DCP_QUERY_SPLIT=1
VLLM_B12X_MLA_CKV_GATHER=1
```

Instead of gathering query heads and running attention against rank-local KV,
the new prefill path keeps local query heads and transiently gathers the
compressed KV history. Layer `L` prefetches layer `L+1` on a side stream into
a ping-pong workspace. This removes the old query-gather and partial-output
merge bottleneck as DCP grows. Decode does not use this path.

The helper enables it automatically only on the validated TP8/DCP4 and
TP8/DCP8 topologies. An A/B test can force it off with both:

```bash
DCP_QUERY_SPLIT=0 DCP_CKV_GATHER=0
```

The older project-before-merge workspace remains automatic for other
validated DCP topologies. DCP1 has no DCP collective and is unaffected.

## Accuracy

The v18 changes do not change standard-checkpoint weights or arithmetic, so
KLD was not rerun. These are the corrected five-run means inherited from v15
against the current BF16 reference logits:

| Case | KLD mean +/- sample SD |
|---|---:|
| Luke NVFP4 A4 original | 0.10228 +/- 0.00634 |
| Luke NVFP4 A4 online MXFP8 | 0.10800 +/- 0.00697 |
| Luke NVFP4 A16 original | 0.05994 +/- 0.00129 |
| Luke NVFP4 A16 online MXFP8 | 0.06587 +/- 0.00253 |
| AMD MXFP4 experts A8 original | 0.08160 +/- 0.00432 |
| AMD MXFP4 experts A8 online MXFP8 | 0.08030 +/- 0.00309 |

Use [the GLM KLD reproduction page](../benchmarks/glm52-kld-evaluation.md)
for the reference-logit provenance and exact commands. Do not combine these
numbers with the superseded June reference logits.

## Validation Method

The release campaign used all 16 RTX PRO 6000 Blackwell GPUs as two isolated
instances. Both models were fully loaded and healthy before any client ran;
clients then ran serially so one endpoint was never benchmarked while another
checkpoint was loading. Each pair had a 30-second settle period.

| Profile | TP | DCP | MTP | Max seqs | Graph | Batched tokens | Max model len | GMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 8 | 1, 4, 8 | 0 or 3 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| TP6 | 6 | 3, 6 | 3 | 16 | 64 | 4,096 | 128,000 | 0.950 |
| NF3 hybrid | 4 | 4 | 0 or 3 | 8 | 64 | 3,072 | 131,072 | 0.960 |

All rows use `F8_DMA=0`, B12X sparse MLA and MoE, FP8 KV unless stated,
InstantTensor `BUFFERED`, and the helper's hybrid A2A/AG-RS policy. DCP2 was
not repeated. DCP1 historical comparisons use the v15/v17 estimated-token
profile. New DCP4/DCP8 fast-path rows use exact 65,536-token prompts. One 64k
request is discarded before two or three measured repeats.

`F8_DMA=ag` and `ring` were not repeated for v18. Their v17 tables remain
historical transport experiments, and DMA mode does not accelerate decode.
Do not combine those old cells with the new full-CKV numbers as one campaign.

The runner asserts the image ID, rejects source mounts, checks mode-specific
kernel logs, verifies MTP acceptance from the exact server-log window, and
requires both fast-path log markers for every TP8/DCP4 or TP8/DCP8 row.

## DCP1 Regression Gate

The table compares the same MTP-off, DCP1 profile. Prior values are the
canonical v15/v17 values, except online FP8 decode, which uses the directly
comparable final-v16 JSON (`99.278 tok/s`). The older `101.9 tok/s` text was a
different earlier probe and is not the release regression baseline.

| Case | v18 decode CC1 | Prior | Change | v18 prefill 64k | Prior | Change |
|---|---:|---:|---:|---:|---:|---:|
| Luke NVFP4 A4 original | 88.11 | 87.99 | +0.14% | 6,334.5 | 6,257 | +1.24% |
| Luke NVFP4 A4 online MXFP8 | 93.78 | 94.96 | -1.24% | 6,362 | 6,351 | +0.17% |
| Luke NVFP4 A16 original | 87.20 | 86.56 | +0.74% | 5,912.5 | 5,849 | +1.09% |
| Luke NVFP4 A16 online MXFP8 | 94.33 | 93.30 | +1.10% | 5,996 | 5,941 | +0.93% |
| AMD MXFP4 experts A8 original | 88.37 | 88.72 | -0.39% | 6,424 | 6,307 | +1.86% |
| AMD MXFP4 experts A8 online MXFP8 | 94.37 | 94.03 | +0.36% | 6,442 | 6,364 | +1.23% |
| AMD MXFP4 experts A8 online FP8 | 99.40 | 99.28 | +0.13% | 6,528 | 6,350 | +2.80% |

No maintained DCP1 mode regressed. One A16-online prefill repeat was a slow
`5,642 tok/s` outlier; the next two were `6,002` and `5,996`, so the median is
reported and the raw values remain in the machine-readable results.

## MTP3 And Acceptance

All rows below are TP8/DCP1 with MTP3. The server log, not the currently empty
client acceptance-length field, is the source of acceptance statistics.

| Case | Decode CC1 | Prefill 64k | Mean accepted length | Draft acceptance |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 original | 145.79 | 6,202 | 3.03 | 67.7% |
| Luke NVFP4 A4 online MXFP8 | 146.52 | 6,298 | 2.77 | 59.1% |
| Luke NVFP4 A16 original | 131.12 | 5,729 | 2.80 | 59.9% |
| Luke NVFP4 A16 online MXFP8 | 139.32 | 5,879 | 2.88 | 62.6% |
| AMD MXFP4 experts A8 original | 138.73 | 6,241 | 2.75 | 58.4% |
| AMD MXFP4 experts A8 online MXFP8 | 146.86 | 6,393 | 2.83 | 61.1% |

The inherited full concurrency sweeps and coding peaks remain in
[v17](glm5.2_v17.md#inherited-decode-results). v18's MTP0 regression gate and
the measured acceptance above show no MTP-specific regression.

## TP8 Fast DCP Results

These are exact 64k results from the baked release image. The `fast path`
assertion passed for every row.

### DCP4

| Case | Decode CC1 | Exact prefill 64k | KV tokens | v17 64k | Change |
|---|---:|---:|---:|---:|---:|
| Luke NVFP4 A4 original | 72.79 | 5,684.5 | 2,219,264 | 3,740 | +52.0% |
| Luke NVFP4 A4 online MXFP8 | 77.84 | 5,747 | 2,415,872 | 3,793 | +51.5% |
| Luke NVFP4 A16 original | 72.00 | 5,345 | 2,234,880 | 3,576 | +49.5% |
| Luke NVFP4 A16 online MXFP8 | 76.67 | 5,372.5 | 2,431,232 | 3,618 | +48.5% |
| AMD MXFP4 experts A8 original | 72.79 | 5,744 | 2,414,848 | 3,792 | +51.5% |
| AMD MXFP4 experts A8 online MXFP8 | 77.97 | 5,829.5 | 2,611,456 | 3,812 | +52.9% |
| AMD MXFP4 experts A8 online FP8 | 80.25 | 5,828 | 2,353,152 | n/a | n/a |

### DCP8

| Case | Decode CC1 | Exact prefill 64k | KV tokens | v17 64k | Change |
|---|---:|---:|---:|---:|---:|
| Luke NVFP4 A4 original | 67.68 | 4,680.5 | 4,451,328 | 2,474 | +89.2% |
| Luke NVFP4 A4 online MXFP8 | 71.80 | 4,714.5 | 4,844,544 | 2,477 | +90.3% |
| Luke NVFP4 A16 original | 66.75 | 4,427 | 4,467,200 | 2,388 | +85.4% |
| Luke NVFP4 A16 online MXFP8 | 70.69 | 4,476.5 | 4,859,904 | 2,422 | +84.8% |
| AMD MXFP4 experts A8 original | 67.85 | 4,727 | 4,842,496 | 2,476 | +90.9% |
| AMD MXFP4 experts A8 online MXFP8 | 71.94 | 4,765 | 5,235,712 | 2,491 | +91.3% |
| AMD MXFP4 experts A8 online FP8 | 73.93 | 4,771.5 | 4,719,104 | n/a | n/a |

The v17 comparison column is its published estimate-targeting campaign. The
direct exact-to-exact A16-original comparison is `3,576 -> 5,345` on DCP4
and `2,388 -> 4,427` on DCP8, confirming that the gain is not a token-targeting
artifact.

## TP6 MTP3

TP6 virtual sharding remains automatic. The target and same-repository draft
both receive the padded 66-head / 2,112-expert layout. PR #117 also retains
captured DCP A2A buffer ownership, fixing the long-context corruption that
appeared only with multiple FULL graph sizes.

| Case | DCP | Decode CC1 | Prefill 64k | KV tokens | Accepted length | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| AMD MXFP4 A8 original | 3 | 95.84 | 3,500 | 680,107 | 2.77 | 58.9% |
| AMD MXFP4 A8 online MXFP8 | 3 | 97.90 | 3,491 | 851,862 | 2.79 | 59.8% |
| AMD MXFP4 A8 original | 6 | 84.41 | 2,372 | 1,342,467 | 2.75 | 58.3% |
| AMD MXFP4 A8 online MXFP8 | 6 | 88.39 | 2,373 | 1,685,461 | 2.75 | 58.2% |

All four configurations completed graph capture, greedy correctness, decode,
and 64k prefill without garbled output. TP6/DCP3 and TP6/DCP6 use the v17
workspace path; the new PR #111 full-CKV gate is TP8-only.

## NF3 Hybrid

The mixed checkpoint uses TP4/DCP4, A16, `nvfp4_ds_mla` KV cache, and exact
Grid188 one-grid decode. Both the `armed` and `executing` log assertions must
be present.

| MTP | Decode CC1 | Prefill 64k | KV tokens | Accepted length | Acceptance |
|---:|---:|---:|---:|---:|---:|
| 0 | 45.37 | 2,347 | 778,240 | n/a | n/a |
| 3 | 105.69 | 2,288 | 529,920 | 2.89 | 63.1% |

The MTP0 result matches v17 (`44.7 / 2,341`) within normal run variance. The
MTP3 result retains the previously measured Grid188 decode gain.

## Reproduce The Campaign

The exact runner is
[`scripts/bench-glm52-v18-validation.sh`](../scripts/bench-glm52-v18-validation.sh).
It defaults to the immutable image ID and refuses to run against a different
local tag target.

```bash
cd rtx6kpro

# Historical-compatible DCP1 rows.
TOKEN_TARGETING=estimate scripts/bench-glm52-v18-validation.sh dcp1-mtp0
TOKEN_TARGETING=estimate scripts/bench-glm52-v18-validation.sh dcp1-mtp3

# New TP8 DCP4/DCP8 path. DCP2 is intentionally excluded.
TOKEN_TARGETING=exact scripts/bench-glm52-v18-validation.sh dcp-fast

# TP6 MTP3 correctness/performance and TP4 NF3 Grid188.
TOKEN_TARGETING=estimate scripts/bench-glm52-v18-validation.sh tp6-mtp3
TOKEN_TARGETING=estimate scripts/bench-glm52-v18-validation.sh nf3
```

The runner can resume safely: a row is skipped only when both `summary.json`
and its `complete` marker exist. `FORCE_RERUN=1` invalidates that shortcut.

Published machine-readable snapshots:

- [JSON summary](../benchmarks/glm52-v18-validation-summary.json)
- [TSV summary](../benchmarks/glm52-v18-validation-summary.tsv)

Raw host result root used for this release:

```text
/root/bench-results/glm52-v18-validation-estimate-check-20260718
```

The same image serves DS4 through
[`docker-compose-ds4-v18.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/7f3cbc6/examples/docker-compose-ds4-v18.yml).
DS4/DSpark performance remains documented separately; this GLM validation did
not reuse or reinterpret DS4 benchmark cells.
