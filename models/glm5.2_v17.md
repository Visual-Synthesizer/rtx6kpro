# GLM-5.2 v17: Unified Serving And DCP Prefill

This page is the cumulative successor to the GLM-5.2
[v15](glm5.2_v15.md) and [v16](glm5.2_v16.md) pages for RTX 6000 Pro
Blackwell. It keeps their checkpoint modes, online quantization controls,
accuracy references, decode sweeps, InstantTensor loader, and TP6 support in
one clean source image. It adds:

- native TP4 serving for `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`;
- NVFP4 MLA KV cache for the hybrid checkpoint;
- the B12X NF3 tile-binding correctness fix;
- optimized sparse-MLA DCP prefill for TP4/DCP4, TP6/DCP2/3/6, and
  TP8/DCP2/4/8;
- virtual-TP padding for independently constructed MTP draft models, fixing
  TP6 with MTP enabled;
- a selective v17 benchmark campaign that remeasures only the affected prefill
  cells and preserves unchanged v15/v16 results with explicit provenance.

The same image remains the unified GLM-5.2 and DS4 base. It is a clean source
build; runtime source or wheel overlays are not required.

## Version Scope

| Area | v17 source | Remeasured for v17? |
|---|---|---|
| Image, helper, TP6 correctness | v16 plus current FF/B12X stack | yes, startup/runtime validation |
| KLD and quantization accuracy | corrected v15 BF16-reference campaign | no; weights and arithmetic are unchanged |
| Decode sweeps and coding peaks | v15/v16 | no; PR #94 is gated out of decode |
| DCP1 prefill | v15/v16 | no; no DCP collective or workspace path exists |
| TP8 DCP2/4/8 prefill | v15 matrix, remeasured on v17 | yes |
| TP6 DCP2/3/6 prefill | v16 matrix, remeasured on v17 | yes |
| TP4 hybrid checkpoint | new in v17 | yes |

This separation is intentional. Copying the old TP8/TP6 DCP prefill cells
would hide the PR #94 speedup, while rerunning KLD, decode, coding, or DCP1
would only measure noise on code paths the change cannot reach.

## Release Image

```text
voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715
Docker manifest: sha256:9b6f1ab6db4d3a7b7b786481eb32abe82e86d185648d62c3ac1cfa6d72a55e47
Local image ID: sha256:b9346a51992e4ff6897905fac1aa6819ab069a25de6a395a2d726ce520de5230
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| vLLM | `local-inference-lab/vllm build/fathomless-firmament-v17-tp6-mtp-fix-20260715` @ `05f50ae79c48835275f22f76e8dfb10b0024dec6` |
| vLLM v17 parent | `codex/fathomless-firmament-v17-dcp-prefill-opt-20260714` @ `6ccc3ebbd17edb05ce11b095a5b14f25839774dd` |
| vLLM changes | hybrid format [#92](https://github.com/local-inference-lab/vllm/pull/92), NVFP4 KV [#82](https://github.com/local-inference-lab/vllm/pull/82), generalized DCP prefill [#94](https://github.com/local-inference-lab/vllm/pull/94), and TP6 MTP draft padding [#96](https://github.com/local-inference-lab/vllm/pull/96) |
| B12X | `voipmonitor/b12x codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714` @ `1377d5f22c98de0c17d9b3f35a5b56d7587992fa` |
| B12X changes | NF3/NVFP4 work from [lukealonso/b12x #31](https://github.com/lukealonso/b12x/pull/31) plus the [preplanned-tile fix](https://github.com/MadeBy561/b12x/pull/1) |
| FlashInfer | `voipmonitor/flashinfer codex/sm120-dspark-stack-20260711` @ `801d57a08958c13d375ddbb6be3be4808f48a708` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | local-inference `2.30.4`, CUDA 13.2 |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main` @ `ee75ffa239565504cc2b86735cd91a65cf711501` |

The canonical build script is
[`build-fathomless-firmament-v17-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/ee75ffa239565504cc2b86735cd91a65cf711501/build-fathomless-firmament-v17-cu132.sh).
It clones the exact vLLM, B12X, FlashInfer, InstantTensor, DeepGEMM, and
CUTLASS commits, builds the wheel, maps PyTorch and InstantTensor to the same
local NCCL 2.30.4 library, and validates the installed source paths.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout ee75ffa239565504cc2b86735cd91a65cf711501
PUSH_IMAGE=1 ./build-fathomless-firmament-v17-cu132.sh
```

### TP6 With MTP

Virtual TP pads GLM-5.2 from 64 to 66 attention heads and from 2,048 to 2,112
routed experts so TP6 can shard both dimensions. The July 14 image applied
that transformation to the target model, but an independently constructed MTP
draft was still validated with 64 heads. TP6 with MTP therefore failed before
worker or weight initialization.

[vLLM PR #96](https://github.com/local-inference-lab/vllm/pull/96) applies the
same virtual-TP transformation to the draft configuration before TP
validation. The PR is based directly on `dev/fathomless-firmament` at
`522c626de89b629a18d05db21ba02b5acf6e6f30`, so it does not depend on a Codex
integration branch. The release commit
`05f50ae79c48835275f22f76e8dfb10b0024dec6` carries the equivalent fix on the
exact v17 parent so the rest of the tested v17 stack is preserved.

The release build checks the helper expansion for TP6/DCP6/MTP3. A
model-aware validation using the real `lukealonso/GLM-5.2-NVFP4`
configuration produced:

```text
target: attention_heads=66, moe_experts=2112
draft:  attention_heads=66, moe_experts=2112
```

The existing minimal Compose configuration is valid; only its image tag must
be updated. For example:

```yaml
environment:
  GPUS: 1,2,3,4,5,6
  TP: 6
  DCP: 6
  MTP: 3
```

This validation constructed both configs from the cached checkpoint. It did
not repeat a full six-GPU weight load or speed benchmark because the patch
does not change runtime kernels.

## Standard Checkpoints And Modes

The general GLM helper supports the two checkpoint families carried forward
from v15/v16:

| Checkpoint | `QUANTIZATION` | Expert mode | Optional dense conversion |
|---|---|---|---|
| `lukealonso/GLM-5.2-NVFP4` | `modelopt_fp4` | `MOE_MODE=a4` or `a16` | `ONLINE_QUANT=none` or `mxfp8` |
| `festr2/GLM-5.2-BF16-AMDMXFP4experts` | `mxfp4` | `MOE_MODE=force-a8-experimental` | `ONLINE_QUANT=none`, `mxfp8`, or `fp8` |

For Luke NVFP4, A4 is the checkpoint-native NVFP4 activation path. A16 keeps
the same 4-bit expert weights but forces BF16 expert activations through the
B12X W4A16 path. A8 force is not a valid comparison axis for that checkpoint.

For the AMD MXFP4-experts checkpoint, `force-a8-experimental` selects its
W4A8-MX expert path. Online conversion applies only to eligible BF16 dense
linear weights; it does not rewrite the stored MXFP4 routed experts.

`ONLINE_QUANT=mxfp8` resolves to:

```json
{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}
```

`kv_b_proj` remains BF16 because MLA absorb materializes BF16 `W_UK`/`W_UV`
copies at load time; quantizing the source adds rounding error without running
a quantized GEMM. Shared-expert projections are also excluded unless an
explicit `shared_experts` rule is supplied. Existing NVFP4, NF3, or MXFP4
expert tensors are never requantized by the generic `linear` rule.

### Online FP8 Versus MXFP8 Dense Weights

The AMD MXFP4-experts checkpoint also supports `ONLINE_QUANT=fp8`, which
converts eligible BF16 dense linears to the static block-FP8 format while
leaving routed MXFP4 experts unchanged. The inherited DCP1 measurements show
the tradeoff clearly:

| Online dense format | `F8_DMA` | Decode CC1 | Coding peak | Prefill 30k | Prefill 64k | Prefill 120k |
|---|---|---:|---:|---:|---:|---:|
| MXFP8 | `0` | 97.2 | 97.6 | 6,706 | 6,396 | 6,058 |
| FP8 block | `0` | 101.9 | 102.5 | 6,638 | 6,350 | 5,984 |
| MXFP8 | `ring` | n/a | n/a | 8,303 | 7,841 | 7,284 |
| FP8 block | `ring` | n/a | n/a | 8,304 | 7,837 | 7,271 |

FP8 block is about 5 tok/s faster for single-user decode. The two dense
formats are effectively tied for prefill; transport mode dominates that axis.
Decode is intentionally not repeated for `ring`, because FP8 DMA does not
carry the small decode payloads.

### Start A Standard Variant

The image contains the canonical helper at
`/usr/local/bin/serve-fathomless-firmament.sh`. It selects the GLM launcher
with `MODEL_FAMILY=glm52`, computes `GRAPH=4*MAX_NUM_SEQS` when omitted, and
owns the exact 78-character sparse-indexer pattern. The host only supplies the
deployment and model choices.

The maintained minimal Compose file is
[`examples/docker-compose-glm52-v17.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/ee75ffa239565504cc2b86735cd91a65cf711501/examples/docker-compose-glm52-v17.yml).

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

# Luke NVFP4, highest-accuracy A16 mode, MTP off.
IMAGE=voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715 \
GPUS=0,1,2,3,4,5,6,7 PORT=8000 TP=8 DCP=1 MTP=0 \
MOE_MODE=a16 ONLINE_QUANT=none MAX_NUM_SEQS=64 \
  docker compose -f examples/docker-compose-glm52-v17.yml up -d
```

Online MXFP8 uses the same command with `ONLINE_QUANT=mxfp8`. The AMD MXFP4
checkpoint uses:

```bash
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental ONLINE_QUANT=mxfp8 \
  docker compose -f examples/docker-compose-glm52-v17.yml up -d
```

The user-facing controls are:

| Environment | Default | Meaning |
|---|---|---|
| `GPUS` | `0,1,2,3,4,5,6,7` | Physical GPUs assigned to the instance |
| `PORT` | `8000` | OpenAI-compatible API port |
| `TP` | `8` | Tensor-parallel size; TP6 virtual padding is automatic |
| `DCP` | `1` | Decode-context parallel size |
| `DCP_PREFILL_WORKSPACE` | `auto` | Enable PR #94 only for validated topologies; `0` is the A/B baseline |
| `MTP` | `0` | Native speculative-token count; `0` disables MTP |
| `MAX_NUM_SEQS` | `64` | Scheduler concurrency cap |
| `GRAPH` | `4*MAX_NUM_SEQS` | Maximum CUDA graph capture size when explicitly overridden |
| `MAX_BATCHED_TOKENS` | `8192` | Chunked-prefill scheduler budget |
| `MAX_MODEL_LEN` | `131072` | Maximum request length |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Per-GPU vLLM memory budget |
| `MOE_MODE` | `a4` | `a4`, `a16`, or `force-a8-experimental` as described above |
| `ONLINE_QUANT` | `none` | `none`, `mxfp8`, `fp8`, or advanced `custom` |
| `F8_DMA` | `0` | FP8 PCIe DMA prefill mode: `0`, `ag`, or `ring` |
| `LOAD_FORMAT` | `instanttensor` | Canonical loader; use another value only for an explicit loader experiment |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Uses the Linux page cache for hot model reloads |

For DCP greater than one, the helper uses hybrid communication: B12X A2A up
to 64 active rows and AG/RS for larger prefill/extend batches. FP8 DMA is a
large-payload prefill knob; it does not materially change decode speed, so
decode tables are consolidated at `F8_DMA=0`.

## Accuracy Reference

PR #94 changes eager prefill storage and collectives, not model weights or
arithmetic. KLD was therefore not rerun. These are the corrected five-run
means inherited from v15, all scored against the same current BF16 reference
logits with `context_length=2048`, `stride=512`, and `max_windows=1`:

| Checkpoint / mode | KLD mean +/- sample SD | Min | Max |
|---|---:|---:|---:|
| Luke NVFP4 A4 original | 0.10228 +/- 0.00634 | 0.09368 | 0.11098 |
| Luke NVFP4 A4 online MXFP8 | 0.10800 +/- 0.00697 | 0.09941 | 0.11877 |
| Luke NVFP4 A16 original | 0.05994 +/- 0.00129 | 0.05844 | 0.06167 |
| Luke NVFP4 A16 online MXFP8 | 0.06587 +/- 0.00253 | 0.06288 | 0.06921 |
| AMD MXFP4 experts A8 original | 0.08160 +/- 0.00432 | 0.07460 | 0.08597 |
| AMD MXFP4 experts A8 online MXFP8 | 0.08030 +/- 0.00309 | 0.07818 | 0.08568 |

The inherited online-A4 transport check is the only matched KLD comparison
available for FP8 DMA modes:

| Online A4 `F8_DMA` | KLD mean +/- sample SD | Interpretation |
|---|---:|---|
| `0` | 0.10800 +/- 0.00697 | uncompressed transport baseline |
| `ag` | 0.10468 +/- 0.00729 | overlaps the baseline run distribution |
| `ring` | 0.11525 +/- 0.00275 | slightly higher mean; still the same broad range |

Do not interpret `ag` as an accuracy improvement: these are independent
five-run measurements, and its mean shift is smaller than the observed run
spread. FP8 DMA is selected for prefill throughput, not quality.

Reference and raw result roots:

```text
/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref
/root/kld/glm52_v14_keypoints_current_bf16ref_20260708T0027Z
/root/kld/glm52_v14_online_a4_f8_ag_current_bf16ref_20260708T0130Z
/root/kld/glm52_v14_online_a4_f8_ring_current_bf16ref_20260708T0130Z
```

The BF16 self-check is exactly zero over 2,047 positions. Do not combine these
numbers with the superseded June 18 reference logits. The separate
[Unsloth-style reproduction](glm5.2/glm52-unsloth-style-prefill-kld-2026-07-07.md)
uses a different corpus/window protocol and is not numerically interchangeable.

## Inherited Decode Results

Decode is unchanged from v15/v16 because the DCP workspace optimization is
restricted to eager prefills above 1,024 rows. The table below consolidates
the TP8, MTP-off, `F8_DMA=0` comparison. Each cell is `CC1 / CC32` aggregate
tok/s at context zero. These are the later v15 hybrid-DCP table values and are
the canonical MTP-off comparison for this page.

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 original | 87.99 / 934.07 | 72.44 / 838.57 | 71.65 / 747.11 | 67.29 / 606.35 |
| Luke NVFP4 A4 online MXFP8 | 94.96 / 953.24 | 76.26 / 847.24 | 75.32 / 760.87 | 70.84 / 617.18 |
| Luke NVFP4 A16 original | 86.56 / 932.72 | 71.48 / 828.30 | 70.74 / 750.20 | 66.11 / 610.88 |
| Luke NVFP4 A16 online MXFP8 | 93.30 / 954.52 | 74.85 / 837.81 | 73.99 / 752.91 | 69.45 / 610.40 |
| AMD MXFP4 experts A8 original | 88.72 / 938.10 | 71.84 / 832.28 | 71.73 / 745.91 | 67.15 / 613.70 |
| AMD MXFP4 experts A8 online MXFP8 | 94.03 / 956.30 | 75.66 / 840.02 | 75.37 / 761.43 | 71.01 / 607.69 |

For decode, DCP1 is fastest and online MXFP8 adds roughly 5-7 tok/s at CC1.
DCP greater than one is primarily a KV-capacity choice. A4 versus A16 is an
accuracy/activation-path choice; A16 has the best KLD in this set.

<details>
<summary>Historical v14 full TP8 decode concurrency sweep retained by v15</summary>

Values are aggregate tok/s at context zero and `F8_DMA=0`. This is the older
full-sweep campaign retained by v15, not another repetition of the compact
hybrid-DCP table above. Small differences between the two sets are run/profile
variance; use the compact table for direct MTP-off mode selection and this
section for concurrency shape and MTP3 behavior.

#### Luke NVFP4 A4 original

| MTP | DCP | CC1 | CC2 | CC4 | CC8 | CC16 | CC32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 87.99 | 146.28 | 252.23 | 372.84 | 615.69 | 934.07 |
| 0 | 2 | 68.72 | 110.48 | 183.96 | 308.64 | 517.44 | 770.13 |
| 0 | 4 | 67.48 | 107.97 | 179.56 | 298.21 | 484.50 | 722.11 |
| 0 | 8 | 62.95 | 101.12 | 165.40 | 269.64 | 422.81 | 632.34 |
| 3 | 1 | 125.90 | 208.47 | 351.03 | 547.58 | 867.20 | 1,427.00 |
| 3 | 2 | 100.78 | 177.56 | 301.14 | 475.80 | 756.94 | 1,186.00 |
| 3 | 4 | 99.30 | 167.94 | 289.98 | 454.40 | 691.08 | 1,070.00 |
| 3 | 8 | 95.84 | 159.88 | 265.06 | 410.45 | 600.76 | 827.86 |

#### Luke NVFP4 A16 original

| MTP | DCP | CC1 | CC2 | CC4 | CC8 | CC16 | CC32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 86.56 | 142.02 | 235.72 | 324.45 | 616.97 | 932.72 |
| 0 | 2 | 67.65 | 104.27 | 169.32 | 273.86 | 517.73 | 782.30 |
| 0 | 4 | 66.31 | 102.63 | 165.97 | 264.11 | 482.12 | 722.91 |
| 0 | 8 | 61.77 | 96.06 | 154.14 | 241.11 | 422.22 | 635.66 |
| 3 | 1 | 119.62 | 182.71 | 350.45 | 553.79 | 843.71 | 1,345.00 |
| 3 | 2 | 90.69 | 154.19 | 304.37 | 481.78 | 735.46 | 1,134.00 |
| 3 | 4 | 89.44 | 150.41 | 296.73 | 453.77 | 685.09 | 1,030.00 |
| 3 | 8 | 90.48 | 152.68 | 263.93 | 413.46 | 584.04 | 793.75 |

#### Luke NVFP4 A4 online MXFP8

| MTP | DCP | CC1 | CC2 | CC4 | CC8 | CC16 | CC32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 94.96 | 152.04 | 254.11 | 378.62 | 641.43 | 953.24 |
| 0 | 2 | 72.99 | 113.89 | 187.98 | 312.84 | 532.22 | 802.18 |
| 0 | 4 | 71.82 | 111.84 | 184.29 | 300.67 | 498.74 | 739.96 |
| 0 | 8 | 66.51 | 104.36 | 170.11 | 270.55 | 429.92 | 638.57 |
| 3 | 1 | 129.37 | 211.16 | 359.88 | 557.84 | 902.52 | 1,461.00 |
| 3 | 2 | 104.96 | 179.20 | 307.62 | 486.24 | 779.92 | 1,225.00 |
| 3 | 4 | 100.28 | 172.18 | 286.85 | 452.26 | 715.04 | 1,085.00 |
| 3 | 8 | 98.23 | 164.85 | 270.51 | 413.74 | 611.94 | 842.56 |

#### Luke NVFP4 A16 online MXFP8

| MTP | DCP | CC1 | CC2 | CC4 | CC8 | CC16 | CC32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 93.30 | 148.15 | 237.03 | 326.64 | 632.38 | 954.52 |
| 0 | 2 | 71.67 | 107.49 | 173.35 | 274.93 | 526.95 | 789.85 |
| 0 | 4 | 70.44 | 106.09 | 169.77 | 267.38 | 496.04 | 735.91 |
| 0 | 8 | 65.36 | 99.20 | 156.32 | 242.38 | 430.73 | 643.02 |
| 3 | 1 | 120.69 | 184.78 | 358.23 | 559.65 | 873.98 | 1,378.00 |
| 3 | 2 | 92.47 | 159.22 | 310.52 | 481.54 | 752.39 | 1,163.00 |
| 3 | 4 | 95.56 | 154.51 | 293.70 | 460.27 | 692.15 | 1,051.00 |
| 3 | 8 | 92.51 | 146.57 | 266.80 | 420.01 | 597.00 | 803.96 |

</details>

### DCP1 MTP3 Coding Peaks

These three-position coding-task peaks are also inherited from v15. PR #94
cannot reach this decode workload.

| Variant | Mode | CC1 decode tok/s | CC32 decode tok/s | Coding peak mean | Median | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Luke original | A4 | 136.50 | 1,409.11 | 180.52 | 178.90 | 176.75 | 185.86 |
| Luke original | A16 | 126.60 | 1,319.22 | 166.55 | 169.51 | 160.15 | 171.27 |
| Luke online MXFP8 | A4 | 144.46 | 1,479.78 | 177.18 | 178.76 | 166.59 | 184.13 |
| Luke online MXFP8 | A16 | 130.07 | 1,386.32 | 166.26 | 166.44 | 158.90 | 172.76 |

## Standard Checkpoint Performance

The generated tables in this section combine inherited DCP1/decode/KLD values,
the already-completed two-run PR #94 measurements, and the selective v17
campaign. Three-run validation cells use the median; stable two-run cells and
the imported final-image A/B cells use their mean. The runner preserves the
v15 standalone-prefill client profile with estimated token targeting, so the
reported rates are normalized by the actual prompt-token count (typically
about 8.2k and 64.5k). The separate PR #94 A/B table below uses exact
8,192/65,536-token prompts.

The 64k rows are the primary regression and speedup signal. They were stable
within 0.6% across repeats. The 8k profile sits directly on the 8,192
token scheduler boundary and can switch launch/chunk shapes when estimated
tokenization differs by one token; medians are shown, but small 8k deltas
should not be interpreted as regressions. No v15/v16 DCP-prefill cell is
carried into a v17 column unless it is explicitly labeled as inherited.

| Profile | TP | DCP values | MTP | Max seqs | Graph | Batched tokens | Max model len | GMU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Standard NVFP4/MXFP4 | 8 | 1, 2, 4, 8 | 0 or 3 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| MXFP4 TP6 DCP1 | 6 | 1 | 0 | 16 | 64 | 2,048 | 128,000 | 0.957 |
| MXFP4 TP6 DCP>1 | 6 | 2, 3, 6 | 0 | 16 | 64 | 2,048 | 128,000 | 0.950 |

All v17 cells use `B12X_MLA_SPARSE`, B12X MoE, FP8 KV, InstantTensor
`BUFFERED`, and the helper's hybrid DCP policy. DCP1 values are inherited from
v15/v16 and retain the exact settings documented by those source campaigns.
The 42 newly measured cells are also published as a machine-readable
[TSV snapshot](../benchmarks/glm52-v17-selective-dcp-prefill.tsv).

<!-- BEGIN GLM52_V17_GENERATED_RESULTS -->

### Cross-Quant Decision Table

| Case | KLD mean +/- sd | DCP1 decode CC1 | DCP1 prefill 64k | v17 DCP4 prefill 64k | vs v15 DCP4 |
| --- | --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 0.10228 +/- 0.00634 | 87.99 | 6,257 | 3,740 | +8.2% |
| Luke NVFP4 A4 online MXFP8 | 0.10800 +/- 0.00697 | 94.96 | 6,351 | 3,793 | +8.6% |
| Luke NVFP4 A16 original | 0.05994 +/- 0.00129 | 86.56 | 5,849 | 3,576 | +7.2% |
| Luke NVFP4 A16 online MXFP8 | 0.06587 +/- 0.00253 | 93.30 | 5,941 | 3,618 | +7.8% |
| AMD MXFP4 experts A8 original | 0.08160 +/- 0.00432 | 88.72 | 6,307 | 3,792 | +8.6% |
| AMD MXFP4 experts A8 online MXFP8 | 0.08030 +/- 0.00309 | 94.03 | 6,364 | 3,812 | +9.1% |

### TP8 MTP0 DCP Prefill

#### Prefill 8k

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 6,557 | 4,477 | 3,613 | 2,462 |
| Luke NVFP4 A4 online MXFP8 | 6,681 | 4,892 | 3,730 | 2,398 |
| Luke NVFP4 A16 original | 6,140 | 4,633 | 3,552 | 2,378 |
| Luke NVFP4 A16 online MXFP8 | 6,239 | 4,340 | 3,291 | 2,348 |
| AMD MXFP4 experts A8 original | 6,698 | 4,565 | 3,760 | 2,462 |
| AMD MXFP4 experts A8 online MXFP8 | 6,731 | 5,006 | 3,738 | 2,339 |

#### Prefill 64k

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 6,257 | 4,972 | 3,740 | 2,474 |
| Luke NVFP4 A4 online MXFP8 | 6,351 | 4,975 | 3,793 | 2,477 |
| Luke NVFP4 A16 original | 5,849 | 4,642 | 3,576 | 2,388 |
| Luke NVFP4 A16 online MXFP8 | 5,941 | 4,779 | 3,618 | 2,422 |
| AMD MXFP4 experts A8 original | 6,307 | 4,981 | 3,792 | 2,476 |
| AMD MXFP4 experts A8 online MXFP8 | 6,364 | 5,103 | 3,812 | 2,491 |

#### 64k Change Versus v15

| Case | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- |
| Luke NVFP4 A4 original | +6.4% | +8.2% | +12.7% |
| Luke NVFP4 A4 online MXFP8 | +5.3% | +8.6% | +12.1% |
| Luke NVFP4 A16 original | +4.6% | +7.2% | +11.6% |
| Luke NVFP4 A16 online MXFP8 | +6.7% | +7.8% | +12.3% |
| AMD MXFP4 experts A8 original | +4.1% | +8.6% | +11.5% |
| AMD MXFP4 experts A8 online MXFP8 | +6.7% | +9.1% | +12.1% |

### TP8 MTP3 DCP Prefill

Cells are `8k / 64k` tok/s.

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 6,441 / 6,136 | 4,825 / 4,874 | 3,644 / 3,683 | 2,410 / 2,426 |
| Luke NVFP4 A4 online MXFP8 | 6,546 / 6,222 | 4,777 / 4,873 | 3,313 / 3,697 | 2,396 / 2,426 |
| Luke NVFP4 A16 original | 6,016 / 5,740 | 4,240 / 4,618 | 3,499 / 3,536 | 2,348 / 2,362 |
| Luke NVFP4 A16 online MXFP8 | 6,109 / 5,833 | 4,517 / 4,600 | 3,475 / 3,544 | 2,334 / 2,361 |

#### 64k Change Versus v15

| Case | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- |
| Luke NVFP4 A4 original | +6.7% | +8.6% | +12.5% |
| Luke NVFP4 A4 online MXFP8 | +5.5% | +8.0% | +12.0% |
| Luke NVFP4 A16 original | +6.5% | +8.2% | +12.5% |
| Luke NVFP4 A16 online MXFP8 | +4.7% | +7.6% | +11.7% |

### TP8 A4 MTP3 FP8 DMA

Cells are `8k / 64k` tok/s.

| Case | f8 | DCP1 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 0 | 6,441 / 6,136 | 4,825 / 4,874 | 3,644 / 3,683 | 2,410 / 2,426 |
| Luke NVFP4 A4 original | ag | 7,130 / 6,738 | 5,202 / 5,254 | 3,853 / 3,894 | 2,503 / 2,518 |
| Luke NVFP4 A4 original | ring | 7,912 / 7,435 | 5,602 / 5,673 | 4,075 / 4,119 | 2,589 / 2,607 |
| Luke NVFP4 A4 online MXFP8 | 0 | 6,546 / 6,222 | 4,777 / 4,873 | 3,313 / 3,697 | 2,396 / 2,426 |
| Luke NVFP4 A4 online MXFP8 | ag | 7,235 / 6,843 | 5,143 / 5,249 | 3,819 / 3,911 | 2,487 / 2,518 |
| Luke NVFP4 A4 online MXFP8 | ring | 8,035 / 7,564 | 5,560 / 5,684 | 4,042 / 4,137 | 2,569 / 2,609 |

#### 64k Change Versus v15

| Case | f8 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | 0 | +6.7% | +8.6% | +12.5% |
| Luke NVFP4 A4 original | ag | +7.4% | +9.0% | +13.1% |
| Luke NVFP4 A4 original | ring | +7.6% | +9.6% | +13.3% |
| Luke NVFP4 A4 online MXFP8 | 0 | +5.5% | +8.0% | +12.0% |
| Luke NVFP4 A4 online MXFP8 | ag | +5.8% | +8.6% | +12.4% |
| Luke NVFP4 A4 online MXFP8 | ring | +6.7% | +9.1% | +12.7% |

#### 64k DMA Gain Versus `f8=0` On v17

| Case | f8 | DCP2 | DCP4 | DCP8 |
| --- | --- | --- | --- | --- |
| Luke NVFP4 A4 original | ag | +7.8% | +5.7% | +3.8% |
| Luke NVFP4 A4 original | ring | +16.4% | +11.8% | +7.5% |
| Luke NVFP4 A4 online MXFP8 | ag | +7.7% | +5.8% | +3.8% |
| Luke NVFP4 A4 online MXFP8 | ring | +16.6% | +11.9% | +7.5% |

### TP6 MTP0

Decode is inherited from v16; PR #94 does not change decode.

| Case | DCP | Decode CC1 | v17 prefill 8k | v17 prefill 64k | 64k vs v16 |
| --- | --- | --- | --- | --- | --- |
| AMD MXFP4 experts A8 original | 1 | 75.75 | 5,139 | 5,280 | inherited |
| AMD MXFP4 experts A8 original | 2 | 61.98 | 3,976 | 3,966 | +3.0% |
| AMD MXFP4 experts A8 original | 3 | 59.23 | 3,299 | 3,326 | +3.6% |
| AMD MXFP4 experts A8 original | 6 | 45.88 | 2,275 | 2,293 | +7.4% |
| AMD MXFP4 experts A8 online MXFP8 | 1 | 82.96 | 4,906 | 5,244 | inherited |
| AMD MXFP4 experts A8 online MXFP8 | 2 | 66.64 | 3,718 | 4,068 | +4.9% |
| AMD MXFP4 experts A8 online MXFP8 | 3 | 63.82 | 3,308 | 3,361 | +5.8% |
| AMD MXFP4 experts A8 online MXFP8 | 6 | 50.05 | 2,296 | 2,330 | +9.2% |

<!-- END GLM52_V17_GENERATED_RESULTS -->

### How To Choose

- **Accuracy first:** Luke NVFP4 A16 original has the best corrected-reference
  KLD (`0.05994 +/- 0.00129`). Its DCP1 decode is `86.56 tok/s`; DCP4 reaches
  `3,576 tok/s` at 64k with `f8=0`.
- **Accuracy/speed balance:** Luke NVFP4 A16 online MXFP8 raises DCP1 decode to
  `93.30 tok/s` with KLD `0.06587 +/- 0.00253`; its DCP4 64k prefill is
  `3,618 tok/s`.
- **Fastest tested `f8=0` standard case:** AMD MXFP4 experts A8 online MXFP8
  reaches `94.03 tok/s` DCP1 decode and `3,812 tok/s` DCP4 64k prefill, with
  KLD `0.08030 +/- 0.00309`.
- **Prefill first:** `ring` is the fastest tested DMA mode. On Luke NVFP4 A4
  original with MTP3, DCP4 rises from `3,683` to `4,119 tok/s` at 64k
  (`+11.8%` versus `f8=0`). DMA mode does not accelerate decode.

Every affected 64k cell improved over its v15/v16 source value. DCP1 remains
the single-request decode choice; DCP2/4/8 and TP6 DCP2/3/6 trade per-request
speed for proportionally larger effective KV capacity. The PR #94 gain grows
with DCP because it removes more DCP gather/merge/scatter allocation and copy
overhead.

One TP6 DCP6 online server boot hit an asynchronous CUDA illegal-address error
during FULL graph warmup after the preceding rapid sweep. No benchmark ran on
that process. An unchanged clean-container retry completed and produced the
stable three-run `2,330 tok/s` 64k result above; the failed startup log remains
in the raw campaign directory.

## Hybrid NF3 Checkpoint Layout

The tested checkpoint revision is
`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at
`68babde27a97a4c980c2494e830dd424975cd5a3`.

- Routed experts are selected per layer by the checkpoint's
  `hybrid_bit_map`: the kept tier is NVFP4 and the remaining experts use the
  checkpoint's packed NF3 format.
- Both routed-expert tiers run through the B12X W4A16 path. `A16` therefore
  describes the expert activation path; it does not rewrite the checkpoint's
  NVFP4 or NF3 weights.
- Eligible BF16 non-expert linear weights are converted once at load time to
  MXFP8. Existing NVFP4/NF3 routed experts are not requantized.
- Shared experts remain on their checkpoint path unless an explicit
  `shared_experts` online-quantization rule is supplied. The v17 preset does
  not supply one.
- `kv_b_proj` is explicitly excluded from online MXFP8 conversion.
- MLA KV cache uses `nvfp4_ds_mla`.

The helper resolves this configuration as:

```text
QUANTIZATION=nvfp4_nf3_hybrid
MOE_MODE=a16
ONLINE_QUANT=mxfp8
KV_CACHE_DTYPE=nvfp4_ds_mla
LOAD_FORMAT=instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

## Start The Hybrid Server

The helper is already inside the image at
`/usr/local/bin/serve-glm52-hybrid-v17.sh`. No host-side launch script is
needed. This is the exact tested TP4/DCP4, MTP-off profile:

```bash
docker run -d --name glm52-v17-hybrid \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e GPUS=0,1,2,3 -e PORT=8000 -e DCP=4 -e MTP=0 \
  -e MAX_NUM_SEQS=8 -e GRAPH=64 \
  -e MAX_MODEL_LEN=131072 -e MAX_BATCHED_TOKENS=3072 \
  -e GPU_MEMORY_UTILIZATION=0.96 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/models:/root/models:ro \
  -v /root/.cache/vllm-glm52-v17:/cache \
  -v /root/vllm/tmp/glm52-v17:/container-tmp \
  voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715
```

To use a local checkpoint, add:

```bash
-e MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
```

The maintained minimal Compose file is
[`examples/docker-compose-glm52-hybrid-v17.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/ee75ffa239565504cc2b86735cd91a65cf711501/examples/docker-compose-glm52-hybrid-v17.yml).
It exposes only the deployment envelope; the quantization, backend, loader,
and exact 78-character `index_topk_pattern` stay in the image helper.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
GPUS=0,1,2,3 PORT=8000 DCP=4 MTP=0 \
MAX_NUM_SEQS=8 GRAPH=64 MAX_MODEL_LEN=131072 \
MAX_BATCHED_TOKENS=3072 GPU_MEMORY_UTILIZATION=0.96 \
  docker compose -f examples/docker-compose-glm52-hybrid-v17.yml up -d
```

User-facing controls:

| Environment | Tested value | Meaning |
|---|---:|---|
| `GPUS` | `0,1,2,3` | Four physical GPUs used by TP4 |
| `PORT` | `8000` | OpenAI-compatible API port |
| `DCP` | `1`, `2`, or `4` | Decode-context parallel size |
| `DCP_PREFILL_WORKSPACE` | `auto` | Enables validated workspace paths by default; use `0` for the baseline path |
| `MTP` | `0` | This page validates MTP off |
| `MAX_NUM_SEQS` | `8` | Scheduler concurrency ceiling |
| `GRAPH` | `64` | Maximum CUDA graph capture size |
| `MAX_MODEL_LEN` | `131072` | Tested maximum request length |
| `MAX_BATCHED_TOKENS` | `3072` | Prefill scheduler budget and DCP4 optimization profile |
| `GPU_MEMORY_UTILIZATION` | `0.96` | Per-GPU memory target |

## Clean-Image Performance

Profile: TP4, MTP off, A16, online MXFP8 non-expert linear weights, NVFP4
KV, `MAX_NUM_SEQS=8`, graph 64, and `MAX_BATCHED_TOKENS=3072`. Three model
instances were loaded first on GPU groups 0-3, 4-7, and 8-11. Every endpoint
was ready, followed by a 30-second settle period, before clients were run
serially. No model loaded while a benchmark was active.

| DCP | KV cache tokens | Decode CC1 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---:|---:|---:|---:|---:|
| 1 | 185,216 | 49.9 | 4,469 | 4,163 |
| 2 | 384,000 | 43.9 | 3,113 | 3,031 |
| 4 | 768,000 | 44.7 | 2,378 | 2,341 |

DCP2 decode was repeated and produced 43.9 tok/s both times. An earlier DCP2
run on GPU 0-3 measured 44.9 tok/s, while the clean release run used GPU 8-11.
PR #94's runtime optimization is strictly gated out for TP4/DCP1, TP4/DCP2,
and decode, so the one-token difference is treated as GPU-group/run variance.

### DCP4 Prefill Improvement

| Metric | Current stack before workspace reuse | Final clean v17 | Change |
|---|---:|---:|---:|
| Prefill 8k | 2,172 tok/s | 2,378 tok/s | +9.5% |
| Prefill 64k | 2,144 tok/s | 2,341 tok/s | +9.2% |
| Decode CC1 | 44.9 tok/s | 44.7 tok/s | -0.4%, noise |
| KV cache | 768,000 tokens | 768,000 tokens | unchanged |

The prefill path projects each sparse-MLA partial output from 512 to 256 before
the LSE-corrected reduce-scatter. For eligible eager TP4/DCP4 prefills it also
borrows existing B12X query and scratch workspaces for gather, projection, and
caller-owned reduce-scatter output. The gate requires B12X sparse MLA, a
validated TP/DCP topology, AG/RS, non-DBO eager execution, and at least 1,025
active rows. Other shapes retain the existing path.

The image helper enables this gate automatically for TP4/DCP4, TP6/DCP2/3/6,
and TP8/DCP2/4/8. `DCP_PREFILL_WORKSPACE=0` disables it for A/B testing;
`DCP_PREFILL_WORKSPACE=1` requests it explicitly but does not bypass
source-level topology, shape, capture, and backend safety checks.

### Generalized TP6/TP8 Results

These A/B runs validate the generalized PR #94 implementation. All rows use
MTP off, `F8_DMA=0`, InstantTensor `BUFFERED`, hybrid DCP (`a2a` for small
rows and `ag_rs` for large rows), exact 8,192/65,536-token prompts, and two
runs per side. All servers were loaded before benchmarking and clients ran
serially.

TP8 used `lukealonso/GLM-5.2-NVFP4`, A16, `MAX_BATCHED_TOKENS=8192`,
`MAX_NUM_SEQS=32`, and graph 128.

| Topology | Baseline 8k | Optimized 8k | Change | Baseline 64k | Optimized 64k | Change |
|---|---:|---:|---:|---:|---:|---:|
| TP8/DCP2 | 4,476 | 4,633 | +3.51% | 4,481.5 | 4,641.5 | +3.57% |
| TP8/DCP4 | 3,312 | 3,551.5 | +7.23% | 3,328.5 | 3,576 | +7.44% |
| TP8/DCP8 | 2,150.5 | 2,378.5 | +10.60% | 2,157.5 | 2,388 | +10.68% |

TP6 used `/root/models/GLM-5.2-BF16-AMDMXFP4experts`, forced A8,
`MAX_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=16`, and graph 64.

TP6 relies on FF's automatic B12X virtual-TP layout. Before vLLM's normal
divisibility check, it pads attention heads 64 -> 66, MoE intermediate width
2048 -> 2112, and vocabulary 129280 -> 129408; checkpoint tails are
zero-filled by the loader. There is no user-facing virtual-sharding flag. A
`64 heads must be divisible by TP 6` error with MTP off means this B12X
configuration step did not run, normally because the wrong image/backend was
used. With MTP enabled, the same error on the July 14 image came from the
separate unpadded draft config and is fixed by the July 15 image above.

| Topology | Baseline 8k | Optimized 8k | Change | Baseline 64k | Optimized 64k | Change |
|---|---:|---:|---:|---:|---:|---:|
| TP6/DCP2 | 3,912 | 3,975.5 | +1.62% | 3,912.5 | 3,966.5 | +1.38% |
| TP6/DCP3 | 3,172.5 | 3,299 | +3.99% | 3,200.5 | 3,326.5 | +3.94% |
| TP6/DCP6 | 2,119 | 2,275 | +7.36% | 2,132.5 | 2,293 | +7.53% |

The fixed-half comparison is conservative because GPUs 0-7 were faster than
GPUs 8-15. Cross-over runs on both identical GPU groups measured the intrinsic
TP8/DCP2 gain at 4.56-4.80% and TP6/DCP2 at 2.92-3.31%. Logs confirmed the
borrowed-workspace path on every optimized topology. Decode and KV capacity
are unchanged because the optimization is confined to eager prefill.

## v1.3 Investigation

The workspace idea was adapted from the fast647 implementation in
[`davidsyoung/vllm-glm52` v1.3](https://github.com/davidsyoung/vllm-glm52/tree/v1.3).
An isolated A/B on that exact v1.3 stack measured:

| v1.3 mode | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---:|---:|
| Workspace disabled | 2,414 | 2,373 |
| Workspace enabled | 2,516 | 2,474 |
| Change | +4.2% | +4.2% |

The complete v1.3 overlay was not adopted. It includes an older vLLM base,
compact-KV/loading assumptions, and deployment tuning that do not match the
Fathomless Firmament stack. Current FF already contains the useful guarded
small-row B12X A2A decode transport and paged-indexer carry-fold work. v17
ports only the reusable sparse-prefill workspace concept and keeps explicit
fallbacks for decode, CUDA graphs, non-B12X backends, small prefills, and
unsupported shapes.

## Correctness Fixes

### NF3 preplanned tile binding

NF3 weights are packed for a specific tile-N geometry. The previous custom-op
boundary could silently rebuild a preplanned `(64,256,64,256)` launch as
`(128,128,64,256)`. The launch then interpreted the packed NF3 layout with the
wrong tile geometry and could produce garbled output. B12X now carries the
planned tile K/N values through the custom op and reuses that exact geometry
during compile and cache lookup.

### B12X DCP startup consensus

The first clean source image exposed a startup deadlock after every rank had
successfully created its B12X DCP pool. The four ranks then blocked in a Gloo
error-consensus reduction even though `init_error=None` everywhere. v17 keeps
that four-byte status reduction on the same NCCL DCP exchange group used by
the immediately preceding IPC-handle exchange. B12X A2A remains enabled; no
backend, stream, graph, or runtime feature is disabled.

### TP6 W4A8 cooperative launch

v16 fixed the TP6/DCP3 forced-A8 graph-capture failure at local `m=9`. The
dynamic W4A8 kernel selected a 376-CTA resident-grid launch while a BF16 shared
expert GEMM could occupy SM resources on an auxiliary stream. Some CTAs then
waited at software barriers for CTAs that could not become resident, ending in
an illegal memory access/Xid 31.

B12X now uses a CUDA cooperative launch and a CuTe-safe CTA-leader predicate
for this kernel. The fix preserves overlap, B12X backends, CUDA graphs, DCP,
and forced A8; it is not a workaround that serializes streams. TP6 also keeps
the automatic virtual layout described above and the zero-padded expert-shard
handoff from vLLM PR #80.

Validation:

- 26 focused PR #94 tests passed in the final runtime environment.
- Ruff check and format check passed.
- DCP2 and DCP4 booted and warmed B12X PCIe DCP collective signatures.
- TP4/DCP4, TP6/DCP2/3/6, and TP8/DCP2/4/8 all served exact 8k/64k requests;
  every optimized log confirmed the borrowed-workspace path was active.
- A 30,017-token DCP4 generation produced coherent Python code with zero CJK
  characters; TTFT was 8.26 seconds.
- A deterministic long-prompt baseline/optimized comparison produced exactly
  the same output.
- The running containers had only model, Hugging Face, JIT-cache, and temporary
  mounts. There were no source or site-packages overlays.

## Reproduce The Benchmarks

### Standard TP8/TP6 Matrix

The selective runner verifies the immutable image ID, starts every server
through the image helper, confirms InstantTensor `BUFFERED`, waits for both
paired endpoints, settles for 30 seconds, and then benchmarks them serially.
It also requires the borrowed-workspace runtime marker before accepting a
result. No model is loaded while a benchmark client is active.

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
cd rtx6kpro

RESULT_ROOT=/root/bench-results/glm52-v17-dcp-prefill \
  ./scripts/bench-glm52-v17-selective-dcp-prefill.sh all

./scripts/render-glm52-v17-results.py \
  /root/bench-results/glm52-v17-dcp-prefill \
  --output /tmp/glm52-v17-results.md
```

The default manifest skips the six final-image cells already present in the
PR #94 A/B result root: TP8 A16-original DCP2/4/8 and TP6 A8-original
DCP2/3/6. Set `INCLUDE_IMPORTED=1` to remeasure them too and create the full
dataset from scratch. Individual modes are `tp8-mtp0`, `tp8-mtp3`,
`tp8-dma`, and `tp6`. Completed cells carry a marker and are skipped when the
same `RESULT_ROOT` is resumed.

`TOKEN_TARGETING=estimate` is explicit in the runner and reproduces the v15
standalone-prefill profile used by the generated tables. Set
`TOKEN_TARGETING=exact` only when reproducing the separate exact-token PR #94
A/B experiment. If a server fails before readiness, rerun the same command;
completed cells are skipped and the failed cell is the only one started again.

To repeat only selected cells, pass quoted five-field manifests. The runner
still pairs adjacent entries, waits for both servers, and measures them
serially:

```bash
RESULT_ROOT=/root/bench-results/glm52-v17-dcp-prefill \
  ./scripts/bench-glm52-v17-selective-dcp-prefill.sh configs \
  "nvfp4-a16-online-mxfp8 8 2 0 0" \
  "mxfp4-a8-online-mxfp8 8 2 0 0"
```

### TP4 NVFP4/NF3 Hybrid

```bash
MODEL=/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
RESULT_ROOT=/root/bench-results/glm52-v17-hybrid-reproduction \
  ./scripts/bench-glm52-v17-hybrid-tp4.sh
```

Defaults use GPU 0-3 for DCP4, 4-7 for DCP1, and 8-11 for DCP2. To reproduce
only DCP4 on four GPUs:

```bash
DCP_VALUES=4 GPU_DCP4=0,1,2,3 \
RESULT_ROOT=/root/bench-results/glm52-v17-hybrid-dcp4 \
  ./scripts/bench-glm52-v17-hybrid-tp4.sh
```

Published raw local result roots:

```text
/root/bench-results/glm52-v17-selective-dcp-prefill-20260714
/root/bench-results/glm52-hybrid-v17-tp4-20260714/final-clean-source
/root/bench-results/pr94-generalization-20260714
```

This v17 campaign does not introduce a new KLD reference campaign. Use the
corrected BF16-reference procedure documented on the
[v15 page](glm5.2_v15.md#kld-keypoint-rerun) when comparing checkpoint
quality.
