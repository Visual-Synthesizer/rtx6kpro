# GLM-5.2 v14 NVFP4 / Online MXFP8 Overlay

This page documents the July 2026 GLM-5.2 serving recipe for RTX 6000 Pro
Blackwell. The main checkpoint is Luke Alonso's NVFP4 model, with optional
online conversion of selected BF16 linear weights to MXFP8 at model load time.

The goal of this version is reproducibility: each result below points back to
the local benchmark or KLD result directory that produced it. Values marked
`TODO measure` were not found in the historical run logs and should be filled
by rerunning the scripts listed at the end.

## Image And Model

Online MXFP8 overlay image:

```text
voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-pr74-mxfp8overlay-cu132-20260705
```

Clean Luke NVFP4 comparison image:

```text
voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-glm52-dcp-fp8nvfp4fix-cu132-20260705
```

Model:

```text
lukealonso/GLM-5.2-NVFP4
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

Runtime defaults:

| Setting | Value |
|---|---|
| TP | `8` |
| DCP | `1` for these measurements |
| MTP | `0` |
| Quantization | `modelopt_fp4` |
| Attention | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| KV cache | `fp8` |
| Max num seqs / graph | benchmark script defaults |
| Index-cache pattern | model override from existing GLM-5.2 scripts |

## A4 vs A16

For the Luke NVFP4 checkpoint, `A4` means the checkpoint-native NVFP4 MoE path:

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=0
```

`A16` means forcing B12X MoE onto the W4A16 path: BF16 activations with the
NVFP4/W4 expert weights. The expected boot log says force-A16 is enabled.

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=1
export B12X_W4A16_TC_DECODE=1
```

`A8` force is not the relevant mode for this NVFP4 checkpoint. It was useful for
the separate MXFP4/FP4 experiments, but do not use it as the NVFP4 comparison
axis here.

## Default: No Online MXFP8 Conversion

The default load path keeps BF16 layers BF16. Do not pass `--quantization-config`
or pass an empty config. The model is still loaded as `--quantization modelopt_fp4`;
this is the ModelOpt NVFP4 checkpoint path, not online dense MXFP8 conversion.

The `f8=0` rows, meaning both `ag` and `ring` FP8 DMA modes disabled, still need
to be measured for the clean Luke checkpoint.

| Mode | f8 | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | Prefill KLD mean +/- sd |
|---|---:|---:|---:|---:|---:|---:|---|
| A4 | `0` | TODO measure | TODO measure | TODO measure | TODO measure | TODO measure | TODO measure |
| A4 | `ag` | 87.07 | 87.60 | 7055 | 6771 | 6360 | 0.11195 +/- 0.00256 |
| A4 | `ring` | 88.32 | 88.78 | 7975 | 7580 | 7060 | 0.11676 +/- 0.00094 |
| A16 | `0` | TODO measure | TODO measure | TODO measure | TODO measure | TODO measure | TODO measure |
| A16 | `ag` | 85.80 | 86.36 | 6552 | 6277 | 5917 | 0.06977 +/- 0.00106 |
| A16 | `ring` | 86.88 | 87.40 | 7336 | 6974 | 6516 | 0.09048 +/- 0.00923 |

Source directories:

```text
/root/bench-results/glm52-luke-nvfp4-dma-compare-20260705T155345Z
/root/kld/glm52_luke_nvfp4_dma_compare_20260705_161345
```

## Online MXFP8 Conversion

Online conversion is enabled through `--quantization-config`. The current
overlay PR path accepts JSON. This converts BF16 linear weights to MXFP8 as the
model is loaded; it does not require a separate offline checkpoint.

```bash
--quantization modelopt_fp4 \
--quantization-config '{"linear":{"weight":"mxfp8"}}'
```

This is intended to reproduce the useful behavior of the static
`GLM-5.2-MXFP8dense-NVFP4experts-BF16shared` checkpoint while starting from
Luke's original NVFP4 checkpoint. `indexer.wk` and similar explicitly ignored
layers should remain BF16; forcing `kv_b_proj` to stay BF16 was tried and did
not improve KLD in the historical 5-run check.

Direct online measurements currently exist for A16 only:

| Mode | f8 | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | Prefill KLD mean +/- sd |
|---|---:|---:|---:|---:|---:|---:|---|
| A16 online MXFP8 | `0` | 93.80 | 94.55 | 6173 | 5925 | 5604 | 0.07278 +/- 0.00266 |
| A16 online MXFP8 | `ag` | 91.87 | 92.49 | 6656 | 6366 | 5995 | 0.07467 +/- 0.00311 |
| A16 online MXFP8 | `ring` | 94.13 | 94.64 | 7410 | 7042 | 6579 | 0.08581 +/- 0.01129 |

The `ring` decode value uses the later decode-only rerun against the same mode.
The first scripted sweep produced `85.98` decode agg tok/s and `86.48` coding
peak tok/s, but that row was discarded as an invalid scripted-run outlier. The
prefill and KLD values are still from the full sweep because the issue was only
observed in the decode part of that run.

Source directories:

```text
/root/bench-results/glm52-luke-nvfp4-pr74-mxfp8overlay-bf16shared-a16-dma-compare-20260705T205941Z
/root/bench-results/glm52-pr74-mxfp8overlay-a16-ring-live-decode-rerun-20260705T220631Z
/root/kld/glm52_luke_nvfp4_pr74_mxfp8overlay_bf16shared_a16_dma_compare_20260705_205941
/root/kld/glm52_luke_nvfp4_pr74_mxfp8overlay_bf16shared_a16_ring_kld5_verify_20260705_225321
```

A4 online MXFP8 still needs a direct run from Luke's checkpoint. The closest
historical proxy is the offline-equivalent static checkpoint:
`/root/models/GLM-5.2-MXFP8dense-NVFP4experts-BF16shared`.

| Mode | f8 | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | Prefill KLD mean +/- sd |
|---|---:|---:|---:|---:|---:|---:|---|
| A4 static MXFP8 dense | `ag` | 93.56 | 94.25 | 7222 | 6893 | 6461 | 0.10990 +/- 0.00716 |
| A4 static MXFP8 dense | `ring` | 95.52 | 96.09 | 8139 | 7725 | 7185 | 0.11847 +/- 0.00471 |
| A16 static MXFP8 dense | `ag` | 92.13 | 92.69 | 6647 | 6384 | 6012 | 0.07280 +/- 0.00331 |
| A16 static MXFP8 dense | `ring` | 93.71 | 94.40 | 7475 | 7101 | 6632 | 0.07822 +/- 0.00609 |

The old A16 static `ring` KLD sample above was only 3 runs. A later 5-run rerun
measured `0.08898 +/- 0.01335`, so do not interpret the apparent online/static
KLD difference as real without more samples.

Source directories:

```text
/root/bench-results/glm52-mxfp8dense-nvfp4experts-bf16shared-dma-compare-20260705T182500Z
/root/kld/glm52_mxfp8dense_nvfp4experts_bf16shared_dma_compare_20260705_182500
/root/kld/glm52_offline_bf16shared_a16_ring_kld5_rerun_20260706_001007
```

## What f8=0/ag/ring Means

`f8` is the FP8 PCIe DMA allreduce selector used by the B12X PCIe DMA
communicator. vLLM reads:

```bash
export VLLM_PCIE_DMA_FP8=0      # or ag/ring
export B12X_PCIE_DMA_FP8=0      # same value, kept for B12X-side fallback
```

The code passes the selected string into `b12x.distributed.PCIeDmaAllReduce` as
`fp8=<value>`.

| f8 value | Meaning in these builds | Observed effect |
|---|---|---|
| `0` | Disable FP8-compressed PCIe DMA payloads. B12X PCIe allreduce can still be enabled, but not this FP8 DMA mode. | Best A16 online decode in the current direct run; lower prefill than `ag`/`ring`. |
| `ag` | Enable the all-gather style FP8 DMA mode. | Improves prefill versus `0`; decode was slightly below `0` for A16 online; KLD was slightly higher than `0`. |
| `ring` | Enable ring FP8 DMA mode. | Best prefill in both clean and MXFP8 logs; after discarding the bad scripted decode outlier, decode is also in the normal `94 tok/s` range; KLD is more variable. |

For the A16 online MXFP8 direct run, the measured impact was:

| f8 | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | KLD mean +/- sd |
|---|---:|---:|---:|---:|---:|---|
| `0` | 93.80 | 94.55 | 6173 | 5925 | 5604 | 0.07278 +/- 0.00266 |
| `ag` | 91.87 | 92.49 | 6656 | 6366 | 5995 | 0.07467 +/- 0.00311 |
| `ring` | 94.13 | 94.64 | 7410 | 7042 | 6579 | 0.08581 +/- 0.01129 |

The first automated sweep measured `ring` decode as `85.98` decode agg tok/s and
`86.48` coding peak tok/s. That number was later invalidated by the decode-only
rerun of the same already-running `ring` server, which measured `94.13` decode
agg tok/s and `94.64` coding peak tok/s. Use the rerun value above; treat the
old low decode row as a bad scripted-run outlier.

## Reproduction Commands

Clean Luke NVFP4 DMA comparison:

```bash
cd /root/vllm/dspark
bash ./bench_luke_nvfp4_dma_compare.sh
```

Online MXFP8 overlay DMA comparison:

```bash
cd /root/vllm/dspark
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-pr74-mxfp8overlay-cu132-20260705 \
MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 \
QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
bash ./bench_mxfp8_overlay_dma_compare.sh
```

To force A16 in ad-hoc launches:

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=1
export B12X_W4A16_TC_DECODE=1
```

To use checkpoint-native A4 in ad-hoc launches:

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=0
export B12X_W4A16_TC_DECODE=1
```

To switch FP8 DMA allreduce mode:

```bash
export VLLM_PCIE_DMA_FP8=0      # or ag/ring
export B12X_PCIE_DMA_FP8=0      # keep same as VLLM_PCIE_DMA_FP8
```

KLD references used by these historical runs:

```text
/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618
context_length=2048
stride=512
max_windows=1
```

## Open TODOs

| Item | Why |
|---|---|
| Clean Luke NVFP4 `f8=0` for A4 and A16 | Required to complete the default/no-online-quant table. |
| Direct online MXFP8 A4 for `f8=0/ag/ring` | Current A4 numbers are from the offline-equivalent static checkpoint, not online conversion from Luke's checkpoint. |
| More KLD samples for `ring` modes | Existing results show high variance; 3-run samples are not enough to rank close variants. |
