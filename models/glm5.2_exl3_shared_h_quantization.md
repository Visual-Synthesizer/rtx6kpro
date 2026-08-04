# GLM-5.2 EXL3 shared-H quantization

This page documents the reproducible procedure for creating the calibrated
`shared_h_v1` EXL3 artifact qualified by Gilded Gnosis r28. It is a new encode
from the BF16 model, not a post-processing pass over an existing EXL3 model.

## Purpose

Legacy GLM-5.2 EXL3 checkpoints store three hidden-sized rotation vectors for
every routed expert. The calibrated shared-H recipe stores these vectors once
per MoE layer and TP rank:

- `gate_proj.suh` shared across 256 experts;
- `up_proj.suh` shared across 256 experts;
- `down_proj.svh` shared across 256 experts.

The intermediate-side vectors remain expert-local. For 75 GLM-5.2 MoE layers
this removes exactly 705,024,000 persistent bytes, or 672.36 MiB, from every
GPU at MTP0. Loading the MTP layer raises the exact saving to 714,424,320 bytes,
or 681.33 MiB/GPU. The loader keeps each shared row physically shaped `[1, H]`;
it does not expand it back to 256 rows.

## Compatibility contract

New artifacts must declare this metadata:

```json
{
  "rotation_layout": "shared_h_v1",
  "shared_h_tensor_schema": "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
}
```

A checkpoint without `rotation_layout` is interpreted as legacy
`per_expert_v1` and loads unchanged. A loader that does not recognize
`shared_h_v1` must reject the new artifact. Existing EXL3 checkpoints cannot be
deduplicated losslessly because their expert-local H-side vectors are not
identical.

## Published checkpoint

The complete qualified artifact is:

```text
willfalco/GLM-5.2-EXL3-TR3-3.42bpw
revision: ae68c65947efa90bea37308e15421872f124c46d
```

All 79 model shard hashes were verified against that immutable revision. Its
MoE partitions are 206 K3 + 50 K4 in layer 3, 148 K3 + 108 K4 in layers 4-77,
and 256 K3 in layer 78. Gilded Gnosis r28 detects `shared_h_v1` automatically;
there is no serving-time layout switch.

## Reviewed implementation

The complete recipe, exact patches, input hashes, and tests are in
[local-inference-lab/kquant PR #1](https://github.com/local-inference-lab/kquant/pull/1).
The matching backward-compatible runtime is
[local-inference-lab/vllm PR #228](https://github.com/local-inference-lab/vllm/pull/228),
with the dynamic mixed-Trellis ABI in
[SparkInfer PR #117](https://github.com/local-inference-lab/sparkinfer/pull/117).
Both are included in the immutable r28 composition. PR #228 supersedes the
earlier integration role of #225/#226.

The preparation script accepts only the published `calibration_encoder` bundle
from
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/calibration_encoder).
It verifies these source files before applying the reviewable patches:

| Input | SHA-256 |
|---|---|
| `encode_tr3_v31.py` | `e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032` |
| `encode_b300.py` | `f378817b212dc9f4a8c9dc049803542e7c91748283f6e8ec1ebe0427be96aaf1` |
| `calibration/reap_recall_calib.jsonl` | `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4` |

Prepare a clean encoder directory:

```bash
git clone https://github.com/local-inference-lab/kquant.git
cd kquant
git fetch origin pull/1/head:glm52-shared-h
git checkout glm52-shared-h

export ORIGINAL_BUNDLE=/workspace/original/calibration_encoder
export SHARED_ENCODER=/workspace/tr3-shared-h

python3 recipes/glm52_exl3_shared_h/prepare_shared_h_encoder.py \
  --bundle "$ORIGINAL_BUNDLE" \
  --output "$SHARED_ENCODER"
```

The generated encoder hashes must be:

| Output | SHA-256 |
|---|---|
| `encode_tr3_v31.py` | `400c0df1c95c81c30a2ce31e060f0445a798fd29ad9339923d4e02e3ee40f6f7` |
| `encode_b300.py` | `b41f9397a1754e67f41b2356db413b5da18228bcd3961c97023b4e1cabd01010` |

## Why two passes are required

The shared magnitude depends on each expert's BF16 weights, calibrated Hessian,
and automatic output-scale choice. Each layer is therefore encoded in two
passes:

1. Run normal calibrated pre-regularization for all experts and form a signed
   geometric-mean H-side profile for each projection and TP rank.
2. Rerun the original calibrated LDLQ/Trellis encode with that profile forced
   on the H side.

Gate/up keep expert-local output vectors and move each expert's scalar
`g_scale` from shared `SU` to expert-local `SV`. Down retains expert-local `SU`
and its standard scale placement. Tests verify the algebraic equivalence.

## Full conversion

Use the same B300 environment as the source recipe: exllamav3 0.0.43, CUDA
12.9, [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) in BF16,
sufficient tmpfs for captures, and about 0.5 TB of assembly scratch. The
reference process uses eight B300 GPUs and emits a TP4 artifact.

```bash
export SCRIPT_DIR="$SHARED_ENCODER"
export WORK_ROOT=/workspace/tr3-shared-h-work
export BF16_SRC=/workspace/bf16
export OWNER_CORPUS="$SHARED_ENCODER/calibration/reap_recall_calib.jsonl"
export BASE_ENCODER_PY="$SHARED_ENCODER/encode_tr3_v31.py"
export OUT_DIR=/workspace/output/GLM-5.2-EXL3-TR3-Shared-H
export CUDA_HOME=/usr/local/cuda-12.9

"$SHARED_ENCODER/convert_b300.sh" preflight
"$SHARED_ENCODER/convert_b300.sh" ext
"$SHARED_ENCODER/convert_b300.sh" plan

for window in 3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-77; do
  LAYERS="$window" "$SHARED_ENCODER/convert_b300.sh" capture-window
  LAYERS="$window" "$SHARED_ENCODER/convert_b300.sh" encode-window
done

"$SHARED_ENCODER/convert_b300.sh" assemble
```

`encode-window` creates and verifies the shared profile before LDLQ. No extra
quantization switch is required.

## Release gates

Before publishing a full checkpoint, require all of the following:

1. Source-payload audit and `MANIFEST.sha256` pass.
2. Every MoE layer contains 9,228 EXL3 tensors.
3. Teacher-forced KLD is measured against the standard GLM BF16 logits.
4. Both legacy and shared-H checkpoints boot in the release image.
5. TP4 decode, prefill, CUDA-graph replay, and tool-call checks pass.

The layer-40 POC measured no CUDA-graph latency regression, weight NMSE changing
by +0.00175% relative, and activation NMSE by -0.057% relative. The complete
3.42 checkpoint then passed all five gates in r28:

| Gate | Result |
|---|---:|
| Shard integrity | 79/79 |
| MTP0 shared-H saving | 672.36 MiB/GPU |
| MTP3 shared-H saving | 681.33 MiB/GPU |
| TP4/DCP1/MTP0 decode | 53.25 / 53.33 tok/s |
| TP4/DCP1/MTP0 prefill | 3,586.81 / 3,386.11 tok/s at 8k / 64k |
| TP4/DCP1/MTP3 decode | 113.40 tok/s |
| TP4/DCP4/MTP3 correctness | 24/24 at c8; 32/32 at c16 |
| Checkpoint-only KLD | 0.074145973 |
| Default K6 + NVFP4 KV KLD | 0.108828284 |

The two KLD rows use the same BF16 reference and 2,047 positions. The
checkpoint-only row uses FP8 KV to match the reference capture. A same-NVFP4
control shows that online K6 itself adds only 0.000856839 mean KLD; most of the
headline difference is the production NVFP4 KV format. Full commands and the
exact token input are on the
[KLD evaluation page](../benchmarks/glm52-kld-evaluation.md).

Audit a downloaded checkpoint before serving:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
python3 scripts/audit_glm52_exl3_shared_h.py \
  /root/models/GLM-5.2-EXL3-TR3-3.42bpw
```

The audit fails closed on missing metadata, malformed one-row H tensors,
unexpected tier counts, or a mismatch between declared and physical savings.
