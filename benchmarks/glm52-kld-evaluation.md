# GLM-5.2 KLD Evaluation

This page documents the current GLM-5.2 KLD workflow used for the v14
Blackwell/vLLM runs. It is separate from the older Qwen/SGLang KLD page because
GLM-5.2 uses vLLM prompt-logit capture, GLM-5.2 BF16 reference tensors, and the
78-character GLM-5.2 IndexCache pattern.

Status as of 2026-07-08:

- The current BF16 prompt-logit reference is available locally under
  `/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref`.
- The current BF16 prompt-logit reference is published as the Hugging Face
  dataset `festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708`.
- The dataset was uploaded on 2026-07-08 at commit
  `a8fbe8a277394e838c75190a0ab376625dfb1393`.
- The older 2026-06-18 reference dataset is historical and should not be used
  for new GLM-5.2 v14 comparisons unless the run explicitly says so.

## Reference Data

The current prefill comparison uses:

```text
/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref/logits_0.safetensors
```

The current published dataset contains prompt logits only. The older decode
reference, kept for historical teacher-forced decode KLD/JS checks, is:

```text
/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors
/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors.json
```

Reference generation details:

| Field | Value |
|---|---|
| Source model | `zai-org/GLM-5.2` |
| Local BF16 snapshot | `/root/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/4d67f66cc64d3219133b767c253b2ad1425c6c88` |
| Generation date | 2026-07-08 00:04 UTC |
| Dataset | `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test` |
| Prefill shape | tensor key `logits`, shape `[2047, 154880]`, dtype `float32` |
| Decode shape | historical 2026-06-18 decode file only |
| Context | 2048 tokens |
| Stride | 512 |
| Windows | 1 |
| Attention backend | `B12X_MLA_SPARSE` |
| KV cache dtype during capture | `fp8` |
| GLM-5.2 IndexCache override | `{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}` |

The safetensor key is named `logits` for compatibility. Some fallback capture
paths save full-vocab log-probabilities under that key. The v14 KLD runner calls
`log_softmax()` on both reference and candidate tensors, so it is compatible
with either raw logits or already-normalized log-probabilities.

The KLD direction is `KL(BF16 reference || candidate model)`.

## Hugging Face Upload

The current upload helper is:

```text
models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/upload_glm52_current_bf16_ref_to_hf.py
```

It targets this dataset:

```text
https://huggingface.co/datasets/festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708
```

Current HF file layout:

```text
README.md
reference-logits/logits_0.safetensors
reference-logits/manifest.json
generation-log/config.env
generation-log/scoremode_kld.log
```

Verified HF state on 2026-07-08:

```text
repo: festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708
sha:  a8fbe8a277394e838c75190a0ab376625dfb1393
```

To republish or repair the dataset, login with a write token and run:

```bash
python3 models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/upload_glm52_current_bf16_ref_to_hf.py \
  --repo-id festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708
```

A fresh machine can restore the expected local layout with:

```bash
mkdir -p /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z

huggingface-cli download festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708 \
  --repo-type dataset \
  --local-dir /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/hf

ln -sfn \
  /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/hf/reference-logits \
  /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref
```

## Current v14 Keypoint Run

The current reproducible v14 prefill KLD script is:

```text
scripts/bench-glm52-v14-kld-keypoints.sh
```

Default image:

```text
voipmonitor/vllm:eldritch-enlightenment-v5-vllmcd272c7-b12xe44cb77-cu132-20260707
```

Default checkpoints:

```text
LUKE_MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
MXFP4_MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts
PREFILL_REF=/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref
```

The script runs five repeats by default and compares these six key cases:

| Case | Checkpoint | Quantization | MoE mode | Online quant |
|---|---|---|---|---|
| `luke-a4-orig` | Luke NVFP4 | `modelopt_fp4` | A4 / checkpoint default | none |
| `luke-a16-orig` | Luke NVFP4 | `modelopt_fp4` | force A16 | none |
| `luke-a4-online-mxfp8` | Luke NVFP4 | `modelopt_fp4` | A4 / checkpoint default | `{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}` |
| `luke-a16-online-mxfp8` | Luke NVFP4 | `modelopt_fp4` | force A16 | `{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}` |
| `mxfp4-a8-orig` | BF16 dense + MXFP4 experts | `mxfp4` | force A8 | none |
| `mxfp4-a8-online-mxfp8` | BF16 dense + MXFP4 experts | `mxfp4` | force A8 | `{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}` |

Run all cases:

```bash
cd /root/rtx6kpro
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)" RUNS=5 \
  ./scripts/bench-glm52-v14-kld-keypoints.sh all
```

Summarize an existing run:

```bash
cd /root/rtx6kpro
KLD_ROOT=/root/kld/glm52_v14_keypoints_20260707Tkeypoints-v5 \
  ./scripts/bench-glm52-v14-kld-keypoints.sh summarize
```

The latest run path is written to:

```text
/root/kld/latest_glm52_v14_keypoints.out
```

Each case writes:

```text
config.json
prefill_dcp1.log
summary.json
```

The run root writes:

```text
summary.md
aggregate_summary.json
```

## r28 shared-H EXL3 results

The 2026-08-04 release gate evaluates the complete shared-H checkpoint, not a
single-layer POC:

| Field | Value |
|---|---|
| Checkpoint | `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` |
| Revision | `ae68c65947efa90bea37308e15421872f124c46d` |
| Image | `voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28` |
| Reference | `festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708@a8fbe8a277394e838c75190a0ab376625dfb1393` |
| KLD direction | `KL(BF16 reference || candidate)` |
| Geometry | TP4, DCP1, MTP off, eager |
| Input | fixed 2,048-token WikiText2 window; 2,047 scored positions |
| Repeats | 3 per profile |

The exact runner is
[`scripts/glm52_exl3_shared_h_kld.py`](../scripts/glm52_exl3_shared_h_kld.py),
invoked by
[`scripts/bench-glm52-exl3-shared-h-kld.sh`](../scripts/bench-glm52-exl3-shared-h-kld.sh).
The fixed token input is
[`benchmarks/data/glm52-kld-tokens-2048.json`](data/glm52-kld-tokens-2048.json).
Their SHA-256 values are:

```text
a4f4532c6687372e71471c3df24c3ba7fb05e03648d89a479d1197e8f4bf9e3e  scripts/bench-glm52-exl3-shared-h-kld.sh
cb618edbf5bc863d7561c87e9f781c5e743de8c9fd9587cf7a0008836427c0d1  scripts/glm52_exl3_shared_h_kld.py
d0be87a4909ad00c311e36fc5c81c7d6942216c16a8fbbbf77778edb273346f1  benchmarks/data/glm52-kld-tokens-2048.json
```

Lower KLD is better:

| Profile | Online quantization | KV format | Runs | Mean KLD | Run SD | Position SD |
|---|---|---|---:|---:|---:|---:|
| Checkpoint only | none | FP8, matched to reference | 3 | **0.074145973** | 0 | 0.292925745 |
| K6 isolation | cached online K6 | FP8, matched to reference | 3 | **0.077949159** | 0 | 0.331558079 |
| KV isolation | none | NVFP4 MLA | 1 | **0.107971445** | - | 0.432494372 |
| Release default | cached online K6 | NVFP4 MLA | 3 | **0.108828284** | 0 | 0.424119890 |

`checkpoint only` means the serialized K3/K4 expert weights are evaluated as
published and all remaining BF16 layers stay BF16; no online weight conversion
is applied. FP8 KV is used because the BF16 reference was captured with FP8 KV,
making this the closest checkpoint-quality comparison available from that
reference.

The release-default row intentionally measures the deployed path, including
online K6 and NVFP4 MLA KV. With FP8 KV held constant, K6 adds 0.003803186 mean
KLD, about 5.13% relative. With NVFP4 KV held constant, it adds 0.000856839,
about 0.79% relative. KLD is not additive, so these two interaction-dependent
deltas must not be summed. The KLD process loaded 84.08 GiB/GPU without online
quantization and 79.47 GiB/GPU with K6.

Serving capacity was measured separately because the KLD runner deliberately
caps its own cache at 0.5 GiB. These values use TP4/DCP1/MTP0, GMU 0.95,
`MAX_NUM_SEQS=1`, graph cap 6, and otherwise identical r28 settings:

| Online mode | KV format | Model load/GPU | KV budget/GPU | Logical KV tokens | 131k request fits |
|---|---|---:|---:|---:|---|
| none | FP8 | 83.85 GiB | 3.64 GiB | **72,448** | no |
| K6 | FP8 | 79.25 GiB | 8.19 GiB | **163,072** | yes |
| none | NVFP4 MLA | 83.85 GiB | 3.64 GiB | **107,136** | no |
| K6 | NVFP4 MLA | 79.25 GiB | 8.19 GiB | **241,216** | yes |

The no-online rows were booted with `MAX_MODEL_LEN=65536` to expose their exact
allocated block count. An otherwise identical 131k FP8 run was rejected after
profiling and reported the same 72,448-token maximum. `Logical KV tokens` is
total serving capacity, not a per-request context guarantee when requests run
concurrently.

After restoring the reference data as shown above and downloading the pinned
checkpoint under `/root/models`, reproduce the three repeated rows with:

```bash
cd /root/rtx6kpro

MODE=checkpoint GPUS=0,1,2,3 REPEATS=3 \
  MODEL=/root/models/GLM-5.2-EXL3-TR3-3.42bpw \
  ./scripts/bench-glm52-exl3-shared-h-kld.sh

MODE=runtime-fp8 GPUS=0,1,2,3 REPEATS=3 \
  MODEL=/root/models/GLM-5.2-EXL3-TR3-3.42bpw \
  ./scripts/bench-glm52-exl3-shared-h-kld.sh

MODE=runtime GPUS=0,1,2,3 REPEATS=3 \
  MODEL=/root/models/GLM-5.2-EXL3-TR3-3.42bpw \
  ./scripts/bench-glm52-exl3-shared-h-kld.sh
```

Each output directory contains `summary.json`, a run log, and one
`kld_positions_runN.safetensors` file per repeat. The runner fails on token
identity mismatches, missing full-vocabulary logits, or any non-finite value;
it does not silently publish a partial comparison.

## Current v14 Keypoint Results

Latest local run against the 2026-07-08 BF16 reference:

```text
/root/kld/glm52_v14_keypoints_current_bf16ref_20260708T0027Z
```

| Case | Quantization | MoE mode | Online MXFP8 | Runs | KLD mean +/- sd | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| `luke-a16-online-mxfp8` | `modelopt_fp4` | A16 | yes | 5 | 0.06587 +/- 0.00253 | 0.06288 | 0.06921 |
| `luke-a16-orig` | `modelopt_fp4` | A16 | no | 5 | 0.05994 +/- 0.00129 | 0.05844 | 0.06167 |
| `luke-a4-online-mxfp8` | `modelopt_fp4` | A4 | yes | 5 | 0.10800 +/- 0.00697 | 0.09941 | 0.11877 |
| `luke-a4-orig` | `modelopt_fp4` | A4 | no | 5 | 0.10228 +/- 0.00634 | 0.09368 | 0.11098 |
| `mxfp4-a8-online-mxfp8` | `mxfp4` | A8 | yes | 5 | 0.08030 +/- 0.00309 | 0.07818 | 0.08568 |
| `mxfp4-a8-orig` | `mxfp4` | A8 | no | 5 | 0.08160 +/- 0.00432 | 0.07460 | 0.08597 |

The weight-only GGUF/NVFP4/MXFP4 audit that uses the same 2026-07-08 reference
logits is documented separately:

```text
models/glm5.2/glm52-gguf-bf16-dequant-kld-2026-07-08.md
```

## Regenerating The BF16 References

The historical generation helper is:

```text
/root/kld/run_glm52_kld_pipeline_20260618.sh
```

Reference-only generation command:

```bash
ACTION=refs \
OUT_ROOT=/root/kld/glm52_kld_refs_$(date -u +%Y%m%dT%H%M%SZ) \
PREFILL_REF=/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref \
DECODE_REF=/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors \
TP=16 \
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  /root/kld/run_glm52_kld_pipeline_20260618.sh
```

Use the regenerated tensors only after checking that:

- the prefill safetensor has key `logits`, shape `[2047, 154880]`;
- the decode safetensor has keys `logprobs`, `prompt_token_ids`,
  `generated_token_ids`;
- the IndexCache pattern length is exactly 78;
- the model source is still `zai-org/GLM-5.2` and not a quantized checkpoint.

Quick local inspection:

```bash
python3 - <<'PY'
from safetensors import safe_open
for path in [
    "/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref/logits_0.safetensors",
    "/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors",
]:
    print(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(" ", key, tuple(tensor.shape), tensor.dtype)
PY
```

Expected output:

```text
/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref/logits_0.safetensors
  logits (2047, 154880) torch.float32
/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors
  generated_token_ids (17,) torch.int64
  logprobs (17, 154880) torch.float32
  prompt_token_ids (2048,) torch.int64
```

## Notes

- Do not compare GLM-5.2 candidates against the older GLM-5.1 or Qwen
  references. The vocabulary, tokenizer, prompt trace, and model family differ.
- The v14 keypoint script is a prefill KLD workflow. The decode reference exists
  for teacher-forced decode checks, but it is not used by
  `scripts/bench-glm52-v14-kld-keypoints.sh`.
- Keep MTP out of KLD unless explicitly testing decode teacher forcing. The
  prefill KLD script loads the model offline and compares prompt-position
  distributions.
- The 78-character `index_topk_pattern` is mandatory for GLM-5.2. The script
  exits if the pattern is truncated.
- For online MXFP8 quantization, keep `kv_b_proj` ignored. It was intentionally
  excluded from the v14 keypoint config because quantizing that path hurts KLD.
