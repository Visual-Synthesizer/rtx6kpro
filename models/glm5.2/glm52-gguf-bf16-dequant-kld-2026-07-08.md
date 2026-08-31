# GLM-5.2 GGUF to BF16 Dequant KLD Audit

Measured on 2026-07-08 on the local 16x RTX PRO 6000 Blackwell host.

This page documents the exact procedure used to compare GLM-5.2 4-bit weight
formats by converting or emulating their quantized weights back to BF16 and then
measuring prompt-logit KLD against the same BF16 GLM-5.2 reference logits.

The point of this run is to remove activation-kernel effects from the comparison:
the candidate models are loaded as BF16 weights after dequantization/emulation, and
the KLD is measured with the same vLLM prompt-logit path.

Lower KLD is better.

## Summary

| Variant | Mean KLD | Delta vs Unsloth UD-Q4_K_XL |
|---|---:|---:|
| Luke NVFP4 -> BF16 | 0.063780469 | -0.003956938 better |
| Unsloth UD-Q4_K_XL -> BF16 | 0.067737407 | baseline |
| AMD MXFP4 experts -> BF16 | 0.078153106 | +0.010415698 worse |

Interpretation:

- This is evidence for these exact checkpoints, this exact GLM-5.2 BF16
  reference, and this exact one-window WikiText-2 KLD setup.
- It does not claim that every NVFP4 quant is better than every GGUF quant.
- The measured conclusion is narrower: Luke's GLM-5.2 NVFP4 checkpoint has lower
  weight-only BF16-emulation KLD than Unsloth's GLM-5.2 `UD-Q4_K_XL` GGUF in
  this setup.

## Published Reference Logits

The BF16 reference prompt logits are published here:

```text
https://huggingface.co/datasets/festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708
```

Verified upload:

```text
repo: festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708
sha:  a8fbe8a277394e838c75190a0ab376625dfb1393
```

Dataset layout:

```text
README.md
reference-logits/logits_0.safetensors
reference-logits/manifest.json
generation-log/config.env
generation-log/scoremode_kld.log
```

Reference metadata:

| Field | Value |
|---|---|
| Source model | `zai-org/GLM-5.2` |
| Source snapshot | `4d67f66cc64d3219133b767c253b2ad1425c6c88` |
| Local source path | `/root/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/4d67f66cc64d3219133b767c253b2ad1425c6c88` |
| Generation time | 2026-07-08 00:04 UTC |
| Docker image | `voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707` |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, split `test` |
| Context length | 2048 |
| Stride | 512 |
| Windows | 1 |
| Tensor key | `logits` |
| Tensor shape | `[2047, 154880]` |
| Tensor dtype | `float32` |
| Attention backend | `B12X_MLA_SPARSE` |
| KV cache dtype during capture | `fp8` |
| TP / DCP | `16 / 1` |

Restore the local reference path used by the commands below:

```bash
mkdir -p /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z

huggingface-cli download festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708 \
  --repo-type dataset \
  --local-dir /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/hf

ln -sfn \
  /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/hf/reference-logits \
  /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref
```

Quick inspection:

```bash
python3 - <<'PY'
from safetensors import safe_open
path = "/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref/logits_0.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(key, tuple(tensor.shape), tensor.dtype)
PY
```

Expected output:

```text
logits (2047, 154880) torch.float32
```

The upload helper used for this dataset is
[upload_glm52_current_bf16_ref_to_hf.py](gguf-bf16-kld-2026-07-08/scripts/upload_glm52_current_bf16_ref_to_hf.py).

## Inputs

### Unsloth GGUF

Source:

```text
https://huggingface.co/unsloth/GLM-5.2-GGUF/tree/main/UD-Q4_K_XL
```

Local path used:

```text
/root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL
```

Files:

```text
GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf
...
GLM-5.2-UD-Q4_K_XL-00011-of-00011.gguf
```

Local size:

```text
436G
```

The GGUF tensor types present in this checkpoint were:

| GGUF type | Tensor count |
|---|---:|
| `Q8_0` | 872 |
| `F32` | 709 |
| `Q4_K` | 150 |
| `Q5_K` | 74 |
| `Q6_K` | 4 |

### Luke NVFP4

Local path used:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

### AMD MXFP4 Experts

Local path used:

```text
/root/models/GLM-5.2-BF16-AMDMXFP4experts
```

## Scripts And Overlays

This page uses the scripts and overlay files stored next to it:

```text
models/glm5.2/gguf-bf16-kld-2026-07-08/
```

Key scripts:

- [gguf_glm52_to_bf16_safetensors.py](gguf-bf16-kld-2026-07-08/scripts/gguf_glm52_to_bf16_safetensors.py)
- [check_gguf_dequant_against_ggml.py](gguf-bf16-kld-2026-07-08/scripts/check_gguf_dequant_against_ggml.py)
- [run_glm52_scoremode_kld_probe.sh](gguf-bf16-kld-2026-07-08/scripts/run_glm52_scoremode_kld_probe.sh)
- [prefill_kld_fallback.py](gguf-bf16-kld-2026-07-08/scripts/prefill_kld_fallback.py)
- [collect_prefill_return_logits_ref.py](gguf-bf16-kld-2026-07-08/scripts/collect_prefill_return_logits_ref.py)

The `overlays/` directory contains the small runtime overrides used for:

- loading the converted Unsloth BF16 checkpoint with GLM-5.2 FSS IndexCache;
- NVFP4 weight-only BF16 emulation;
- MXFP4 weight-only BF16 emulation.

For the commands below, set:

```bash
export RTX6KPRO=/root/rtx6kpro
export REPRO="${RTX6KPRO}/models/glm5.2/gguf-bf16-kld-2026-07-08"
export RUNNER="${REPRO}/scripts/run_glm52_scoremode_kld_probe.sh"
export REF=/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref
export IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
```

The GLM-5.2 IndexCache pattern used by all runs is exactly 78 characters:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

## GGUF To BF16 Conversion

The conversion script reads every GGUF tensor, dequantizes it to BF16, and writes
a Hugging Face/vLLM-style sharded safetensors checkpoint.

Important GLM-5.2 mappings:

- `token_embd.weight` -> `model.embed_tokens.weight`
- `output.weight` -> `lm_head.weight`
- `output_norm.weight` -> `model.norm.weight`
- `blk.N.attn_k_b.weight` + `blk.N.attn_v_b.weight` -> `model.layers.N.self_attn.kv_b_proj.weight`
- `blk.N.ffn_{gate,up,down}_exps.weight` -> split into `model.layers.N.mlp.experts.*.{gate,up,down}_proj.weight`
- GLM indexer tensors map to the matching `self_attn.indexer.*` HF names.

Dry run:

```bash
PYTHONPATH=/cache/kld_pydeps \
python3 "${REPRO}/scripts/gguf_glm52_to_bf16_safetensors.py" \
  --gguf-dir /root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL \
  --out-dir /root/models/GLM-5.2-Unsloth-UD-Q4_K_XL-BF16-dequant \
  --config-dir /root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL \
  --dry-run
```

Full conversion:

```bash
PYTHONPATH=/cache/kld_pydeps \
python3 "${REPRO}/scripts/gguf_glm52_to_bf16_safetensors.py" \
  --gguf-dir /root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL \
  --out-dir /root/models/GLM-5.2-Unsloth-UD-Q4_K_XL-BF16-dequant \
  --config-dir /root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL
```

Output checkpoint:

```text
/root/models/GLM-5.2-Unsloth-UD-Q4_K_XL-BF16-dequant
files: 1737
payload tensors written: 59870
model shard payload size: 1507728278016 bytes
du size: 1.4T
```

## GGUF Dequant Math Check

To verify that Python `gguf.dequantize(data, weight_type)` matches upstream
GGML/llama.cpp for the GGUF quant types in the Unsloth checkpoint, run:

```bash
PYTHONPATH=/cache/kld_pydeps \
python3 "${REPRO}/scripts/check_gguf_dequant_against_ggml.py" \
  --gguf-dir /root/models/unsloth-GLM-5.2-GGUF-UD-Q4_K_XL/UD-Q4_K_XL \
  --llama-cpp /tmp/llama.cpp-ggml-check \
  --llama-ref a646006f09d2f76f2d62d6c0d5e8e8490d570720 \
  --output-json /tmp/glm52_gguf_vs_ggml.json
```

The check compiled a tiny C helper against official `ggml-quants.c` from:

```text
https://github.com/ggml-org/llama.cpp
commit: a646006f09d2f76f2d62d6c0d5e8e8490d570720
```

It sampled real start/middle/end blocks from the actual GLM-5.2 GGUF tensors.

| Type | Sample tensor | Result |
|---|---|---|
| `Q8_0` | `blk.0.attn_k_b.weight` | bit-identical, `max_abs_diff=0`, `nonzero_diffs=0` |
| `Q4_K` | `blk.10.ffn_gate_exps.weight` | bit-identical, `max_abs_diff=0`, `nonzero_diffs=0` |
| `Q5_K` | `blk.10.ffn_down_exps.weight` | bit-identical, `max_abs_diff=0`, `nonzero_diffs=0` |
| `Q6_K` | `blk.8.ffn_down_exps.weight` | bit-identical, `max_abs_diff=0`, `nonzero_diffs=0` |

This verifies the block dequantization math. It does not by itself prove that
every model tensor was mapped to the correct HF/vLLM name; that is why the
converter mapping is documented above.

## KLD Commands

All KLD runs below use:

```text
dataset: Salesforce/wikitext, wikitext-2-raw-v1, split=test
context: 2048
stride: 512
windows: 1
direction: KL(BF16 reference || candidate)
reference: /root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref/logits_0.safetensors
```

### Unsloth UD-Q4_K_XL Dequantized To BF16

```bash
export OUT_ROOT=/root/kld/glm52_unsloth_udq4kxl_bf16dequant_current_bf16ref_repro_$(date -u +%Y%m%dT%H%M%SZ)
export MODEL=/root/models/GLM-5.2-Unsloth-UD-Q4_K_XL-BF16-dequant
export OVERLAY_DEEPSEEK_V2_PY="${REPRO}/overlays/glm52-kld/deepseek_v2.py"
export GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TP=16
export CPU_OFFLOAD_GB=32
export LOAD_FORMAT=safetensors
export QUANTIZATION=
export MOE_BACKEND=auto
export PROBE_RUNNER=fallback
export FALLBACK_QUANTIZATION=auto
"${RUNNER}"
```

Recorded final run:

```text
/root/kld/glm52_unsloth_udq4kxl_bf16dequant_current_bf16ref_20260708T154412Z/scoremode_kld.log
Mean KLD: 0.06773740716226687
Total positions: 2047
```

### Luke NVFP4 Weight-Only BF16 Emulation

```bash
export OUT_ROOT=/root/kld/glm52_nvfp4_weightonly_dequant_current_bf16ref_repro_$(date -u +%Y%m%dT%H%M%SZ)
export MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
export OVERLAY_NVFP4_ORACLE_PY="${REPRO}/overlays/nvfp4-weightonly/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
export OVERLAY_NVFP4_EMULATION_MOE_PY="${REPRO}/overlays/nvfp4-weightonly/vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py"
export OVERLAY_MODELOPT_PY="${REPRO}/overlays/nvfp4-weightonly/vllm/model_executor/layers/quantization/modelopt.py"
export GPU_DEVICES=0,1,2,3,4,5,6,7
export TP=8
export GPU_MEMORY_UTILIZATION=0.74
export LOAD_FORMAT=fastsafetensors
export QUANTIZATION=modelopt_fp4
export MOE_BACKEND=emulation
export PROBE_RUNNER=fallback
export FALLBACK_QUANTIZATION=modelopt_fp4
"${RUNNER}"
```

Recorded final run:

```text
/root/kld/glm52_nvfp4_weightonly_dequant_current_bf16ref_20260708T065557Z/scoremode_kld.log
Mean KLD: 0.063780468930438
Total positions: 2047
```

Four repeat runs in
`/root/kld/glm52_nvfp4_weightonly_dequant_current_bf16ref_repeats_20260708T065926Z`
reproduced the same rounded `0.063780` value.

### AMD MXFP4 Experts Weight-Only BF16 Emulation

```bash
export OUT_ROOT=/root/kld/glm52_mxfp4_weightonly_dequant_current_bf16ref_repro_$(date -u +%Y%m%dT%H%M%SZ)
export MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts
export OVERLAY_MXFP4_ORACLE_PY="${REPRO}/overlays/mxfp4-weightonly/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
export OVERLAY_OCP_MX_EMULATION_MOE_PY="${REPRO}/overlays/mxfp4-weightonly/vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py"
export GPU_DEVICES=0,1,2,3,4,5,6,7
export TP=8
export GPU_MEMORY_UTILIZATION=0.74
export LOAD_FORMAT=fastsafetensors
export QUANTIZATION=mxfp4
export MOE_BACKEND=emulation
export PROBE_RUNNER=fallback
export FALLBACK_QUANTIZATION=mxfp4
"${RUNNER}"
```

Recorded final run:

```text
/root/kld/glm52_mxfp4_weightonly_dequant_current_bf16ref_20260708T063650Z/scoremode_kld.log
Mean KLD: 0.0781531055547974
Total positions: 2047
```

Four repeat runs in
`/root/kld/glm52_mxfp4_weightonly_dequant_current_bf16ref_repeats_20260708T064022Z`
reproduced the same rounded `0.078153` value.

## Notes On QuantTrio

The QuantTrio checkpoint was also rerun against the same 2026-07-08 BF16
reference logits. It remains far outside the weight-only 4-bit comparison above:

| Run | Mean KLD |
|---|---:|
| `/root/kld/glm52_quanttrio_int4_int8mix_vs_current_bf16ref_20260708T074238Z` | 1.114485572 |
| `/root/kld/glm52_quanttrio_int4_int8mix_vs_current_bf16ref_rerun_20260708T164015Z` | 1.077431473 |

This is recorded as a separate suspicious result, not as part of the Unsloth vs
NVFP4 vs MXFP4 weight-only table.

## Review Checklist

To independently review the result:

1. Download the BF16 reference logits from the HF dataset above.
2. Rebuild the Unsloth BF16 checkpoint with
   [gguf_glm52_to_bf16_safetensors.py](gguf-bf16-kld-2026-07-08/scripts/gguf_glm52_to_bf16_safetensors.py).
3. Run [check_gguf_dequant_against_ggml.py](gguf-bf16-kld-2026-07-08/scripts/check_gguf_dequant_against_ggml.py)
   to verify GGUF block dequantization against GGML.
4. Run the three KLD commands above against the same reference logits.
5. If a result differs, first inspect tensor-name mapping and the K/V B
   projection merge. The raw GGUF block dequant math has already matched GGML
   bit-for-bit for all quant types present in this checkpoint.
